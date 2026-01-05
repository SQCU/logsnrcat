# Scripts Directory - Eval Server Testing Patterns

## Eval Server Execution Model

The eval server at `http://{host}:{port}/eval` accepts POST requests with `{"code": "..."}` payloads. Key behavior:

### Async Continuation After HTTP Timeout

When you submit code that runs longer than the HTTP timeout (default 30s), the server returns `{'type': 'Timeout', 'message': 'Execution exceeded 30s'}` **but continues executing the code**. The computation runs to completion server-side.

This means:
- HTTP timeout != execution abort
- Long-running training loops complete even if you get a timeout response
- Results are available by querying `ctx.*` attributes after execution finishes

### Polling Pattern for Long-Running Code

```python
# Submit long-running code (will timeout but continue server-side)
result = requests.post(url, json={"code": training_loop_code}, timeout=60)
# result may be timeout error - that's fine

# Wait for completion
import time
time.sleep(expected_duration)

# Fetch results - they exist because execution continued
result = requests.post(url, json={"code": "ctx._my_result"}, timeout=10)
print(result.json()['result'])
```

### Server State Persistence

The eval server maintains state in `ctx` across requests:
- `ctx.device` - CUDA device
- `ctx.iterator` - Data iterator
- `model` - Loaded model
- `batch` - Pre-generated test batch

Your code can store results for later retrieval:
```python
# In submitted code
ctx._my_results = {"metric": value, "history": [...]}

# Later fetch
result = eval_code("ctx._my_results", host, port)
```

## Writing Tests for the Eval Server

### 1. Always Store Results in ctx

The eval endpoint returns `"executed"` for statements, only returning values for expressions. Store results explicitly:

```python
# BAD - returns "executed", loses the dict
code = '{"a": 1, "b": compute_something()}'

# GOOD - store then fetch
code = 'ctx._result = {"a": 1, "b": compute_something()}'
eval_code(code, host, port)
result = eval_code("ctx._result", host, port)['result']
```

### 2. Sanity Checks Before Training

Always verify your intervention produces measurable change before running long experiments:

```python
# Verify adapter changes output BEFORE training
sanity_code = '''
with torch.no_grad():
    baseline_out = model(x)
    adapted_out = model_with_adapter(x)
    mse_diff = F.mse_loss(baseline_out, adapted_out).item()
ctx._sanity = {"mse_diff": mse_diff, "is_nonzero": mse_diff > 1e-10}
'''
eval_code(sanity_code, host, port)
sanity = eval_code("ctx._sanity", host, port)['result']
if not sanity['is_nonzero']:
    raise ValueError("Adapter produces no change!")
```

### 3. Numerical Verification Over Visual Inspection

Don't trust visual inspection of plots alone. Always compute and log numerical metrics:

```python
# In training loop, at each save point:
mse_base = F.mse_loss(recon_base, images).item()
mse_policy = F.mse_loss(recon_policy, images).item()
mse_diff = F.mse_loss(recon_base, recon_policy).item()
print(f"MSE base={mse_base:.6f}, policy={mse_policy:.6f}, diff={mse_diff:.6f}")
```

If `mse_diff` is ~0, your intervention isn't working regardless of what plots show.

### 4. Run IDs for Output Files

Always include run identifiers in output filenames to prevent overwriting and confusion:

```python
import datetime
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# or pass --run-id explicitly

plt.savefig(f"{output_dir}/experiment_{run_id}_step{step:03d}.png")
```

### 5. Server-Side vs Client-Side Execution

**Server-side (preferred for training):**
- Submit entire training loop as one code block
- All computation happens on GPU without network round-trips
- ~100% GPU utilization possible

**Client-side (per-step HTTP calls):**
- Each training step requires network round-trip
- ~10-15% GPU utilization due to network bottleneck
- Use only for interactive debugging

```python
# SERVER-SIDE: Submit entire loop
training_code = f'''
for step in range({n_steps}):
    # ... all training logic ...
ctx._result = history
'''
eval_code(training_code, host, port, timeout=1200)

# CLIENT-SIDE: Per-step calls (slow, avoid for training)
for step in range(n_steps):
    eval_code(single_step_code, host, port)  # network bottleneck each step
```

## Common Pitfalls

1. **Stale results**: Previous runs leave `ctx._*` attributes. Check timestamps or use unique run_ids.

2. **FX/dynamo conflicts**: Calling `ae.encode()` multiple times in traced code can fail. Use `torch.no_grad()` contexts.

3. **F-string escaping**: When embedding f-strings in f-strings, use `{{` and `}}` for literal braces in the inner string.

4. **Timeout != failure**: A timeout response means HTTP gave up waiting, not that execution failed.
