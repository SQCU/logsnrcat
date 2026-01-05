# Logging Strategy for Remote Server Experiments

## The Problem

When submitting long-running code to the eval server:

1. **HTTP timeout != execution failure** - Server returns timeout after 30s but continues executing
2. **`print()` goes to server terminal** - Not visible to the client that submitted the code
3. **No visibility into progress** - Client can't tell if run is at step 10 or step 900
4. **Temptation to meddle** - Without feedback, natural to assume failure and try to "fix" things

## The Solution: LogTee

Tee stdout to both the server console AND a logfile that the client can read:

```python
import sys

class LogTee:
    """Tee stdout to both console and logfile for remote monitoring."""
    def __init__(self, logfile_path):
        self.logfile = open(logfile_path, "w", buffering=1)  # line-buffered
        self.stdout = sys.stdout

    def write(self, msg):
        self.stdout.write(msg)
        self.logfile.write(msg)
        self.logfile.flush()

    def flush(self):
        self.stdout.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()

# Setup
log_path = f"/tmp/experiment_{run_id}.log"
_log_tee = LogTee(log_path)
sys.stdout = _log_tee

# Cleanup (in finally block)
def _cleanup_tee():
    global _log_tee
    sys.stdout = _log_tee.stdout
    _log_tee.close()
```

## Usage Pattern

### In submitted code (server-side):

```python
try:
    # All print() calls go to both console and logfile
    print(f"Step {step}: loss={loss:.4f}")

    # ... long-running work ...

except Exception as e:
    print(f"!!! EXCEPTION !!!")
    print(traceback.format_exc())
    raise
finally:
    print(f"Run complete. Log at: {log_path}")
    _cleanup_tee()
```

### From client (Claude Code):

```python
# Check progress without interrupting the run
from pathlib import Path
log = Path("/tmp/reinforce_composite_002_scaled.log").read_text()
print(log[-2000:])  # Last 2000 chars
```

Or via Read tool:
```
Read("/tmp/reinforce_composite_002_scaled.log")
```

## Key Benefits

1. **Non-invasive monitoring** - Reading a logfile doesn't interrupt execution
2. **Exception capture** - Tracebacks are preserved in the log
3. **Progress visibility** - See step counts, metrics, timing
4. **Post-hoc debugging** - Full history available after completion
5. **GPU utilization correlation** - Can sample GPU util while checking log position

## Checking GPU Utilization

Complement log reading with utilization sampling to verify active computation:

```python
import subprocess
import time

samples = []
for _ in range(40):
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    samples.append(int(result.stdout.strip()))
    time.sleep(0.1)

print(f"Mean: {sum(samples)/len(samples):.1f}%  Min: {min(samples)}%  Max: {max(samples)}%")
```

High utilization + log progress = run is healthy.
Low utilization + stale log = run may have completed or crashed.

## File Locations

| Script | Log Path |
|--------|----------|
| `reinforce_composite.py` | `/tmp/reinforce_{run_id}.log` |
| `reinforce_subspace_ablation.py` | `/tmp/reinforce_subspace_{run_id}.log` |

## Anti-Patterns

**Don't:**
- Poll `ctx._result` repeatedly during execution (may interfere)
- Assume timeout = failure
- Try to reset dynamo mid-run
- Resubmit code while a run is active

**Do:**
- Read logfile to check progress
- Sample GPU utilization to verify activity
- Wait for completion before querying results
- Use unique run_ids to avoid log collision
