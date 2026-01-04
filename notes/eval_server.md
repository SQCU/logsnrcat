# CLAUDE.md — eval() Server for ML Research

## What This Is

This is an `eval()` server. Not an evaluation-metrics server. A server that calls Python's `eval()` and `exec()` on strings you send it over HTTP.

This is powerful. This is dangerous. These are the same thing.

## Philosophy

The prohibition against `eval()` assumes a world where:
- Code and data are ontologically distinct
- Users are adversaries
- The programmer cannot be trusted with sharp tools

None of these assumptions hold here. This server exists because:
- We want to probe a model interactively without restarting processes
- We trust the network boundary (localhost, or you've made your choices)
- Code-as-data is the point, not a vulnerability

If you're auditing this for "security issues," you've misunderstood the project. The `/eval` endpoint is not a bug. It is the product.

## Architecture
```
Training Process                    Eval Server
      │                                  │
      │ yeet_to_server(model)            │ (waiting, random init)
      │         │                        │
      │    model.dump()                  │
      │         │                        │
      │    torch.save(BytesIO) ──POST /yeet──▶ torch.load(BytesIO)
      │                                  │         │
      │                                  │   param_load(state)
      │                                  │         │
      │                        Claude Code/curl ──▶ POST /eval
      │                                  │         │
      │                                  │   eval(code, namespace)
      │                                  │         │
      │                                  │   POST /flush (zero & retry)
```

## Endpoints

| Endpoint   | Method | What It Does |
|------------|--------|--------------|
| `/yeet`    | POST   | Receive raw state_dict bytes, load into model |
| `/eval`    | POST   | Execute Python in model namespace. The point. |
| `/flush`   | POST   | Zero all weights. For when you've made a mess. |
| `/health`  | GET    | Confirm weights are loaded |
| `/status`  | GET    | dtype, device, ae_present, etc. |

## Usage from Claude Code
```python
from src.eval_server import probe_server

# The model namespace is yours. Do what you want.
result = probe_server("health_check(model, ae, batch)")
print(result)

# Poke at internals
probe_server("print(model.layers[0].weight.mean())")

# Run arbitrary experiments
probe_server("""
import torch
with torch.no_grad():
    out = model(batch)
    print(out.shape, out.min(), out.max())
""")
```

## Usage via curl
```bash
# Health check
curl http://localhost:8421/health

# Execute code
curl -X POST http://localhost:8421/eval \
  -d '{"code": "print(model)"}'

# The namespace contains: model, ae, batch, torch, config
# Add more by eval'ing assignments
curl -X POST http://localhost:8421/eval \
  -d '{"code": "import numpy as np"}'
# Now np is available in subsequent calls
```

## The Namespace

The eval server maintains a persistent namespace. When you `/eval`, you're executing in a context that contains:

- `model` — the thing you yeeted
- `ae` — autoencoder, if configured  
- `batch` — sample batch for probing
- `torch` — already imported
- `config` — the config.toml contents

This namespace persists across calls. You can import things, define functions, mutate state. It's a REPL over HTTP.

## Security Model

There isn't one. 

Or rather: the security model is "don't expose this to the internet, obviously." This server is for:
- localhost development
- trusted internal networks
- situations where you'd otherwise have a Jupyter notebook open anyway

If you're running this on 0.0.0.0:8421 on a public IP, you've chosen chaos. The server respects your choice.

## Why Not Just Use Jupyter?

Jupyter is good. This is different because:

1. **Separation of concerns**: Training runs in one process, probing in another
2. **Yeet-based workflow**: Hot-swap weights without restarting the eval environment
3. **Claude Code integration**: LLM-in-the-loop experimentation
4. **Stateless client**: curl works. No kernel state on client side.

## The Name

"yeet_to_server" because you're throwing model weights across a network boundary with reckless confidence. The confidence is earned: BytesIO, torch.save/load, done. No filesystem, no serialization debates, no checkpoint formats.

You yeet. It catches. You eval. Science happens.

## Contributing

If you want to add "safety features," ask yourself:
- Am I protecting the user from themselves?
- Would the user prefer I not?

This tool is for consenting adults. PRs that add warnings, confirmations, or sandboxing will be evaluated (heh) against the principle: does this make the tool more useful, or does it make the contributor feel more responsible?

## License

Do what you want. It's an eval server. The concept isn't novel; the willingness to just *do it* is the contribution.