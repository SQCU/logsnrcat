"""
Ephemeral Eval Server for Interactive Model Probing

Network-yeet model weights directly from training - NO FILESYSTEM PERSISTENCE.
Exposes eval() endpoint for arbitrary Python diagnostics.

Architecture:
    Training Loop                          Eval Server
    ┌─────────────┐                       ┌─────────────┐
    │ model.dump()│ ─── POST /yeet ──────▶│ param_load()│
    └─────────────┘    (raw bytes)        └──────┬──────┘
                                                 │
                                          POST /eval
                                                 │
                                          ┌──────▼──────┐
                                          │ Claude Code │
                                          └─────────────┘

Usage:
    # Start server (builds model architecture, waits for weights)
    python -m src.eval_server --config configs/exp.toml --port 8421

    # From training code - yeet weights over network
    from src.eval_server import yeet_to_server
    yeet_to_server(model, 'http://localhost:8421')

    # Then probe interactively
    curl -X POST http://localhost:8421/eval -d '{"code": "health_check(model, ae, batch)"}'
"""

import argparse
import json
import traceback
import io
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional
import threading
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Client-side: Yeet weights from training to server
# ============================================================================

def yeet_to_server(
    model: nn.Module,
    server_url: str = 'http://localhost:8421',
    run_id: Optional[str] = None,
    run_path: Optional[str] = None,
    step: Optional[int] = None,
    extra: Optional[Dict] = None
) -> bool:
    """
    Network-yeet model weights to eval server. No filesystem involved.

    Args:
        model: Model to yeet weights from
        server_url: Eval server URL
        run_id: Run identifier (e.g., 'main_run_093')
        run_path: Path to run output directory
        step: Training step
        extra: Additional metadata dict

    Usage in training loop:
        if step % eval_interval == 0:
            yeet_to_server(model, 'http://localhost:8421',
                          run_id='main_run_093',
                          run_path='experiments_swiglu_ae/main_run_093',
                          step=step)
    """
    import urllib.request

    # Dump to bytes
    state = model.dump() if hasattr(model, 'dump') else model.state_dict()

    # Strip _orig_mod prefix from torch.compile() wrapped modules
    # Training model is compiled, eval server model isn't
    cleaned_state = {}
    for k, v in state.items():
        # _orig_mod appears when module is wrapped by torch.compile
        clean_key = k.replace('._orig_mod', '')
        cleaned_state[clean_key] = v
    state = cleaned_state

    buffer = io.BytesIO()
    torch.save(state, buffer)
    data = buffer.getvalue()

    # Yeet weights
    req = urllib.request.Request(
        f"{server_url}/yeet",
        data=data,
        headers={'Content-Type': 'application/octet-stream'}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            if not result.get('success'):
                print(f"[yeet] Failed: {result.get('error')}")
                return False
            print(f"[yeet] Sent {len(data):,} bytes to {server_url}")
    except Exception as e:
        print(f"[yeet] Error: {e}")
        return False

    # Send provenance if provided
    if run_id or run_path or step is not None:
        provenance_data = {
            'run_id': run_id or 'unknown',
            'run_path': run_path or '',
            'step': step,
            'extra': extra
        }
        try:
            prov_req = urllib.request.Request(
                f"{server_url}/provenance",
                data=json.dumps(provenance_data).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(prov_req, timeout=5) as resp:
                prov_result = json.loads(resp.read().decode('utf-8'))
                if prov_result.get('success'):
                    print(f"[yeet] Provenance set: {run_id} step={step}")
        except Exception as e:
            print(f"[yeet] Warning: failed to set provenance: {e}")

    return True


def probe_server(code: str, server_url: str = 'http://localhost:8421') -> Any:
    """
    Send probe code to eval server and get result.

    Usage:
        result = probe_server("health_check(model, ae, batch)")
        print(result)
    """
    import urllib.request

    req = urllib.request.Request(
        f"{server_url}/eval",
        data=json.dumps({'code': code}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode('utf-8'))


def query_health(server_url: str = 'http://localhost:8421') -> Dict[str, Any]:
    """
    Query eval server health. Returns status dict or error.

    Usage in training:
        health = query_health('http://localhost:8421')
        if health.get('weights_loaded'):
            print("Model yeet'd successfully!")
    """
    import urllib.request

    try:
        with urllib.request.urlopen(f"{server_url}/health", timeout=5) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        return {'status': 'error', 'error': str(e), 'weights_loaded': False}


# ============================================================================
# Server-side: EvalContext
# ============================================================================

class EvalContext:
    """
    Holds model architecture, config, and test data generators.
    Weights are yeet'd in via /yeet endpoint.

    This is an eval() REPL - send arbitrary Python code to /eval.
    The server's job is to:
    1. Receive weights and know where they came from (provenance)
    2. Execute arbitrary code against the loaded model
    3. Return detailed errors when things fail

    Key attributes:
    - ctx.provenance - dict with run_id, run_path, step of current weights
    - ctx.run_id, ctx.run_path - shortcuts to provenance fields
    - ctx.weights_loaded - whether weights have been received
    """

    def __init__(self, config_path: str, device: str = 'cuda', checkpoint_path: Optional[str] = None):
        self.device = device
        self.dtype = torch.bfloat16
        self.weights_loaded = False
        self.deps_loaded = False  # Track if plotting deps are loaded
        self._plot_deps = {}  # Cache loaded plotting dependencies
        self._provenance = None  # Provenance of current weights

        # Load config
        from src.config import load_config, sanitize_config
        self.config = sanitize_config(load_config(config_path))
        self.config['device'] = device

        # Determine dtype from config
        dtype_str = self.config['training']['precision']
        dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
        self.dtype = dtype_map[dtype_str]
        self.use_amp = self.dtype in (torch.bfloat16, torch.float16)

        # Build model architecture (random init - weights come via /yeet)
        from main import build_model
        self.model = build_model(self.config, device).to(dtype=self.dtype)

        # Extract AE reference
        self.ae = getattr(self.model, 'sparse_ae', None)

        # Optionally load checkpoint if provided (for backward compat)
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        # Build data iterator for test batches
        from src.data_iterator import CompositeIterator
        self.iterator = CompositeIterator(
            self.device,
            config=self.config['dataset_mix'],
            caching_resolution=self.config['training']['bucketing']['caching_resolution']
        )

        print(f"Model architecture ready: {sum(p.numel() for p in self.model.parameters()):,} params")
        print(f"AE present: {self.ae is not None}")
        print(f"Dtype: {self.dtype}, AMP: {self.use_amp}")
        if not checkpoint_path:
            print(f"Waiting for weights via POST /yeet ...")

    def _load_checkpoint(self, path: str):
        """Load from filesystem (backward compat)."""
        print(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        self.model.param_load(state) if hasattr(self.model, 'param_load') else self.model.load_state_dict(state)
        self.weights_loaded = True
        print(f"Weights loaded from filesystem")

    def receive_weights(self, data: bytes) -> Dict[str, Any]:
        """Receive yeet'd weights from network."""
        try:
            buffer = io.BytesIO(data)
            state = torch.load(buffer, map_location=self.device, weights_only=False)

            # Use param_load if available (respects model's loading semantics)
            if hasattr(self.model, 'param_load'):
                self.model.param_load(state)
            else:
                self.model.load_state_dict(state)

            self.weights_loaded = True
            self.ae = getattr(self.model, 'sparse_ae', None)  # Re-extract in case structure changed

            return {'success': True, 'params_loaded': len(state)}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_batch(self, resolution: int = 64, batch_size: int = 4, source: str = 'fractal_main') -> torch.Tensor:
        """Generate a batch of images at specified resolution.

        Uses generate_from_split() to directly request the target resolution,
        avoiding the wasteful over-generate + filter anti-pattern.

        Args:
            resolution: Target image resolution (pixels)
            batch_size: Number of images to generate
            source: Data source split name. Default 'fractal_main' supports any resolution.
                   Use 'sprite_atlas' for pixel art, 'checker_baseline' for geometric.
        """
        blocks = self.iterator.generate_from_split(source, count=batch_size, resolution=resolution)
        if len(blocks) < batch_size:
            raise ValueError(f"Only generated {len(blocks)} blocks (wanted {batch_size}) from {source}")
        return torch.stack([b.content for b in blocks[:batch_size]]).to(self.device)

    def autocast(self):
        """Return autocast context matching training."""
        return torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=self.use_amp)

    def load_deps(self) -> Dict[str, Any]:
        """Load plotting and visualization dependencies into eval namespace.

        Call via POST /load_deps or probe_server('ctx.load_deps()').
        Once loaded, these are available in the eval namespace without re-importing.

        Returns dict of loaded modules/functions for confirmation.
        """
        if self.deps_loaded:
            return {'success': True, 'message': 'Dependencies already loaded', 'deps': list(self._plot_deps.keys())}

        loaded = {}

        # Core visualization
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            loaded['plt'] = plt
            loaded['matplotlib'] = matplotlib
        except ImportError as e:
            loaded['matplotlib_error'] = str(e)

        try:
            import numpy as np
            loaded['np'] = np
        except ImportError as e:
            loaded['numpy_error'] = str(e)

        # Project plotting utilities
        try:
            from src.plotting import (
                plot_multimetric_analysis, plot_ae_roundtrip,
                plot_loss_schedule_analysis, ExperimentLogger
            )
            loaded['plot_multimetric_analysis'] = plot_multimetric_analysis
            loaded['plot_ae_roundtrip'] = plot_ae_roundtrip
            loaded['plot_loss_schedule_analysis'] = plot_loss_schedule_analysis
            loaded['ExperimentLogger'] = ExperimentLogger
        except ImportError as e:
            loaded['plotting_error'] = str(e)

        # Data pipeline for generating test batches
        try:
            from src.data_functional import (
                generate_checkerboard_query, render_checkerboard,
                generate_torus_query, render_torus
            )
            loaded['generate_checkerboard_query'] = generate_checkerboard_query
            loaded['render_checkerboard'] = render_checkerboard
            loaded['generate_torus_query'] = generate_torus_query
            loaded['render_torus'] = render_torus
        except ImportError as e:
            loaded['data_functional_error'] = str(e)

        # Image utilities
        try:
            from torchvision.utils import make_grid, save_image
            loaded['make_grid'] = make_grid
            loaded['save_image'] = save_image
        except ImportError as e:
            loaded['torchvision_error'] = str(e)

        self._plot_deps = loaded
        self.deps_loaded = True

        dep_names = [k for k in loaded.keys() if not k.endswith('_error')]
        errors = [k for k in loaded.keys() if k.endswith('_error')]

        return {
            'success': True,
            'loaded': dep_names,
            'errors': errors,
            'message': f'Loaded {len(dep_names)} dependencies'
        }

    # ========================================================================
    # Run Provenance - track where weights came from
    # ========================================================================

    def set_provenance(self, run_id: str, run_path: str, step: Optional[int] = None,
                       extra: Optional[Dict] = None):
        """
        Record provenance of current weights. Called by training code when yeeting.

        Args:
            run_id: Run identifier (e.g., 'main_run_093')
            run_path: Full path to run directory
            step: Training step when weights were captured
            extra: Any additional metadata
        """
        from datetime import datetime
        self._provenance = {
            'run_id': run_id,
            'run_path': run_path,
            'step': step,
            'received_at': datetime.now().isoformat(),
            'extra': extra or {}
        }

    @property
    def provenance(self) -> Optional[Dict]:
        """Get provenance of current weights, if known."""
        return getattr(self, '_provenance', None)

    @property
    def run_path(self) -> Optional[str]:
        """Shortcut to get current run path from provenance."""
        prov = self.provenance
        return prov['run_path'] if prov else None

    @property
    def run_id(self) -> Optional[str]:
        """Shortcut to get current run ID from provenance."""
        prov = self.provenance
        return prov['run_id'] if prov else None

    def build_namespace(self) -> Dict[str, Any]:
        """Build the namespace dict for eval()."""
        # Pre-generate batch for convenience
        try:
            batch = self.get_batch(resolution=64, batch_size=4)
        except Exception:
            batch = None

        namespace = {
            # Core objects
            'model': self.model,
            'ae': self.ae,
            'config': self.config,
            'batch': batch,
            'ctx': self,

            # Convenience functions
            'get_batch': self.get_batch,
            'autocast': self.autocast,
            'load_deps': self.load_deps,

            # PyTorch
            'torch': torch,
            'nn': nn,
            'F': F,

            # Diagnostic functions (from CLAUDE.md)
            'activation_variance_sweep': activation_variance_sweep,
            'batch_cosine_similarity': batch_cosine_similarity,
            'codebook_usage': codebook_usage,
            'per_level_importance': per_level_importance,
            'effective_dim': effective_dim,
            'gradient_norms': gradient_norms,
            'health_check': health_check,
        }

        # Include plot deps if loaded
        if self.deps_loaded:
            namespace.update(self._plot_deps)

        return namespace


# ============================================================================
# Diagnostic Functions (copied from CLAUDE.md for standalone operation)
# ============================================================================

def activation_variance_sweep(model, x):
    """Per-layer activation variance. Healthy ≈ 1, dead → 0 or ∞."""
    variances = {}
    hooks = []
    def capture(name):
        def hook(m, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            variances[name] = out.var().item()
        return hook
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(capture(name)))
    try:
        with torch.no_grad():
            model(x)
    finally:
        [h.remove() for h in hooks]
    return variances


def batch_cosine_similarity(hidden_states):
    """Batch diversity check. Mode collapse → mean ≈ 1."""
    flat = F.normalize(hidden_states.flatten(1), dim=-1)
    sim = flat @ flat.T
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    off_diag = sim[mask]
    return {'mean': off_diag.mean().item(), 'std': off_diag.std().item()}


def codebook_usage(ae, images):
    """Codebook utilization per level."""
    if ae is None:
        return []
    codes_list = ae.encode(images)
    return [{
        'level': i,
        'unique_codes': codes.flatten(0, 1).unique(dim=0).shape[0],
        'sparsity': (codes.abs() < 1e-6).float().mean().item(),
    } for i, codes in enumerate(codes_list)]


def per_level_importance(ae, images):
    """Reconstruction importance per level (ablation study)."""
    if ae is None:
        return []
    codes = ae.encode(images)
    full_recon = ae.decode(codes)
    base_mse = F.mse_loss(full_recon, images).item()
    importance = []
    for level in range(len(codes)):
        ablated = [c if i != level else torch.zeros_like(c) for i, c in enumerate(codes)]
        ablated_mse = F.mse_loss(ae.decode(ablated), images).item()
        importance.append(ablated_mse - base_mse)
    return importance


def effective_dim(hidden_states, threshold=0.99):
    """Effective dimensionality via SVD."""
    centered = hidden_states.flatten(0, -2) - hidden_states.flatten(0, -2).mean(0)
    S = torch.linalg.svdvals(centered.float())
    cumvar = (S**2).cumsum(0) / (S**2).sum()
    return (cumvar < threshold).sum().item() + 1


def gradient_norms(model, loss):
    """Gradient norm per layer. Requires a loss tensor."""
    loss.backward(retain_graph=True)
    norms = {name: p.grad.norm().item()
             for name, p in model.named_parameters()
             if p.grad is not None}
    model.zero_grad()
    return norms


def health_check(model, ae, batch):
    """One-shot health report. First thing to run on any failing model."""
    results = {
        'output_range': (batch.min().item(), batch.max().item()),
        'output_variance': batch.var().item(),
    }

    if ae is not None:
        with torch.no_grad():
            recon = ae.decode(ae.encode(batch))
        results['roundtrip_mse'] = F.mse_loss(recon, batch).item()
        results['codebook_usage'] = codebook_usage(ae, batch)
        results['batch_diversity'] = batch_cosine_similarity(recon)

    # Skip activation sweep if model forward is complex
    # results['activation_vars'] = activation_variance_sweep(model, batch)

    return results


# ============================================================================
# HTTP Server
# ============================================================================

class EvalHandler(BaseHTTPRequestHandler):
    """HTTP handler for eval and yeet requests."""

    context: EvalContext = None  # Set by server
    timeout: int = 30

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])

        if self.path == '/yeet':
            # Receive raw bytes of state_dict
            data = self.rfile.read(content_length)
            result = self.context.receive_weights(data)
            self._send_json(result)

        elif self.path == '/eval':
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                request = json.loads(body)
                code = request.get('code', '')
            except json.JSONDecodeError:
                code = body

            # Warn if no weights loaded
            if not self.context.weights_loaded:
                result = {'success': False, 'error': 'No weights loaded. POST /yeet first.'}
            else:
                result = self.execute_with_timeout(code)

            self._send_json(result)

        elif self.path == '/flush':
            # Zero out model (for testing fresh yeets)
            if hasattr(self.context.model, 'flush'):
                self.context.model.flush()
            self.context.weights_loaded = False
            self.context._provenance = None
            self._send_json({'success': True, 'message': 'Model flushed'})

        elif self.path == '/provenance':
            # Set provenance metadata for current weights
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(body)
                self.context.set_provenance(
                    run_id=data.get('run_id', 'unknown'),
                    run_path=data.get('run_path', ''),
                    step=data.get('step'),
                    extra=data.get('extra')
                )
                self._send_json({'success': True, 'provenance': self.context.provenance})
            except json.JSONDecodeError as e:
                self._send_json({'success': False, 'error': f'Invalid JSON: {e}'})

        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == '/health':
            self._send_json({
                'status': 'ok',
                'weights_loaded': self.context.weights_loaded,
                'params': sum(p.numel() for p in self.context.model.parameters()),
            })
        elif self.path == '/status':
            self._send_json({
                'weights_loaded': self.context.weights_loaded,
                'dtype': str(self.context.dtype),
                'device': self.context.device,
                'ae_present': self.context.ae is not None,
                'deps_loaded': self.context.deps_loaded,
            })
        elif self.path == '/load_deps':
            # Load plotting/visualization dependencies into namespace
            result = self.context.load_deps()
            self._send_json(result)
        elif self.path == '/provenance':
            # Get provenance of current weights
            prov = self.context.provenance
            if prov:
                self._send_json(prov)
            else:
                self._send_json({'error': 'No provenance recorded. Weights may have been loaded without metadata.'})
        else:
            self.send_error(404)

    def _send_json(self, data: dict):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, default=str).encode('utf-8'))

    def execute_with_timeout(self, code: str) -> Dict[str, Any]:
        """Execute code with timeout. Returns detailed error info on failure."""
        result = {'success': False, 'result': None, 'error': None}

        def target():
            try:
                namespace = self.context.build_namespace()
                # Use exec for statements, eval for expressions
                try:
                    result['result'] = eval(code, namespace)
                    result['success'] = True
                except SyntaxError:
                    exec(code, namespace)
                    result['result'] = namespace.get('result', 'executed')
                    result['success'] = True
            except Exception as e:
                # Capture detailed exception info
                exc_type, exc_value, exc_tb = sys.exc_info()
                tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)

                # Extract the innermost frame's local variables (if available)
                locals_snapshot = {}
                if exc_tb:
                    # Walk to innermost frame
                    tb = exc_tb
                    while tb.tb_next:
                        tb = tb.tb_next
                    frame_locals = tb.tb_frame.f_locals
                    # Capture serializable locals (skip large objects)
                    for k, v in frame_locals.items():
                        if k.startswith('_'):
                            continue
                        try:
                            # Try to get a useful repr, skip if too large
                            r = repr(v)
                            if len(r) < 500:
                                locals_snapshot[k] = r
                        except Exception:
                            pass

                result['error'] = {
                    'type': type(e).__name__,
                    'message': str(e),
                    'traceback': ''.join(tb_lines),
                    'locals': locals_snapshot if locals_snapshot else None
                }

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            result['error'] = {'type': 'Timeout', 'message': f'Execution exceeded {self.timeout}s'}

        return result

    def log_message(self, format, *args):
        print(f"[eval-server] {args[0]}")


def run_server(context: EvalContext, port: int = 8421, timeout: int = 30):
    """Run the eval server."""
    EvalHandler.context = context
    EvalHandler.timeout = timeout

    server = HTTPServer(('0.0.0.0', port), EvalHandler)
    print(f"\n{'='*60}")
    print(f"Eval server running on http://localhost:{port}")
    print(f"{'='*60}")
    print(f"\nEndpoints:")
    print(f"  POST /yeet       - Receive model weights (raw bytes)")
    print(f"  POST /eval       - Execute Python code (returns detailed errors)")
    print(f"  POST /provenance - Set run metadata: {{run_id, run_path, step, extra}}")
    print(f"  POST /flush      - Zero out model weights and provenance")
    print(f"  GET  /health     - Health check")
    print(f"  GET  /status     - Detailed status")
    print(f"  GET  /provenance - Get current weights provenance")
    print(f"  GET  /load_deps  - Load plotting/viz deps into namespace")
    print(f"\nFrom training code:")
    print(f"  from src.eval_server import yeet_to_server")
    print(f"  yeet_to_server(model, 'http://localhost:{port}',")
    print(f"                 run_id='main_run_093', run_path='experiments/...', step=1000)")
    print(f"\nNamespace: model, ae, config, batch, ctx, torch, F, nn")
    print(f"Context: ctx.provenance, ctx.run_id, ctx.run_path")
    print(f"Diagnostics: health_check, codebook_usage, per_level_importance, ...")
    print(f"\nPlotting deps: GET /load_deps → adds plt, np, make_grid, ...")
    print(f"{'='*60}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Ephemeral Eval Server - Network-yeet weights, no filesystem',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server, wait for weights via /yeet
  python -m src.eval_server -f configs/exp.toml

  # Start server with checkpoint (backward compat)
  python -m src.eval_server -f configs/exp.toml -c runs/exp/ckpt.pt

  # From training: yeet weights
  from src.eval_server import yeet_to_server
  yeet_to_server(model, 'http://localhost:8421')
"""
    )
    parser.add_argument('--config', '-f', required=True, help='Path to config TOML')
    parser.add_argument('--checkpoint', '-c', default=None, help='Optional checkpoint (or use /yeet)')
    parser.add_argument('--port', '-p', type=int, default=8421, help='Server port')
    parser.add_argument('--device', '-d', default='cuda', help='Device')
    parser.add_argument('--timeout', '-t', type=int, default=30, help='Eval timeout (seconds)')

    args = parser.parse_args()

    context = EvalContext(args.config, args.device, args.checkpoint)
    run_server(context, port=args.port, timeout=args.timeout)


if __name__ == '__main__':
    main()
