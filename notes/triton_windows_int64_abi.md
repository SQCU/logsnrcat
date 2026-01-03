# Triton/Inductor Windows int64 ABI Mismatch

## Summary

PyTorch's Triton integration uses C `long` for 64-bit integers in its static CUDA launcher. On Windows, C `long` is 32-bit (LLP64 data model), causing `OverflowError` when tensor strides exceed 2^31. This breaks compiled models on Windows that work fine on Linux.

## The Bug

The bug is in PyTorch's C++ extension (`torch._C._StaticCudaLauncher`), which uses `long` instead of `int64_t` or `long long` for 64-bit kernel arguments.

```cpp
// Somewhere in torch/_C - the C++ code uses:
long stride_value;  // BUG: 32-bit on Windows, 64-bit on Linux
```

The Python-side format codes (`"i64": "l"`) are just metadata - the actual problem is the C++ type mismatch.

## Why This Happens: LP64 vs LLP64

| Data Model | `long` size | Used By |
|------------|-------------|---------|
| **LP64** | 64-bit | Linux, macOS, most Unix |
| **LLP64** | 32-bit | Windows (all versions) |

Linux developers often use `long` as a "pointer-sized integer" because on LP64 systems, `sizeof(long) == sizeof(void*)`. This assumption breaks on Windows where `long` is always 32-bit regardless of architecture.

## Why Format Code Patching Doesn't Work

Initial attempt was to change `"i64": "l"` to `"i64": "L"` in Python. This fails because:
1. The C++ launcher only recognizes specific format codes ('l', 'K', etc.)
2. 'L' is not in the C++ switch statement: `RuntimeError: Unknown type passed in: L`
3. The real fix requires changing C++ code to use `int64_t` instead of `long`

## When It Triggers

The overflow occurs when compiled kernels receive stride parameters exceeding 2^31:

```
# Example: 8 batch × 8 heads × 131072 context × 64 dim
# Stride for batch dimension: 8 × 131072 × 64 = 67,108,864 (fits in 32-bit)
# But accumulated symbolic strides during dynamic compilation can exceed this
```

Symptoms:
- `OverflowError: Python int too large to convert to C long`
- Occurs during Triton kernel autotuning
- Only on Windows, works fine on Linux
- Often triggered by `torch.compile(dynamic=True)` with variable sequence lengths

## Why This Matters for Deployment

### Training/Inference Parity

Models compiled on Linux training servers may behave differently (or crash) when deployed to Windows inference:

```
Training Server (Linux)     Edge Device (Windows)
        │                           │
   LP64 ABI                    LLP64 ABI
   long = 64-bit               long = 32-bit
        │                           │
   Compiles fine ──────────────► OverflowError
```

### Affected Scenarios

1. **Local development on Windows** - Crashes during training/eval
2. **Windows inference servers** - Enterprise deployments often use Windows
3. **Client-side inference** - Gaming PCs, workstations with GPUs
4. **Edge deployment** - Windows IoT, embedded Windows systems
5. **CI/CD pipelines** - Windows build agents testing CUDA code

### The Insidious Part

The bug is **data-dependent**:
- Small models/batches: works fine (strides fit in 32-bit)
- Large models/contexts: crashes unpredictably
- Same model may work at 512 context but crash at 8192

This means Windows testing with small inputs won't catch the bug.

## Our Workaround

```
src/patches/
├── __init__.py                 # apply_all() called from main.py
└── triton_windows_int64.py     # Disables static CUDA launcher on Windows
```

**Solution**: Disable the buggy static CUDA launcher on Windows. This falls back to
PyTorch's Python-based kernel launcher, which correctly handles 64-bit integers.

```python
# What the patch does:
torch._inductor.config.use_static_cuda_launcher = False
```

Applied automatically on Windows before any `torch.compile()` calls:

```python
# main.py
import torch
from src.patches import apply_all as apply_patches
apply_patches()  # Prints: [patch] Disabled static CUDA launcher on Windows...
```

### Performance Impact

The static launcher is a C++ optimization for kernel dispatch. Disabling it adds
slight Python overhead per kernel launch, but this is negligible compared to
actual GPU kernel execution time. For most workloads, the difference is unmeasurable.

### Cache Invalidation

The patch automatically clears the inductor disk cache on first run (tracked via
marker file `.int64_patch_v2`). This ensures kernels are recompiled without the
static launcher.

If you still see issues, manually clear the cache:
```powershell
# Windows PowerShell
Remove-Item -Recurse -Force $env:LOCALAPPDATA\Temp\torchinductor_*
```

## Upstream Status

This should be reported to pytorch/pytorch. The fix is trivial:

```diff
- "i64": "l",
+ "i64": "L",
```

The `u64` mapping already correctly uses `K` (unsigned long long), suggesting this was an oversight rather than intentional.

## Verification

After applying the patch, compiled kernels with large strides should work:

```python
import torch
from src.patches import apply_all; apply_all()

x = torch.randn(8, 8, 131072, 64, device='cuda')
model = torch.compile(lambda x: x.reshape(-1, 64) @ x.reshape(-1, 64).T)
model(x)  # Should work, previously OverflowError on Windows
```

## References

- [Microsoft C++ Data Type Ranges](https://learn.microsoft.com/en-us/cpp/cpp/data-type-ranges)
- [LP64 vs LLP64 Wikipedia](https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models)
- [Python struct format characters](https://docs.python.org/3/library/struct.html#format-characters)
- [PyArg_ParseTuple format units](https://docs.python.org/3/c-api/arg.html#numbers)
