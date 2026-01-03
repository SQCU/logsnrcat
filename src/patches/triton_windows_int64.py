# src/patches/triton_windows_int64.py
"""
Workaround for Triton/Inductor int64 stride overflow on Windows.

BUG: torch._inductor.runtime.static_cuda_launcher uses C 'long' for i64 types.
On Windows, C long is 32-bit even on 64-bit systems (LLP64 data model),
causing OverflowError when tensor strides exceed 2^31.

The bug is in the C++ extension (torch._C._StaticCudaLauncher) which can't be
patched from Python. The C++ code uses 'long' instead of 'int64_t'/'long long'.

WORKAROUND: Disable the static CUDA launcher on Windows. This falls back to
the Python-based launcher which handles 64-bit integers correctly.

Performance impact: Slight overhead from Python dispatch vs C++ static launcher.
This is negligible compared to actual kernel execution time.

UPSTREAM: This should be reported to pytorch/pytorch. The C++ code needs to use
platform-independent 64-bit types (int64_t) instead of 'long'.
"""

import sys
import os


# Marker file to track if cache was invalidated for this patch version
_PATCH_VERSION = "v2"  # Bump this when the patch changes meaningfully

# Module-level flag to track if patch was applied this session
_applied = False


def _get_inductor_cache_dir():
    """Get the inductor disk cache directory."""
    import tempfile
    import getpass
    temp_dir = tempfile.gettempdir()
    try:
        username = getpass.getuser()
    except Exception:
        username = "user"
    return os.path.join(temp_dir, f"torchinductor_{username}")


def _get_patch_marker_path():
    """Get path to marker file that tracks cache invalidation."""
    cache_dir = _get_inductor_cache_dir()
    return os.path.join(cache_dir, f".int64_patch_{_PATCH_VERSION}")


def _needs_cache_clear():
    """Check if disk cache needs to be cleared for this patch version."""
    marker_path = _get_patch_marker_path()
    return not os.path.exists(marker_path)


def _mark_cache_cleared():
    """Create marker file to indicate cache was cleared for this patch version."""
    marker_path = _get_patch_marker_path()
    cache_dir = os.path.dirname(marker_path)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(marker_path, 'w') as f:
            f.write(f"Cache cleared for int64 patch {_PATCH_VERSION}\n")
        return True
    except Exception:
        return False


def _clear_disk_cache():
    """
    Clear inductor disk cache to force recompilation.

    Returns: (success: bool, message: str)
    """
    import shutil

    cache_dir = _get_inductor_cache_dir()

    if not os.path.exists(cache_dir):
        _mark_cache_cleared()
        return True, "no disk cache found"

    try:
        # Count items before clearing (excluding our marker)
        items = [f for f in os.listdir(cache_dir) if not f.startswith('.int64_patch')]
        item_count = len(items)

        # Remove all subdirectories (the actual cache)
        for item in items:
            item_path = os.path.join(cache_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)
            else:
                try:
                    os.remove(item_path)
                except Exception:
                    pass

        # Mark as cleared
        _mark_cache_cleared()

        return True, f"cleared {item_count} cached items"
    except Exception as e:
        return False, f"failed to clear cache: {e}"


def _invalidate_inductor_cache():
    """
    Invalidate inductor cache so kernels recompile without static launcher.

    Only clears disk cache once per patch version (uses marker file).

    Returns tuple: (success: bool, message: str)
    """
    # Clear in-memory caches (always, they're per-process anyway)
    try:
        from torch._inductor.utils import clear_caches
        clear_caches()
    except ImportError:
        pass

    try:
        from torch._inductor.async_compile import CompiledTritonKernels
        CompiledTritonKernels.cache_clear()
    except (ImportError, AttributeError):
        pass

    # Check if disk cache needs clearing
    if _needs_cache_clear():
        disk_success, disk_msg = _clear_disk_cache()
        return disk_success, disk_msg
    else:
        return True, "cache already cleared for this patch version"


def apply():
    """
    Disable static CUDA launcher on Windows to avoid int64 overflow.

    The static launcher's C++ code uses 'long' which is 32-bit on Windows.
    Disabling it falls back to Python-based launching which handles 64-bit correctly.

    Must be called before any torch.compile() or Triton kernel compilation.
    Safe to call multiple times (idempotent).
    """
    global _applied

    if sys.platform != 'win32':
        return  # Only needed on Windows

    # Check if already patched this session
    if _applied:
        return

    try:
        import torch._inductor.config as inductor_config
    except ImportError:
        # Older torch version or inductor not available
        return

    # Disable the static CUDA launcher - fall back to Python launcher
    # The Python launcher correctly handles 64-bit integers on all platforms
    inductor_config.use_static_cuda_launcher = False
    _applied = True

    # Invalidate cache so old kernels compiled with static launcher are rebuilt
    cache_success, cache_msg = _invalidate_inductor_cache()

    if cache_success:
        status_msg = f" ({cache_msg})"
    else:
        status_msg = f" (WARNING: {cache_msg})"

    print(f"[patch] Disabled static CUDA launcher on Windows (int64 ABI fix){status_msg}")
