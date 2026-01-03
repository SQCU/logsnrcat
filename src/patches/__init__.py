# src/patches/__init__.py
"""
Monkey patches for upstream library bugs.

These patches document defects in dependencies and should be removed
when the upstream issues are fixed. Each patch should reference the
relevant issue tracker or commit where the bug exists.
"""

from .triton_windows_int64 import apply as apply_triton_int64_patch

def apply_all():
    """Apply all patches. Call early in main.py before any torch imports."""
    apply_triton_int64_patch()
