#!/usr/bin/env python3
"""
patch_torch_headers.py - Fix PyTorch/CUTLASS MSVC compilation bugs

PyTorch 2.7+ has header bugs that break CUDA extension compilation on Windows:
1. Unqualified `std::` in compiled_autograd.h causes C2872 "ambiguous symbol"
2. CUTLASS doesn't detect C++17 because MSVC lies about __cplusplus

This script patches installed PyTorch headers and sets up environment for builds.

Usage:
    python patch_torch_headers.py           # Patch headers
    python patch_torch_headers.py --check   # Check if patches needed
    python patch_torch_headers.py --revert  # Revert patches (if backup exists)
"""

import re
import os
import sys
import shutil
import argparse
from pathlib import Path


def find_torch_include() -> Path:
    """Find PyTorch include directory."""
    try:
        import torch
        torch_dir = Path(torch.__file__).parent
        include_dir = torch_dir / "include"
        if include_dir.exists():
            return include_dir
        # Some installations put it in site-packages directly
        include_dir = torch_dir.parent / "torch" / "include"
        if include_dir.exists():
            return include_dir
    except ImportError:
        pass

    # Fallback: check common locations
    candidates = [
        Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "include",
        Path(sys.prefix) / "lib" / "python3.10" / "site-packages" / "torch" / "include",
        Path.home() / ".local" / "lib" / "python3.10" / "site-packages" / "torch" / "include",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Could not find PyTorch include directory")


def patch_std_namespace(content: str) -> str:
    """
    Replace unqualified std:: with ::std:: to fix MSVC ambiguity.

    Careful not to double-qualify already-fixed references.
    """
    # Pattern: std:: not preceded by :: (word boundary or start of token)
    # Negative lookbehind for :: ensures we don't match already-qualified ::std::
    pattern = r'(?<!:)\bstd::'
    replacement = '::std::'

    return re.sub(pattern, replacement, content)


def patch_file(filepath: Path, dry_run: bool = False) -> tuple[bool, int]:
    """
    Patch a single file. Returns (changed, num_replacements).
    """
    if not filepath.exists():
        return False, 0

    content = filepath.read_text(encoding='utf-8', errors='replace')
    original = content

    patched = patch_std_namespace(content)

    if patched == original:
        return False, 0

    # Count changes
    num_changes = len(re.findall(r'(?<!:)\bstd::', original))

    if not dry_run:
        # Backup original
        backup = filepath.with_suffix(filepath.suffix + '.orig')
        if not backup.exists():
            shutil.copy2(filepath, backup)

        filepath.write_text(patched, encoding='utf-8')

    return True, num_changes


def patch_compiled_autograd(include_dir: Path, dry_run: bool = False) -> dict:
    """Patch compiled_autograd.h and related headers."""
    results = {}

    # Files known to have the std:: ambiguity issue
    targets = [
        "torch/csrc/dynamo/compiled_autograd.h",
        "torch/csrc/dynamo/cache_entry.h",
        "torch/csrc/autograd/saved_variable.h",
        "ATen/core/ivalue.h",
        "ATen/core/ivalue_inl.h",
    ]

    for target in targets:
        filepath = include_dir / target
        if filepath.exists():
            changed, count = patch_file(filepath, dry_run)
            results[target] = {"changed": changed, "replacements": count}

    return results


def create_cutlass_env_shim() -> str:
    """
    Return environment setup commands for CUTLASS C++17 detection fix.

    MSVC needs /Zc:__cplusplus to report correct __cplusplus value,
    otherwise CUTLASS macros don't expose C++17 features.
    """
    # These get passed to cl.exe and nvcc
    cxx_flags = "/std:c++17 /Zc:__cplusplus"
    nvcc_flags = f"-Xcompiler \"{cxx_flags}\""

    if sys.platform == "win32":
        return f'''
# CUTLASS C++17 detection fix for MSVC
# Add these to your environment before building CUDA extensions:
set CXXFLAGS={cxx_flags}
set NVCC_APPEND_FLAGS={nvcc_flags}
set CL=/std:c++17 /Zc:__cplusplus
'''
    else:
        return f'''
# CUTLASS C++17 detection fix (Linux - usually not needed)
export CXXFLAGS="-std=c++17"
'''


def setup_build_env():
    """Set environment variables for the current process."""
    if sys.platform == "win32":
        os.environ["CXXFLAGS"] = os.environ.get("CXXFLAGS", "") + " /std:c++17 /Zc:__cplusplus"
        os.environ["CL"] = os.environ.get("CL", "") + " /std:c++17 /Zc:__cplusplus"
        # For nvcc to pass to host compiler
        os.environ["NVCC_APPEND_FLAGS"] = os.environ.get("NVCC_APPEND_FLAGS", "") + ' -Xcompiler "/Zc:__cplusplus"'
        os.environ["DISTUTILS_USE_SDK"] = "1"  # Tell setuptools we're using SDK


def revert_patches(include_dir: Path) -> dict:
    """Revert patches by restoring .orig backups."""
    results = {}

    for orig_file in include_dir.rglob("*.orig"):
        target = orig_file.with_suffix("")  # Remove .orig suffix
        if orig_file.exists():
            shutil.copy2(orig_file, target)
            orig_file.unlink()
            results[str(target.relative_to(include_dir))] = "reverted"

    return results


def main():
    parser = argparse.ArgumentParser(description="Patch PyTorch headers for MSVC compatibility")
    parser.add_argument("--check", action="store_true", help="Check if patches needed (dry run)")
    parser.add_argument("--revert", action="store_true", help="Revert patches from backups")
    parser.add_argument("--env", action="store_true", help="Print environment setup for CUTLASS fix")
    parser.add_argument("--setup-env", action="store_true", help="Set env vars for current process")
    args = parser.parse_args()

    if args.env:
        print(create_cutlass_env_shim())
        return

    if args.setup_env:
        setup_build_env()
        print("Environment configured for CUDA extension builds")
        for key in ["CXXFLAGS", "CL", "NVCC_APPEND_FLAGS", "DISTUTILS_USE_SDK"]:
            print(f"  {key}={os.environ.get(key, '')}")
        return

    try:
        include_dir = find_torch_include()
        print(f"Found PyTorch include: {include_dir}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if args.revert:
        results = revert_patches(include_dir)
        if results:
            print("Reverted patches:")
            for path, status in results.items():
                print(f"  {path}: {status}")
        else:
            print("No backup files found to revert")
        return

    # Patch headers
    print(f"\n{'Checking' if args.check else 'Patching'} PyTorch headers for std:: ambiguity fix...")
    results = patch_compiled_autograd(include_dir, dry_run=args.check)

    total_changes = 0
    for target, info in results.items():
        if info["changed"]:
            status = "needs patching" if args.check else "patched"
            print(f"  {target}: {status} ({info['replacements']} replacements)")
            total_changes += info["replacements"]
        else:
            print(f"  {target}: OK")

    if total_changes == 0:
        print("\nNo patches needed - headers already fixed or not present")
    elif args.check:
        print(f"\n{total_changes} total replacements needed")
        print("Run without --check to apply patches")
    else:
        print(f"\n{total_changes} total replacements applied")
        print("Backups saved with .orig extension")

    # Print CUTLASS env reminder
    print("\n" + "=" * 60)
    print("CUTLASS C++17 Fix:")
    print("=" * 60)
    print(create_cutlass_env_shim())


if __name__ == "__main__":
    main()
