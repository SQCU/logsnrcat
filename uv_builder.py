#!/usr/bin/env python3
"""
uv_builder.py - Multi-stage dependency installer for projects with build-time dependencies

                        A MONUMENT TO POOR TOOLING DECISIONS

This script exists because:

1. Python's packaging ecosystem has mass-produced dozens of incompatible build systems,
   lockfile formats, and dependency resolvers over 30+ years, each claiming to be "the
   one that finally gets it right" while introducing novel incompatibilities.

2. uv, despite being genuinely fast and useful, resolves ALL dependencies (including
   optional ones) during lockfile generation. This means packages requiring CUDA torch
   at build time cannot be listed in pyproject.toml at all, because uv will try to
   build them before torch is installed.

3. CUDA extension packages must be built with --no-build-isolation to see the already-
   installed torch, but uv sync doesn't support this flag, forcing us into manual
   multi-stage pip invocations.

4. The Python packaging authorities have spent years bikeshedding PEPs about metadata
   formats while the actual hard problems (build ordering, native dependencies, cross-
   platform CUDA) remain unsolved.

The result: a janky Python script that shells out to uv multiple times because the
"modern" Python tooling cannot express "install A, then build B against A" in any
declarative format.

If you're reading this in 2030 and wondering why this exists: it's because the people
responsible for Python packaging prioritized theoretical purity over practical utility,
and we are all paying the price.

Usage:
    python uv_builder.py          # Full install
    python uv_builder.py --sync   # Just uv sync (stage 1)
    python uv_builder.py --post   # Just post-sync CUDA builds (stage 2)
"""

import subprocess
import sys
import os
import re
import platform
import shutil
from pathlib import Path

# Stage 2 packages: must be installed AFTER uv sync, with --no-build-isolation
# Format: (name, git_url, branch)
POST_SYNC_CUDA_PACKAGES = [
    ("grouped_gemm", "https://github.com/fanshiqing/grouped_gemm", "main"),
]

# Local build directory for patched grouped_gemm
GROUPED_GEMM_LOCAL_DIR = Path(__file__).parent / "_grouped_gemm_build"


def run(cmd: list[str], check: bool = True, env: dict | None = None, **kwargs) -> subprocess.CompletedProcess:
    """Run a command with output streaming."""
    print(f"\n>>> {' '.join(cmd)}\n")
    # Merge provided env with current process env
    if env is not None:
        full_env = os.environ.copy()
        full_env.update(env)
        return subprocess.run(cmd, check=check, env=full_env, **kwargs)
    return subprocess.run(cmd, check=check, **kwargs)


# =============================================================================
# STAGE 1.5: Patch PyTorch headers for MSVC compatibility
# =============================================================================
# PyTorch 2.7+ has two bugs that break CUDA extension builds on Windows:
#
# 1. Unqualified `std::` in compiled_autograd.h causes MSVC C2872 "ambiguous symbol"
#    because MSVC's <valarray> puts `std` in global namespace, conflicting with
#    PyTorch's own namespace pollution.
#
# 2. CUTLASS templates don't compile because MSVC lies about __cplusplus value
#    unless you pass /Zc:__cplusplus, which breaks CUTLASS's C++17 feature detection.
#
# These are THEIR bugs, not ours. We fix them here because nobody else will.
# =============================================================================

def find_torch_include() -> Path | None:
    """Find PyTorch include directory, or None if not installed yet."""
    try:
        import torch
        include_dir = Path(torch.__file__).parent / "include"
        if include_dir.exists():
            return include_dir
    except ImportError:
        pass
    return None


def patch_std_namespace(content: str) -> str:
    """
    Replace unqualified std:: with ::std:: to fix MSVC ambiguity.

    Also applies targeted fixes for MSVC template name lookup bugs where
    even ::std:: doesn't work due to namespace std {} extensions in headers.
    """
    # Negative lookbehind: don't match already-qualified ::std::
    content = re.sub(r'(?<!:)\bstd::', '::std::', content)

    return content


def patch_compiled_autograd_msvc_guard(content: str) -> str:
    """
    Fix PyTorch's Windows+CUDA guard to actually work.

    PyTorch has a guard that's supposed to skip the problematic if-constexpr chain
    on Windows+CUDA, but it requires USE_CUDA to be defined, which doesn't happen
    during extension builds.

    We change the guard from:
        #if defined(_WIN32) && (defined(USE_CUDA) || defined(USE_ROCM))
    to just:
        #if defined(_WIN32)

    This unconditionally skips the buggy template code on Windows.
    """
    old_guard = "#if defined(_WIN32) && (defined(USE_CUDA) || defined(USE_ROCM))"
    new_guard = "#if defined(_WIN32)  // Patched: removed USE_CUDA requirement (not defined during extension builds)"

    if old_guard in content:
        content = content.replace(old_guard, new_guard)

    return content


def patch_torch_headers(include_dir: Path) -> int:
    """
    Patch PyTorch headers for MSVC compatibility.
    Returns number of files patched.
    """
    targets = [
        "torch/csrc/dynamo/compiled_autograd.h",
        "torch/csrc/dynamo/cache_entry.h",
        "torch/csrc/autograd/saved_variable.h",
        "ATen/core/ivalue.h",
        "ATen/core/ivalue_inl.h",
    ]

    patched = 0
    for target in targets:
        filepath = include_dir / target
        if not filepath.exists():
            continue

        content = filepath.read_text(encoding='utf-8', errors='replace')
        original = content

        # Apply std:: -> ::std:: fix
        content = patch_std_namespace(content)

        # For compiled_autograd.h, also apply the MSVC guard around
        # the problematic packed_type template
        if target == "torch/csrc/dynamo/compiled_autograd.h":
            content = patch_compiled_autograd_msvc_guard(content)

        if content != original:
            # Backup if not already backed up
            backup = filepath.with_suffix(filepath.suffix + '.orig')
            if not backup.exists():
                shutil.copy2(filepath, backup)

            filepath.write_text(content, encoding='utf-8')
            num_changes = len(re.findall(r'(?<!:)\bstd::', original))
            extra = " + MSVC guard" if "MSVC guard" in target or target == "torch/csrc/dynamo/compiled_autograd.h" else ""
            print(f"  Patched {target} ({num_changes} std:: replacements{extra})")
            patched += 1

    return patched


# =============================================================================
# STAGE 2: grouped_gemm patching for Windows MSVC
# =============================================================================
# grouped_gemm needs three patches for Windows:
# 1. setup.py: Add MSVC flags (/Zc:__cplusplus, /std:c++17, /EHsc)
# 2. setup.py: Add cublas to libraries for linking
# 3. CUTLASS: Guard SM90 headers (MSVC template bugs)
# =============================================================================

def patch_grouped_gemm_setup(setup_py: Path) -> bool:
    """Patch grouped_gemm's setup.py for Windows MSVC compatibility."""
    if not setup_py.exists():
        return False

    content = setup_py.read_text(encoding='utf-8')
    original = content

    # Patch 1: Add MSVC nvcc flags after the nvcc_flags list
    if "IS_WINDOWS = sys.platform" not in content:
        old = '''nvcc_flags = [
    "-std=c++17",  # NOTE: CUTLASS requires c++17
    "-DENABLE_BF16",  # Enable BF16 for cuda_version >= 11
    # "-DENABLE_FP8",  # Enable FP8 for cuda_version >= 11.8
]'''
        new = '''nvcc_flags = [
    "-std=c++17",  # NOTE: CUTLASS requires c++17
    "-DENABLE_BF16",  # Enable BF16 for cuda_version >= 11
    # "-DENABLE_FP8",  # Enable FP8 for cuda_version >= 11.8
]

# MSVC-specific flags for Windows
# - /Zc:__cplusplus makes MSVC report correct __cplusplus value for C++17 detection
# - CUTLASS uses __cplusplus to conditionally expose C++17 features like SharedStorage
IS_WINDOWS = sys.platform == "win32"
if IS_WINDOWS:
    nvcc_flags.extend([
        '-Xcompiler', '/Zc:__cplusplus',  # Fix CUTLASS C++17 detection
        '-Xcompiler', '/std:c++17',        # Explicit C++17 for host compiler
    ])'''
        content = content.replace(old, new)

    # Patch 2: Add MSVC cxx_flags and cublas library
    old_ext = '''    ext_modules.append(
        CUDAExtension(
            "grouped_gemm_backend",
            [
                "csrc/ops.cu",
                "csrc/grouped_gemm.cu",
                "csrc/sinkhorn.cu",
                "csrc/permute.cu",
            ],
            include_dirs=[f"{cwd}/third_party/cutlass/include/", f"{cwd}/csrc"],
            extra_compile_args={
                "cxx": ["-fopenmp", "-fPIC", "-Wno-strict-aliasing"],
                "nvcc": nvcc_flags,
            },
        )
    )'''
    new_ext = '''    # Platform-specific C++ compiler flags
    if IS_WINDOWS:
        cxx_flags = ["/std:c++17", "/Zc:__cplusplus", "/EHsc"]
    else:
        cxx_flags = ["-fopenmp", "-fPIC", "-Wno-strict-aliasing"]

    ext_modules.append(
        CUDAExtension(
            "grouped_gemm_backend",
            [
                "csrc/ops.cu",
                "csrc/grouped_gemm.cu",
                "csrc/sinkhorn.cu",
                "csrc/permute.cu",
            ],
            include_dirs=[f"{cwd}/third_party/cutlass/include/", f"{cwd}/csrc"],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
            libraries=["cublas"],  # Required for SM90 fallback to cuBLAS
        )
    )'''
    content = content.replace(old_ext, new_ext)

    if content != original:
        setup_py.write_text(content, encoding='utf-8')
        return True
    return False


def patch_cutlass_sm90_guard(cutlass_dir: Path) -> bool:
    """Guard SM90 CUTLASS headers to skip MSVC-incompatible code."""
    gemm_universal = cutlass_dir / "include" / "cutlass" / "gemm" / "kernel" / "gemm_universal.hpp"
    if not gemm_universal.exists():
        return False

    content = gemm_universal.read_text(encoding='utf-8')
    original = content

    old = '''#include "cutlass/gemm/kernel/sm70_gemm.hpp"
#include "cutlass/gemm/kernel/sm90_gemm_tma.hpp"
#include "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp"
#include "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp"
#include "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp"'''

    new = '''#include "cutlass/gemm/kernel/sm70_gemm.hpp"
// SM90 (Hopper) headers have MSVC template bugs - skip on Windows
// We only build for SM89 anyway, so this doesn't affect functionality
#if !defined(_MSC_VER)
#include "cutlass/gemm/kernel/sm90_gemm_tma.hpp"
#include "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp"
#include "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp"
#include "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp"
#endif'''

    content = content.replace(old, new)

    if content != original:
        gemm_universal.write_text(content, encoding='utf-8')
        return True
    return False


def setup_grouped_gemm_local(uv: str) -> Path | None:
    """Clone and patch grouped_gemm for Windows build. Returns local path or None."""
    if platform.system() != "Windows":
        return None  # Only needed on Windows

    local_dir = GROUPED_GEMM_LOCAL_DIR

    # Clone if not exists
    if not local_dir.exists():
        print(f"  Cloning grouped_gemm to {local_dir}...")
        run(["git", "clone", "--depth", "1",
             "https://github.com/fanshiqing/grouped_gemm.git", str(local_dir)])
        run(["git", "-C", str(local_dir), "submodule", "update", "--init", "--recursive"])

    # Apply patches
    setup_py = local_dir / "setup.py"
    cutlass_dir = local_dir / "third_party" / "cutlass"

    if patch_grouped_gemm_setup(setup_py):
        print("  Patched grouped_gemm setup.py for MSVC")
    if patch_cutlass_sm90_guard(cutlass_dir):
        print("  Patched CUTLASS SM90 headers for MSVC")

    return local_dir


def find_windows_sdk_bin() -> str | None:
    """Find Windows SDK bin directory containing rc.exe."""
    sdk_base = Path(r"C:\Program Files (x86)\Windows Kits\10\bin")
    if not sdk_base.exists():
        return None

    # Find the latest SDK version
    versions = sorted([d for d in sdk_base.iterdir() if d.is_dir() and d.name.startswith("10.")], reverse=True)
    for version_dir in versions:
        rc_path = version_dir / "x64" / "rc.exe"
        if rc_path.exists():
            return str(version_dir / "x64")
    return None


def get_msvc_build_env() -> dict:
    """
    Get environment variables for CUTLASS C++17 detection on MSVC.

    MSVC doesn't update __cplusplus by default (reports 199711L even in C++17 mode).
    CUTLASS checks __cplusplus to decide whether to expose C++17 features like
    std::is_same_v. Without /Zc:__cplusplus, CUTLASS's SharedStorage types aren't
    properly declared, causing template errors.

    Also defines USE_CUDA to trigger PyTorch's Windows+CUDA guard in compiled_autograd.h
    which skips the problematic if-constexpr chain that triggers MSVC name lookup bugs.

    Returns dict of environment variable overrides for subprocess.
    """
    if platform.system() != "Windows":
        return {}

    # These flags make MSVC behave like a real C++ compiler:
    # /std:c++17         - Enable C++17
    # /Zc:__cplusplus    - Report correct __cplusplus value (not 199711L)
    # /DUSE_CUDA         - Trigger PyTorch's Windows+CUDA guard to skip buggy templates
    #
    # For nvcc, each -Xcompiler must wrap a SINGLE flag
    cxx_flags = ["/std:c++17", "/Zc:__cplusplus", "/DUSE_CUDA"]
    nvcc_xcompiler = " ".join(f'-Xcompiler "{f}"' for f in cxx_flags)

    env = {
        "CXXFLAGS": os.environ.get("CXXFLAGS", "") + " " + " ".join(cxx_flags),
        "CL": os.environ.get("CL", "") + " " + " ".join(cxx_flags),
        # Pass each flag to nvcc's host compiler individually
        "NVCC_APPEND_FLAGS": os.environ.get("NVCC_APPEND_FLAGS", "") + " " + nvcc_xcompiler,
        # Tell setuptools we're using the SDK
        "DISTUTILS_USE_SDK": "1",
    }

    # Add Windows SDK bin to PATH for rc.exe (resource compiler)
    sdk_bin = find_windows_sdk_bin()
    if sdk_bin:
        current_path = os.environ.get("PATH", "")
        env["PATH"] = f"{sdk_bin};{current_path}"

    return env


def setup_msvc_env():
    """Set MSVC environment in current process and print status."""
    env = get_msvc_build_env()
    if env:
        os.environ.update(env)
        print("  Set CXXFLAGS, CL, NVCC_APPEND_FLAGS for CUTLASS C++17 + conformance fix")


def stage_1_5_patch_headers() -> None:
    """Stage 1.5: Patch PyTorch headers and set up MSVC environment."""
    print("=" * 60)
    print("STAGE 1.5: Patch PyTorch headers for MSVC bugs")
    print("=" * 60)

    include_dir = find_torch_include()
    if include_dir is None:
        print("  WARNING: PyTorch not installed yet, skipping header patches")
        print("  (Run --sync first, then --patch, then --post)")
        return

    print(f"  Found PyTorch include: {include_dir}")

    # Patch std:: ambiguity
    patched = patch_torch_headers(include_dir)
    if patched == 0:
        print("  No patches needed (already patched or headers not present)")

    # Set up MSVC environment
    setup_msvc_env()


def find_uv() -> str:
    """Find uv executable."""
    uv = shutil.which("uv")
    if uv:
        return uv

    # Check common locations
    if platform.system() == "Windows":
        candidates = [
            Path.home() / ".cargo" / "bin" / "uv.exe",
            Path.home() / "AppData" / "Local" / "uv" / "uv.exe",
        ]
    else:
        candidates = [
            Path.home() / ".cargo" / "bin" / "uv",
            Path("/usr/local/bin/uv"),
        ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    print("ERROR: uv not found. Install from https://docs.astral.sh/uv/")
    print("       curl -LsSf https://astral.sh/uv/install.sh | sh")
    sys.exit(1)


def stage_1_sync(uv: str) -> None:
    """Stage 1: uv sync - install main dependencies including CUDA torch."""
    print("=" * 60)
    print("STAGE 1: uv sync (main dependencies)")
    print("=" * 60)
    run([uv, "sync"])


def stage_2_cuda_post(uv: str) -> None:
    """Stage 2: Install CUDA packages that need torch at build time."""
    print("=" * 60)
    print("STAGE 2: Post-sync CUDA package builds")
    print("=" * 60)

    # Get MSVC-specific environment for CUDA builds
    build_env = get_msvc_build_env()
    if build_env:
        print("Using MSVC build environment:")
        for k, v in build_env.items():
            print(f"  {k}={v[:60]}..." if len(v) > 60 else f"  {k}={v}")

    for name, git_url, branch in POST_SYNC_CUDA_PACKAGES:
        if name == "grouped_gemm" and platform.system() == "Windows":
            # Windows needs a patched local build
            print(f"\n--- Installing {name} (Windows patched build) ---")
            local_dir = setup_grouped_gemm_local(uv)
            if local_dir:
                # Clean any previous build
                build_dir = local_dir / "build"
                if build_dir.exists():
                    shutil.rmtree(build_dir)
                # Install from local patched copy (--no-deps since deps already installed by Stage 1)
                run([uv, "pip", "install", str(local_dir), "--no-build-isolation", "--no-deps"], env=build_env)
            else:
                print("  ERROR: Failed to setup local grouped_gemm")
        else:
            print(f"\n--- Installing {name} from {git_url}@{branch} ---")
            pkg_spec = f"git+{git_url}@{branch}"
            run([uv, "pip", "install", pkg_spec, "--no-build-isolation", "--no-deps"], env=build_env)


def verify_install() -> None:
    """Verify critical imports work."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    checks = [
        ("torch", "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"),
        ("grouped_gemm", "import grouped_gemm; print('grouped_gemm OK')"),
    ]

    for name, code in checks:
        try:
            result = run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  [OK] {name}: {result.stdout.strip()}")
            else:
                print(f"  [FAIL] {name}: {result.stderr.strip()}")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Multi-stage uv dependency installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  1.   uv sync         Install main dependencies (torch, etc.)
  1.5  patch headers   Fix PyTorch MSVC bugs (std:: ambiguity, CUTLASS C++17)
  2.   CUDA post-build Build extensions that need torch at compile time

Examples:
  python uv_builder.py              # Full install (all stages)
  python uv_builder.py --sync       # Just stage 1
  python uv_builder.py --patch      # Just stage 1.5 (after sync)
  python uv_builder.py --post       # Just stage 2 (after patch)
"""
    )
    parser.add_argument("--sync", action="store_true", help="Only run stage 1 (uv sync)")
    parser.add_argument("--patch", action="store_true", help="Only run stage 1.5 (patch PyTorch headers)")
    parser.add_argument("--post", action="store_true", help="Only run stage 2 (CUDA post-builds)")
    parser.add_argument("--verify", action="store_true", help="Only verify installation")
    args = parser.parse_args()

    uv = find_uv()
    print(f"Using uv: {uv}")

    # If no specific stage requested, run all
    run_all = not (args.sync or args.patch or args.post or args.verify)

    if args.sync or run_all:
        stage_1_sync(uv)

    if args.patch or run_all:
        stage_1_5_patch_headers()

    if args.post or run_all:
        stage_2_cuda_post(uv)

    if args.verify or run_all:
        verify_install()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
