# build_pytorch3d.py
# python build_pytorch3d.py
import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path

# --- Configuration ---
CUB_VERSION = "2.1.0"
CUB_URL = f"https://github.com/NVIDIA/cub/archive/refs/tags/{CUB_VERSION}.zip"
PYTORCH3D_REPO = "git+https://github.com/facebookresearch/pytorch3d.git"

def download_and_extract_cub(target_dir):
    """Downloads NVIDIA CUB headers required for PyTorch3D."""
    cub_dir = target_dir / f"cub-{CUB_VERSION}"
    if cub_dir.exists():
        print(f"✅ CUB found at: {cub_dir}")
        return cub_dir

    print(f"⬇️  Downloading NVIDIA CUB {CUB_VERSION}...")
    zip_path = target_dir / "cub.zip"
    try:
        urllib.request.urlretrieve(CUB_URL, zip_path)
        print("📦 Extracting CUB...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
    finally:
        if zip_path.exists():
            os.remove(zip_path)
            
    print(f"✅ CUB installed.")
    return cub_dir

def enable_long_paths_check():
    """Checks the Windows Registry for LongPathsEnabled."""
    if os.name != 'nt': return
    
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem")
        value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
        if value != 1:
            print("\n⚠️  WARNING: 'LongPathsEnabled' is not set to 1 in the Registry.")
            print("    If the build fails with 'filename too long', run PowerShell as Admin:")
            print("    Set-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem' -Name 'LongPathsEnabled' -Value 1")
            print("    (Then reboot).\n")
    except Exception:
        pass

def main():
    # 1. Setup Build Directory (Short path to avoid Ninja limit)
    # Using a short path relative to drive root is safest, but local is ok if project path isn't deep.
    # We will force UV to use a short temp dir.
    build_root = Path("build_temp").resolve()
    build_root.mkdir(exist_ok=True)
    
    # 2. Get CUB
    cub_home = download_and_extract_cub(build_root)
    
    # 3. Prepare Environment
    env = os.environ.copy()
    env["CUB_HOME"] = str(cub_home)
    env["DISTUTILS_USE_SDK"] = "1"
    env["FORCE_CUDA"] = "1"
    
    # NVCC flags for Windows (sometimes needed to fix generic lambda issues in MSVC)
    env["NVCC_FLAGS"] = "-allow-unsupported-compiler"

    # 4. Handle the Long Filename Issue via Cache Redirection
    # Ninja crashes if paths > 260 chars. UV's default cache is deep in AppData.
    # We force UV to cache in a shallow directory.
    uv_cache = Path("C:/t/uv") if os.name == 'nt' else build_root / "uv_cache"
    uv_cache.mkdir(parents=True, exist_ok=True)
    env["UV_CACHE_DIR"] = str(uv_cache)
    
    # Also set TMP/TEMP for standard pip builds invoked by uv
    env["TMP"] = str(uv_cache)
    env["TEMP"] = str(uv_cache)

    print(f"🔧 Build Env Configured:")
    print(f"   CUB_HOME: {env['CUB_HOME']}")
    print(f"   UV_CACHE: {env['UV_CACHE_DIR']}")
    
    enable_long_paths_check()

    # 5. The Build Command
    # We use --no-build-isolation to see the environment variables we just set
    cmd = [
        "uv", "pip", "install", 
        PYTORCH3D_REPO, 
        "--no-build-isolation",
        "-v" # Verbose to see compilation errors
    ]
    
    print(f"🚀 Launching Build: {' '.join(cmd)}")
    print("   This will take a while (compiling C++/CUDA)...")
    
    try:
        subprocess.check_call(cmd, env=env)
        print("\n✨ PyTorch3D installed successfully!")
        
        # Cleanup
        shutil.rmtree(build_root, ignore_errors=True)
        if os.name == 'nt' and os.path.exists("C:/t/uv"):
            print("   (You can manually delete C:/t/uv to reclaim space)")
            
    except subprocess.CalledProcessError:
        print("\n❌ Build Failed.")
        print("   If the error is still 'filename too long', you MUST enable LongPaths in the registry.")

if __name__ == "__main__":
    main()