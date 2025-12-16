# make_proxies
# python tools/make_proxies.py C:/dox/recordings/rl_capture/videos C:/dox/recordings/rl_capture/proxies --res 320
# python -m src.make_proxies C:/dox/recordings/rl_capture/capture_run_1760343426/videos C:/dox/recordings/rl_capture/capture_run_1760343426/videos/proxies --res 480
import argparse
import subprocess
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 'ffmpeg' not found in PATH. Please install FFmpeg.")
        sys.exit(1)

def process_video(args):
    src, dst, max_res, codec = args
    
    if dst.exists():
        return None
        
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Common flags
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-i', str(src),
        '-vf', f"scale='min({max_res},iw)':-2", 
        '-g', '15',
        '-keyint_min', '15',
        '-sc_threshold', '0',
        '-pix_fmt', 'yuv420p',
        '-an',
    ]

    # Codec-specific flags
    if codec == 'h264_nvenc':
        # Hardware Encoding (Fast, requires NVIDIA GPU)
        cmd.extend([
            '-c:v', 'h264_nvenc',
            '-preset', 'p1',      # Fastest preset
            '-rc', 'constqp',     # Constant Quality
            '-qp', '26'           # Quality (lower is better, 20-28 is reasonable for proxies)
        ])
    else:
        # Software Encoding (Standard)
        cmd.extend([
            '-c:v', 'libx264',
            '-crf', '23',         # Visual quality
            '-preset', 'veryfast'
        ])

    cmd.append(str(dst))
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        return f"Error processing {src.name}: {e.stderr.decode('utf-8', errors='ignore').strip()}"
    return None

def main():
    parser = argparse.ArgumentParser(description="Create high-throughput proxy videos")
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("dest_dir", type=Path)
    parser.add_argument("--res", type=int, default=320)
    parser.add_argument("--codec", type=str, default="h264_nvenc", choices=["libx264", "h264_nvenc"])
    parser.add_argument("--workers", type=int, default=4, help="Parallel ffmpeg processes (Keep <= 4 for NVENC)")
    args = parser.parse_args()
    
    check_ffmpeg()
    
    if not args.source_dir.exists():
        sys.exit(f"Source not found: {args.source_dir}")

    files = [p for p in args.source_dir.rglob("*") if p.suffix.lower() in {'.mp4', '.mov', '.mkv', '.avi', '.webm'}]
    if not files:
        sys.exit("No video files found.")

    print(f"Generating {args.res}px proxies using {args.codec} with {args.workers} workers...")
    
    tasks = []
    for f in files:
        rel_path = f.relative_to(args.source_dir)
        dest_path = args.dest_dir / rel_path.with_suffix('.mp4')
        tasks.append((f, dest_path, args.res, args.codec))
        
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = [exe.submit(process_video, t) for t in tasks]
        
        errors = []
        for f in tqdm(as_completed(futures), total=len(files), desc="Transcoding"):
            result = f.result()
            if result: errors.append(result)
                
    if errors:
        print(f"\n{len(errors)} Errors (First 10):")
        for e in errors[:10]: print(f" - {e}")

if __name__ == "__main__":
    main()