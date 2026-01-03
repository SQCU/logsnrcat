#!/usr/bin/env python3
"""
SNAKE SLAYER
The mother of all pkills. No python survives.
Zero dependencies. Pure stdlib carnage.
"""
import os
import sys
import signal
import subprocess

def get_python_pids_unix():
    """Find all python PIDs on Unix-like systems."""
    pids = []
    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True
        )
        for line in result.stdout.split('\n'):
            lower = line.lower()
            if 'python' in lower:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pids.append(int(parts[1]))
                    except ValueError:
                        pass
    except Exception:
        pass
    return pids

def get_python_pids_windows():
    """Find all python PIDs on Windows."""
    pids = []
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python*", "/FO", "CSV"],
            capture_output=True, text=True, shell=True
        )
        for line in result.stdout.split('\n')[1:]:  # Skip header
            if 'python' in line.lower():
                parts = line.replace('"', '').split(',')
                if len(parts) >= 2:
                    try:
                        pids.append(int(parts[1]))
                    except ValueError:
                        pass
    except Exception:
        pass
    return pids

def kill_pid_unix(pid):
    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except (ProcessLookupError, PermissionError):
        return False

def kill_pid_windows(pid):
    try:
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)],
            capture_output=True, shell=True
        )
        return True
    except Exception:
        return False

def slay():
    my_pid = os.getpid()
    is_windows = sys.platform == 'win32'

    # Find the snakes
    if is_windows:
        pids = get_python_pids_windows()
        kill_fn = kill_pid_windows
    else:
        pids = get_python_pids_unix()
        kill_fn = kill_pid_unix

    # Exclude ourselves
    targets = [p for p in pids if p != my_pid]

    # Execute
    dispatched = 0
    denied = 0
    for pid in targets:
        if kill_fn(pid):
            dispatched += 1
        else:
            denied += 1

    # The reckoning
    print()
    print("=" * 50)
    print("         SNAKE SLAYER - MISSION COMPLETE")
    print("=" * 50)
    print(f"  Snakes dispatched: {dispatched}")
    if denied:
        print(f"  Survivors (access denied): {denied}")
    print("=" * 50)
    print()

if __name__ == "__main__":
    slay()
