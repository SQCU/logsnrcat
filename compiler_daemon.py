import zmq
import cloudpickle
import hashlib
import os
import logging
import json
import atexit
import signal
import sys
import multiprocessing
import threading
import time

# Configuration
CACHE_DIR = os.path.expanduser("~/.torch_cas_cache")
INFO_FILE = os.path.join(CACHE_DIR, "daemon_connection.json")
# Leave 1 core for the OS/Proxy, use rest for compilation
WORKER_COUNT = max(1, multiprocessing.cpu_count() - 1)

logging.basicConfig(
    level=logging.INFO, 
    format="[CAS-DAEMON] %(message)s",
    datefmt="%H:%M:%S"
)

def cleanup():
    """Removes the connection file."""
    if os.path.exists(INFO_FILE):
        try:
            os.remove(INFO_FILE)
            logging.info("Connection file cleaned up.")
        except OSError:
            pass

def signal_handler(sig, frame):
    """
    Forceful shutdown.
    We don't ask ZMQ permission to exit; we just leave.
    """
    print(f"\n[CAS-DAEMON] Caught signal {sig}. Exiting immediately.")
    cleanup()
    # os._exit(0) terminates the process without calling cleanup handlers,
    # flushing stdio buffers, etc. It is the only way to guarantee 
    # we don't hang on a zombie thread or locked mutex.
    os._exit(0)

def compile_task_worker(worker_url, ctx):
    """Background worker thread."""
    socket = ctx.socket(zmq.REP)
    # CRITICAL: LINGER=0 ensures we don't hang waiting to flush on close
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(worker_url)

    while True:
        try:
            message = socket.recv()
        except zmq.ContextTerminated:
            return
        except zmq.ZMQError:
            return

        try:
            fn, args, kwargs = cloudpickle.loads(message)
            task_hash = hashlib.sha256(message).hexdigest()
            
            result = fn(*args, **kwargs)
            response = {"status": "OK", "result": result, "hash": task_hash}
        except Exception as e:
            logging.error(f"Compilation Error: {e}")
            response = {"status": "ERROR", "error": str(e)}
        
        try:
            socket.send(cloudpickle.dumps(response))
        except (zmq.ContextTerminated, zmq.ZMQError):
            return

def run_proxy(frontend, backend):
    """Runs the proxy in a thread."""
    try:
        zmq.proxy(frontend, backend)
    except zmq.ContextTerminated:
        pass
    except Exception as e:
        print(f"Proxy Error: {e}")

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Register Signal Handlers
    # We don't need atexit because signal_handler calls cleanup() directly
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    context = zmq.Context()

    # 1. Frontend (TCP)
    frontend = context.socket(zmq.ROUTER)
    frontend.setsockopt(zmq.LINGER, 0) # <--- IMPORTANT
    try:
        port = frontend.bind_to_random_port("tcp://127.0.0.1")
    except zmq.ZMQError as e:
        logging.critical(f"Bind failed: {e}")
        return

    # 2. Backend (In-Memory)
    backend = context.socket(zmq.DEALER)
    backend.setsockopt(zmq.LINGER, 0) # <--- IMPORTANT
    backend.bind("inproc://workers")

    address = f"tcp://127.0.0.1:{port}"
    logging.info(f"Listening on: {address}")
    logging.info(f"Workers: {WORKER_COUNT}")

    # 3. Start Workers
    for i in range(WORKER_COUNT):
        t = threading.Thread(target=compile_task_worker, args=("inproc://workers", context), name=f"Worker-{i}")
        t.daemon = True
        t.start()

    # 4. Write Discovery File
    info = {"address": address, "pid": os.getpid()}
    with open(INFO_FILE, "w") as f:
        json.dump(info, f)

    # 5. Start Proxy
    proxy_thread = threading.Thread(target=run_proxy, args=(frontend, backend), name="ProxyThread")
    proxy_thread.daemon = True
    proxy_thread.start()

    logging.info("Ready. Press Ctrl+C to stop.")
    
    # 6. Main Loop
    # We just park the main thread here. 
    # If Ctrl+C is pressed, signal_handler fires on this thread.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()