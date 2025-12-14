import zmq
import cloudpickle
import hashlib
import os
import logging
import json
import signal
import sys
import multiprocessing
import time

# Configuration
CACHE_DIR = os.path.expanduser("~/.torch_cas_cache")
INFO_FILE = os.path.join(CACHE_DIR, "daemon_connection.json")
# Use all cores. Since these are processes, they truly run in parallel.
WORKER_COUNT = max(1, int((multiprocessing.cpu_count()/2) -1))  #confusion between thread and core in apis

logging.basicConfig(
    level=logging.INFO, 
    format="[CAS-DAEMON] %(message)s",
    datefmt="%H:%M:%S"
)

def cleanup():
    if os.path.exists(INFO_FILE):
        try:
            os.remove(INFO_FILE)
            logging.info("Connection file cleaned up.")
        except OSError:
            pass

def signal_handler(sig, frame):
    print(f"\n[CAS-DAEMON] Caught signal {sig}. Exiting.")
    cleanup()
    # Kill the entire process tree (Daemon + Workers)
    # On Windows, os._exit catches the children usually if they are daemons.
    os._exit(0)

def compile_task_worker(backend_port):
    """
    Process-based Worker.
    Has its own GIL, its own memory space, and its own ZMQ Context.
    """
    # CRITICAL: Must create a new context inside the process
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    
    # Connect to the Backend TCP port
    socket.connect(f"tcp://127.0.0.1:{backend_port}")

    while True:
        try:
            message = socket.recv()
        except zmq.ContextTerminated:
            return
        except Exception:
            return

        try:
            fn, args, kwargs = cloudpickle.loads(message)
            task_hash = hashlib.sha256(message).hexdigest()
            
            # Pure Parallel Execution (No GIL Interference)
            result = fn(*args, **kwargs)
            
            response = {"status": "OK", "result": result, "hash": task_hash}
        except Exception as e:
            logging.error(f"Worker Error: {e}")
            response = {"status": "ERROR", "error": str(e)}
        
        try:
            socket.send(cloudpickle.dumps(response))
        except Exception:
            return

def run_proxy(frontend_port, backend_port):
    """
    The Traffic Cop.
    Binds to both ports and shuffles packets.
    """
    ctx = zmq.Context()
    
    # Frontend: Clients connect here
    frontend = ctx.socket(zmq.ROUTER)
    frontend.setsockopt(zmq.LINGER, 0)
    frontend.bind(f"tcp://127.0.0.1:{frontend_port}")
    
    # Backend: Workers connect here
    backend = ctx.socket(zmq.DEALER)
    backend.setsockopt(zmq.LINGER, 0)
    backend.bind(f"tcp://127.0.0.1:{backend_port}")
    
    try:
        zmq.proxy(frontend, backend)
    except Exception as e:
        print(f"Proxy Error: {e}")

def main():
    # Windows Multiprocessing Boilerplate
    multiprocessing.freeze_support()
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 1. Reserve Ports (Using temporary sockets to find free ports)
    # We need two ports now: one for clients, one for workers.
    temp_ctx = zmq.Context()
    
    s1 = temp_ctx.socket(zmq.REP)
    frontend_port = s1.bind_to_random_port("tcp://127.0.0.1")
    s1.close()
    
    s2 = temp_ctx.socket(zmq.REP)
    backend_port = s2.bind_to_random_port("tcp://127.0.0.1")
    s2.close()
    
    temp_ctx.term()

    address = f"tcp://127.0.0.1:{frontend_port}"
    logging.info(f"Client Address: {address}")
    logging.info(f"Worker Backend: tcp://127.0.0.1:{backend_port}")
    logging.info(f"Spawning {WORKER_COUNT} worker processes...")

    # 2. Start Proxy Process
    # We run the proxy in a separate process to decouple it completely from the OS shell
    proxy_proc = multiprocessing.Process(
        target=run_proxy, 
        args=(frontend_port, backend_port),
        name="Proxy"
    )
    proxy_proc.daemon = True
    proxy_proc.start()

    # 3. Start Worker Processes
    workers = []
    for i in range(WORKER_COUNT):
        p = multiprocessing.Process(
            target=compile_task_worker, 
            args=(backend_port,), 
            name=f"Worker-{i}"
        )
        p.daemon = True
        p.start()
        workers.append(p)

    # 4. Write Discovery File
    info = {"address": address, "pid": os.getpid()}
    with open(INFO_FILE, "w") as f:
        json.dump(info, f)

    logging.info("Ready. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
            # Optional: Check if proxy is still alive
            if not proxy_proc.is_alive():
                logging.error("Proxy died unexpectedly.")
                break
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()