import torch
import zmq
import cloudpickle
import json
import os
import time
import sys
import atexit
import signal
from concurrent.futures import ThreadPoolExecutor

CACHE_DIR = os.path.expanduser("~/.torch_cas_cache")
INFO_FILE = os.path.join(CACHE_DIR, "daemon_connection.json")

class ZMQCompilerClient:
    def __init__(self):
        self.context = zmq.Context()
        self.address = self._discover_daemon()
        
        import multiprocessing
        self.executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
        
        # Register cleanup
        atexit.register(self.shutdown)
        signal.signal(signal.SIGTERM, self._sig_handler)
        
        print(f"[CAS-Client] Connected to compiler daemon at {self.address}")

    def _discover_daemon(self, retries=3):
        for i in range(retries):
            if os.path.exists(INFO_FILE):
                try:
                    with open(INFO_FILE, "r") as f:
                        data = json.load(f)
                        return data["address"]
                except (json.JSONDecodeError, OSError):
                    pass 
            time.sleep(0.5)
        raise RuntimeError(f"Daemon not found. Run 'python compiler_daemon.py'")

    def _sig_handler(self, sig, frame):
        self.shutdown()
        sys.exit(0)

    def shutdown(self):
        """Cancels pending jobs so we exit instantly."""
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.context.term()
        except Exception:
            pass

    def submit(self, task_fn, *args, **kwargs):
        payload = cloudpickle.dumps((task_fn, args, kwargs))
        return self.executor.submit(self._network_request, payload)

    def _network_request(self, payload):
        try:
            socket = self.context.socket(zmq.REQ)
            socket.connect(self.address)
            socket.setsockopt(zmq.RCVTIMEO, 120000) 
            socket.setsockopt(zmq.LINGER, 0)
            
            socket.send(payload)
            # Daemon sends raw bytes via cloudpickle
            resp_bytes = socket.recv()
            response = cloudpickle.loads(resp_bytes)
            
            socket.close()
            
            if response['status'] == 'OK':
                return response['result']
            else:
                raise RuntimeError(f"Remote Error: {response['error']}")
        except zmq.ZMQError:
            # Context dead
            return None

# --- Mock Object ---
class MockProcessPool:
    def __init__(self, client):
        self.client = client
    def __call__(self, *args, **kwargs):
        return self.client
    def cache_info(self):
        return type('Info', (), {'currsize': 0})()
    def cache_clear(self):
        pass

def install_cas_client():
    print("[CAS-Client] Hijacking Torch Inductor Compilation Backend...")
    try:
        client = ZMQCompilerClient()
    except Exception as e:
        print(f"[CAS-Client] Failed to init: {e}")
        return

    from torch._inductor import async_compile
    
    # Patches
    def zmq_submit(self, task_fn, *args, **kwargs):
        return client.submit(task_fn, *args, **kwargs)
        
    def zmq_wait(self, scope):
        updates = {}
        for k, v in scope.items():
            if hasattr(v, 'result'):
                try:
                    res = v.result()
                    if res is not None: updates[k] = res
                except Exception:
                    pass
            elif isinstance(v, (list, tuple)):
                updates[k] = [x.result() if hasattr(x, 'result') else x for x in v]
        scope.update(updates)
    
    def zmq_wakeup(cls): pass 

    # Apply
    async_compile.AsyncCompile.submit = zmq_submit
    async_compile.AsyncCompile.wait = zmq_wait
    async_compile.AsyncCompile.use_process_pool = classmethod(lambda cls: True)
    async_compile.AsyncCompile.process_pool = MockProcessPool(client)
    async_compile.AsyncCompile.wakeup = classmethod(zmq_wakeup)
    
    import multiprocessing
    import torch._inductor.config
    torch._inductor.config.compile_threads = multiprocessing.cpu_count()
    
    print("[CAS-Client] Success. Inductor is now networked.")