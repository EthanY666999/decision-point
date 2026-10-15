"""Sample whole-device memory (including vLLM), rather than actor-only allocator."""
import subprocess
import threading

class GPUPeak:
    def __init__(self):
        self.peak_mib=0
        self.error=None
        self.event=threading.Event()
        self.thread=threading.Thread(target=self.loop,daemon=True)
    def loop(self):
        while not self.event.is_set():
            try:
                lines=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=3).splitlines()
                if len(lines)!=1:raise ValueError('Expected exactly one GPU')
                self.peak_mib=max(self.peak_mib,int(lines[0]))
            except Exception as exc:self.error=repr(exc)
            self.event.wait(.5)
    def start(self):self.thread.start()
    def stop(self):self.event.set();self.thread.join(timeout=4)
