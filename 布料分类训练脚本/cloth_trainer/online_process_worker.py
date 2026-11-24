"""Child process entry; data-only IPC, never serial or motor operations."""
from pathlib import Path
import json
import sys
import time


class FileEvent:
    def __init__(self,path):self.path=path
    def is_set(self):return self.path.exists()
    def wait(self,seconds):
        until=time.monotonic()+seconds
        while not self.is_set() and time.monotonic()<until:time.sleep(min(.03,max(0,until-time.monotonic())))
        return self.is_set()


def main():
    root=Path(sys.argv[1]);request=json.loads((root/'request.json').read_text(encoding='utf-8'))
    import torch
    torch.set_num_threads(2)
    from .online import OnlineConfig
    from .online_ui import OnlineWorker
    request['config']=OnlineConfig(**request['config']);worker=OnlineWorker(**request);worker.cancel=FileEvent(root/'cancel');worker.paused=FileEvent(root/'pause')
    def send(kind,value):print(json.dumps(dict(kind=kind,value=value),ensure_ascii=False,allow_nan=False),flush=True)
    for kind in ('observed','record','loaded','failed'):getattr(worker,kind).connect(lambda value,k=kind:send(k,value))
    worker.run()


if __name__=='__main__':main()
