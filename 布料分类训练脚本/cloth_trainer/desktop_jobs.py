"""Nonblocking cancellable desktop jobs, with latest-request-wins delivery."""
from pathlib import Path
import tempfile
import json
import sys
import time
import threading
from PyQt5 import QtCore as C


class WorkThread(C.QThread):
    result=C.pyqtSignal(object)
    failed=C.pyqtSignal(str)
    def __init__(self,fn,parent=None):super().__init__(parent);self.fn=fn;self.cancel=threading.Event()
    def run(self):
        try:
            value=self.fn(self.cancel.is_set)
            if not self.cancel.is_set():self.result.emit(value)
        except Exception as e:
            if not self.cancel.is_set():self.failed.emit(str(e))


class DesktopJobs(C.QObject):
    changed=C.pyqtSignal(str)
    failed=C.pyqtSignal(str)
    def __init__(self,parent=None):
        super().__init__(parent);self.active={};self.retired=[];self.serial=0;self.last_stage='';self.timer=C.QTimer(self);self.timer.setInterval(500);self.timer.timeout.connect(self._tick);self.timer.start()
    def running(self):return bool(self.active or self.retired)
    def _tick(self):
        if self.active:
            job=next(reversed(self.active.values()));self.last_stage=f"正在{job['stage']} · 已等待{time.monotonic()-job['start']:.1f}秒（可取消）";self.changed.emit(self.last_stage)
    def _begin(self,key,stage):
        self.cancel(key);self.serial+=1;job=dict(id=self.serial,stage=stage,start=time.monotonic());self.active[key]=job;self._tick();return job
    def submit(self,key,kind,payload,callback,stage='读取实验记录'):
        job=self._begin(key,stage);temp=tempfile.TemporaryDirectory(prefix='cloth-desktop-');root=Path(temp.name)
        request=root/'request.json';destination=root/'result.json';request.write_text(json.dumps(dict(kind=kind,payload=payload)),encoding='utf-8')
        process=C.QProcess(self);job.update(worker=process,temp=temp)
        def done(*args):
            if job.get('completed'):return
            job['completed']=True
            current=self.active.get(key) is job
            if current:
                self.active.pop(key,None)
                try:
                    result=json.loads(destination.read_text(encoding='utf-8'))
                    if not result['ok']:raise ValueError(result['error'])
                    callback(result['value']);self.last_stage=stage+'完成';self.changed.emit(self.last_stage)
                except Exception as e:self.failed.emit(stage+'失败：'+str(e))
            if job in self.retired:self.retired.remove(job)
            temp.cleanup();process.deleteLater()
        process.finished.connect(done)
        process.errorOccurred.connect(lambda error:done() if error==C.QProcess.FailedToStart else None)
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]));process.setProgram(sys.executable);process.setArguments(['-m','cloth_trainer.desktop_worker',str(request),str(destination)]);process.start()
    def submit_thread(self,key,fn,callback,stage):
        job=self._begin(key,stage);worker=WorkThread(fn,self);job['worker']=worker
        def deliver(value):
            if self.active.get(key) is not job:return
            self.active.pop(key,None);self.retired.append(job);self.last_stage=stage+'完成';self.changed.emit(self.last_stage)
            try:callback(value)
            except Exception as e:self.failed.emit(stage+'失败：'+str(e))
        worker.result.connect(deliver)
        worker.failed.connect(lambda message:self.failed.emit(stage+'失败：'+message) if self.active.get(key) is job else None)
        def done():
            if self.active.get(key) is job:self.active.pop(key,None);self.last_stage=stage+'完成';self.changed.emit(self.last_stage)
            if job in self.retired:self.retired.remove(job)
            worker.deleteLater()
        worker.finished.connect(done);worker.start()
    def cancel(self,key):
        job=self.active.pop(key,None)
        if not job:return
        self.retired.append(job);worker=job.get('worker')
        if isinstance(worker,WorkThread):worker.cancel.set()
        elif worker is not None:
            worker.kill()  # Only this owned parser subprocess; never acquisition/training.
        self.last_stage='正在取消'+job['stage'];self.changed.emit(self.last_stage)
    def cancel_all(self):
        for key in list(self.active):self.cancel(key)
