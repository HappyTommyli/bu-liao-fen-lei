"""Isolate model loading, dataset validation and inference from the GUI/GIL."""
from dataclasses import asdict
from pathlib import Path
import json
import sys
import tempfile
import threading
from PyQt5 import QtCore as C


class OnlineProcess(C.QObject):
    observed=C.pyqtSignal(dict);record=C.pyqtSignal(dict);loaded=C.pyqtSignal(dict);failed=C.pyqtSignal(str);finished=C.pyqtSignal()
    def __init__(self,model_path,paths,config,source='independent',live=False,speed=0.,parent=None,policy=None,heldout=False,include_return=None,live_truth=''):
        super().__init__(parent);self.paused=threading.Event();self.temp=tempfile.TemporaryDirectory(prefix='cloth-online-');self.root=Path(self.temp.name);self.buffer='';self.stopped=False;self.done=False
        request=dict(model_path=str(model_path),paths=list(paths),config=asdict(config),source=source,live=live,speed=speed,policy=policy,heldout=heldout,include_return=include_return,live_truth=live_truth)
        (self.root/'request.json').write_text(json.dumps(request,ensure_ascii=False),encoding='utf-8')
        self.process=C.QProcess(self);self.process.readyReadStandardOutput.connect(self._read);self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
    def start(self):
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]));self.process.start(sys.executable,['-m','cloth_trainer.online_process_worker',str(self.root)])
    def isRunning(self):return self.process.state()!=C.QProcess.NotRunning
    def _read(self):
        self.buffer+=bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace')
        lines=self.buffer.split('\n');self.buffer=lines.pop()
        for line in lines:
            try:
                item=json.loads(line);kind=item['kind'];getattr(self,kind).emit(item['value'])
            except (ValueError,KeyError,AttributeError):continue
    def _error(self,error):
        if not self.stopped:self.failed.emit('无法启动在线识别后台进程')
        if error==C.QProcess.FailedToStart:self._finished()
    def _finished(self,*args):
        if self.done:return
        self.done=True;self._read()
        error=bytes(self.process.readAllStandardError()).decode('utf-8',errors='replace')
        if self.process.exitCode()!=0 and not self.stopped:self.failed.emit('后台识别异常退出：'+error[-500:])
        self.finished.emit();self.temp.cleanup()
    def stop(self):
        self.stopped=True
        if self.isRunning():
            (self.root/'cancel').touch();C.QTimer.singleShot(8000,lambda:self.process.kill() if self.isRunning() else None)
    def set_paused(self,value):
        path=self.root/'pause';self.paused.set() if value else self.paused.clear()
        if value:path.touch()
        else:path.unlink(missing_ok=True)
    def wait(self,msecs=30000):return self.process.waitForFinished(msecs)
