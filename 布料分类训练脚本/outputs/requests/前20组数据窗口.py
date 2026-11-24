"""Import only the user's selected first20 CSV/class; never auto-start training."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from train_cloth import configure_qt_platform
configure_qt_platform()
from PyQt5 import QtWidgets as W,QtCore as C
from cloth_trainer.ui import TrainerWindow
from cloth_trainer.storage import write_json
W.QApplication.setAttribute(C.Qt.AA_EnableHighDpiScaling,True)
app=W.QApplication([]);app.setStyle('Fusion');app.setApplicationName('布料分类训练')
window=TrainerWindow(ROOT/'outputs/models');window.setWindowTitle('布料分类训练 · 各布料前20组');window.show()
start_worker=window.process.start

def protected_start(*args):
    if window.worker_kind in ('train','final'):
        program=window.process.program();arguments=window.process.arguments();window.process.setProgram('/usr/bin/systemd-inhibit')
        window.process.setArguments(['--what=sleep','--mode=block','--who=布料分类训练','--why=GPU训练期间防止休眠破坏CUDA上下文',program,*arguments])
    start_worker(*args)

window.process.start=protected_start
paths=json.loads((ROOT/'outputs/requests/各布料前20组路径.json').read_text())

def imported():
    if window.import_jobs:C.QTimer.singleShot(500,imported);return
    from collections import Counter
    report=dict(requested=paths,table_rows=window.data_table.rowCount(),record_count=len(window.records),errors=window.errors,
        labels=dict(Counter(r.label for r in window.records)),eligible=len(window.selected_records()),busy=window.busy(),status=window.status.text(),
        records=[dict(path=r.path,label=r.label,quality=r.quality,eligible=window.record_eligible(r)) for r in window.records])
    write_json(ROOT/'outputs/verification/前20组窗口导入结果.json',report)
    print(json.dumps({k:report[k] for k in ('table_rows','record_count','labels','eligible','busy','status')},ensure_ascii=False),flush=True)

def begin():
    window.import_paths(paths);C.QTimer.singleShot(500,imported)
C.QTimer.singleShot(300,begin)
raise SystemExit(app.exec_())
