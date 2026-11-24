"""One-session desktop resume of the user's existing experiment."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from train_cloth import configure_qt_platform
configure_qt_platform()
from PyQt5 import QtWidgets as W,QtCore as C
from cloth_trainer.ui import TrainerWindow
W.QApplication.setAttribute(C.Qt.AA_EnableHighDpiScaling,True)
app=W.QApplication([]);app.setStyle('Fusion')
window=TrainerWindow(ROOT/'outputs/models')
# Only the training worker holds a sleep inhibitor. Blanking the display is allowed.
start_worker=window.process.start

def protected_start(*args):
    if window.worker_kind in ('train','final'):
        program=window.process.program();arguments=window.process.arguments()
        window.process.setProgram('/usr/bin/systemd-inhibit')
        window.process.setArguments(['--what=sleep','--mode=block','--who=布料分类训练','--why=GPU训练期间防止休眠破坏CUDA上下文',program,*arguments])
    start_worker(*args)

window.process.start=protected_start
window.show();window.show_results(ROOT/'outputs/models/CLOTH4_20261117_170039_86f788')
C.QTimer.singleShot(500,window.resume_selected)
raise SystemExit(app.exec_())
