"""Offscreen layout audit only; synthetic sources, no real training requests."""
import json,os,subprocess,sys
from pathlib import Path
script=r'''
import sys,tempfile,json
from pathlib import Path
from PyQt5 import QtWidgets as W,QtCore as C
sys.path.insert(0,str(Path.cwd()/'tests'))
from test_data import fixture_csv
from cloth_trainer.data import load_record
from cloth_trainer.tcn_ui import TCNTrainerWindow
W.QApplication.setAttribute(C.Qt.AA_EnableHighDpiScaling,True)
app=W.QApplication([]);app.setStyle('Fusion');out=Path('outputs/verification/tcn-workbench')
report=[];scale=sys.argv[1]
with tempfile.TemporaryDirectory() as scratch:
 root=Path(scratch);w=TCNTrainerWindow(root/'models')
 records=[load_record(fixture_csv(root/'raw',c,i,speed=100),signals=('by',)) for c in ('布A','布B') for i in range(6)]
 w.add_records(records,[]);w.task_name.setText('TCN参数与布局预览（合成数据）');w.show()
 for size in [(1024,640),(1280,720),(1366,768),(1920,1080)]:
  w.resize(*size);w.navigation.setCurrentRow(1);app.processEvents()
  bad=[]
  for text in w.pages.currentWidget().findChildren(W.QLabel):
   if text.isVisible() and text.text():
    required=text.heightForWidth(text.width()) if text.hasHeightForWidth() else text.sizeHint().height()
    if required>text.height()+2:bad.append(dict(text=text.text(),height=text.height(),required=required))
  report.append(dict(scale=scale,size=size,clipped_labels=bad,parameter_viewport=[w.parameter_scroll.viewport().width(),w.parameter_scroll.viewport().height()],start_enabled=w.start_button.isEnabled()))
  assert not bad,report[-1]
  assert w.start_button.isEnabled()
  w.grab().save(str(out/f'training-{size[0]}-scale{scale}.png'))
 w.resize(1366,768);w.editor.fields['tcn_mode'].setCurrentIndex(1);w.monitor_tabs.setCurrentIndex(1);app.processEvents()
 w.grab().save(str(out/f'four-way-scale{scale}.png'))
 w.navigation.setCurrentRow(0);app.processEvents();w.grab().save(str(out/f'data-scale{scale}.png'))
 w.close();app.processEvents()
print(json.dumps(report,ensure_ascii=False))
'''
report=[]
for scale in ['1','1.25','1.5']:
 env=dict(os.environ,QT_QPA_PLATFORM='offscreen',QT_SCALE_FACTOR=scale)
 process=subprocess.run([sys.executable,'-c',script,scale],env=env,capture_output=True,text=True)
 if process.returncode:
  print(process.stdout);print(process.stderr[-6000:]);raise SystemExit(process.returncode)
 report.extend(json.loads(process.stdout.strip()))
Path('outputs/verification/tcn-workbench/layout-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(f'{len(report)} size/scale combinations passed; no squeezed labels; synthetic input can start training.')
