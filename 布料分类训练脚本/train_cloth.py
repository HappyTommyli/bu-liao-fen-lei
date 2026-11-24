#!/usr/bin/env python3
"""Standalone CERA-Net desktop, headless training, and process worker entry."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent

def startup_project(explicit=None,allow_default=True,root=ROOT):
    if explicit is not None:return Path(explicit)
    current=Path(root)/'outputs/当前训练项目.json'
    return current if allow_default and current.is_file() else None

def startup_sources(project=None,directory=None,*,screenshot=False,root=ROOT):
    if project is not None:return Path(project),None
    if directory is not None:return None,Path(directory)
    if screenshot:return None,None
    return None,Path(root).parent/'xz-control'/'data'/'布料采集数据新'

def configure_qt_platform(environ=None,*,screenshot=False):
    """Choose the live desktop backend before QApplication loads QPA plugins.

    Qt5 wheels can default to xcb on GNOME Wayland, even when Wayland works
    and the optional X11 compatibility libraries are absent.
    """
    env=os.environ if environ is None else environ
    if env.get('QT_QPA_PLATFORM'):
        return
    if screenshot:
        env['QT_QPA_PLATFORM']='offscreen'
        return
    if not sys.platform.startswith('linux'):
        return
    display=env.get('WAYLAND_DISPLAY')
    if not display and env.get('XDG_SESSION_TYPE','').lower()=='wayland':
        display='wayland-0'
    if not display:
        return
    address=Path(display)
    if not address.is_absolute():
        runtime=env.get('XDG_RUNTIME_DIR')
        if not runtime:
            return
        address=Path(runtime)/address
    if address.is_socket():
        env.setdefault('WAYLAND_DISPLAY',display)
        env['QT_QPA_PLATFORM']='wayland'

def main(argv=None):
    parser=argparse.ArgumentParser(description='TCN / |dBy| 布料分类训练工作台')
    parser.add_argument('--store',type=Path,default=ROOT/'outputs/models')
    parser.add_argument('--project',type=Path,help='启动后打开指定项目，替代默认目录自动导入')
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--kind',choices=['train','predict','test','final'],default='train',help=argparse.SUPPRESS)
    parser.add_argument('--import-directory',type=Path,help='启动后导入CSV目录')
    parser.add_argument('--start-training',action='store_true',help='导入检查完成后开始第一阶段训练（需指定导入目录）')
    parser.add_argument('--exclude-class',action='append',default=[],help='此次训练明确不勾选的类别，可重复使用')
    parser.add_argument('--experiment-name',default='',help='此次实验名称')
    parser.add_argument('--capture-after-start',type=Path,help='开始训练并出现首轮结果后保存真实窗口截图，不关闭应用')
    parser.add_argument('--screenshot',type=Path,help='离屏保存窗口截图后退出')
    parser.add_argument('--page',type=int,default=0,choices=range(5),help='截图页面0到4')
    sub=parser.add_subparsers(dest='command')
    check=sub.add_parser('check',help='检查CSV，不训练');check.add_argument('paths',nargs='+')
    train=sub.add_parser('train',help='按JSON任务训练');train.add_argument('request',type=Path)
    predict=sub.add_parser('predict',help='按JSON任务识别');predict.add_argument('request',type=Path)
    test=sub.add_parser('test',help='对已冻结候选运行留出测试');test.add_argument('request',type=Path)
    final=sub.add_parser('final',help='按选定开发集配置训练最终模型');final.add_argument('request',type=Path)
    sub.add_parser('diagnose-gpu',help='显示真实GPU与PyTorch环境')
    args=parser.parse_args(argv)
    if args.project and args.import_directory:parser.error('--project与--import-directory不能同时指定')
    if args.command=='diagnose-gpu':
        from cloth_trainer.devices import diagnostics
        print(json.dumps(diagnostics(),ensure_ascii=False,indent=2));return 0
    if args.start_training and (not args.import_directory or args.screenshot):parser.error('--start-training需要--import-directory，且不能与退出式截图合用')
    if args.worker or args.command in ('train','predict','test','final'):
        from cloth_trainer.engine import train_request,predict_request,test_request,final_request,Cancelled
        from cloth_trainer.storage import read_json
        request_path=args.worker or args.request
        kind=args.kind if args.worker else args.command
        def emit(event):print(json.dumps(event,ensure_ascii=False),flush=True)
        try:
            request=read_json(request_path);cancel=lambda:request_path.with_suffix('.cancel').exists()
            if kind=='train':train_request(request,emit,cancel)
            elif kind=='final':final_request(request,emit,cancel)
            elif kind=='test':test_request(request,emit,cancel)
            else:predict_request(request,emit,cancel)
            return 0
        except Cancelled as exc:emit(dict(type='cancelled',message=str(exc)));return 130
        except Exception as exc:
            import traceback
            emit(dict(type='error',message=str(exc)));traceback.print_exc(file=sys.stderr);return 1
    if args.command=='check':
        from collections import Counter
        from cloth_trainer.data import import_dataset
        from cloth_trainer.modern_split import make_plan
        records,report=import_dataset(args.paths,signals=('by',))
        from cloth_trainer.admission import eligible_records
        readable=len(records);records=eligible_records(records)
        result=dict(report,readable_csv=readable,usable_csv=len(records),trials=len(records)*2,labels=dict(Counter(r.label for r in records)))
        try:
            plan=make_plan(records);result['partition_counts']=dict(development=len(plan['dev']),test=len(plan['test']),cv=[{k:len(v) for k,v in p.items()} for p in plan['cv']])
        except ValueError as exc:result['training_issue']=str(exc)
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if records else 1
    configure_qt_platform(screenshot=bool(args.screenshot))
    os.environ.setdefault('QT_API','pyqt5')
    try:
        from PyQt5 import QtWidgets as W,QtCore as C
        from cloth_trainer.tcn_ui import TCNTrainerWindow as TrainerWindow
    except ImportError as exc:
        print(f'依赖缺失：{exc}\n请先运行本目录的“安装依赖.sh”。',file=sys.stderr);return 2
    W.QApplication.setAttribute(C.Qt.AA_EnableHighDpiScaling,True)
    app=W.QApplication(sys.argv[:1]);app.setApplicationName('TCN 布料分类训练');app.setStyle('Fusion')
    window=TrainerWindow(args.store);window.show()
    project,directory=startup_sources(args.project,args.import_directory,screenshot=bool(args.screenshot))
    if project:
        try:window.read_project(project)
        except (ValueError,OSError,KeyError) as exc:window.notify('项目未能载入：'+str(exc))
    if args.experiment_name:window.task_name.setText(args.experiment_name)
    if directory:
        if directory.is_dir():C.QTimer.singleShot(0,lambda:window.import_paths([str(directory)]))
        else:window.notify('自动读取目录不存在：'+str(directory))
    if args.start_training:
        def start_imported():
            if window.import_jobs:C.QTimer.singleShot(100,start_imported);return
            started=window.start_imported_training(args.exclude_class)
            print(json.dumps(dict(type='desktop_training_started' if started else 'desktop_training_blocked',message=window.status.text()),ensure_ascii=False),flush=True)
        C.QTimer.singleShot(100,start_imported)
    if args.capture_after_start:
        def live_capture():
            if not window.live_history:C.QTimer.singleShot(1000,live_capture);return
            args.capture_after_start.parent.mkdir(parents=True,exist_ok=True);window.grab().save(str(args.capture_after_start))
        C.QTimer.singleShot(1000,live_capture)
    if args.screenshot:
        def capture():
            if window.import_jobs:C.QTimer.singleShot(100,capture);return
            window.navigation.setCurrentRow(args.page);app.processEvents()
            args.screenshot.parent.mkdir(parents=True,exist_ok=True);window.grab().save(str(args.screenshot));window.close()
        C.QTimer.singleShot(500,capture)
    return app.exec_()

if __name__=='__main__':raise SystemExit(main())
