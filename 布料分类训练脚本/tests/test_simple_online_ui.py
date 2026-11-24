from pathlib import Path
from PyQt5 import QtWidgets as W
from test_ui import app,wait,dispose_test_windows
from cloth_trainer.tcn_ui import TCNTrainerWindow
from cloth_trainer.storage import write_json


def test_daily_online_controls_hide_advanced_and_formal_test(app,tmp_path):
    w=TCNTrainerWindow(tmp_path/'models');w.navigation.setCurrentRow(3);w.show();app.processEvents()
    assert hasattr(w,'online_source_mode')
    assert w.online_source_mode.count()==3
    assert not w.online_advanced.isVisible() and not w.online_formal.isVisible()
    assert not w.online_input_group.isVisible()
    assert w.online_start.text()=='开始' and w.online_stop.text()=='停止'
    assert w.online_tabs.currentIndex()==1
    w.online_source_mode.setCurrentIndex(w.online_source_mode.findData('independent'))
    assert w.online_input_group.isVisible() and not w.online_truth_group.isVisible()
    w.online_source_mode.setCurrentIndex(w.online_source_mode.findData('live'))
    assert not w.online_input_group.isVisible() and w.online_truth_group.isVisible()
    assert w.online_scroll.horizontalScrollBar().maximum()==0
    w.close()


def test_one_start_routes_validation_newdata_live_and_clears_source(app,tmp_path,monkeypatch):
    task=tmp_path/'models'/'fixture';key='cera_example';path=task/key/'model.pt'
    write_json(task/'metadata.json',dict(models={key:dict(status='completed',architecture='cera',signal='by',fold=0,candidate=dict(window_size=900,spec=dict(revision='configurable-cera-v1')))},config=dict(cv_folds=1,tcn_window=900),split_plan=dict(cv=[dict(val=[0])]),sources=[dict(path=str(tmp_path/'val.csv'))]))
    w=TCNTrainerWindow(tmp_path/'empty');wait(app,lambda:not w.desktop_jobs.running());w.online_experiment.addItem('fixture',str(task))
    wait(app,lambda:w.online_model.currentData()==str(path))
    launched=[]
    monkeypatch.setattr(w,'_launch_online',lambda live=False,heldout=False:launched.append((live,heldout,list(w.online_paths),w.online_source)))
    w.start_online()
    assert launched[-1]==(False,False,[str((tmp_path/'val.csv').resolve())],'validation-tuning')
    w.online_source_mode.setCurrentIndex(w.online_source_mode.findData('independent'))
    assert not w.online_paths
    w._set_online_paths([str(tmp_path/'new.csv')]);w.start_online()
    assert launched[-1][0] is False and launched[-1][3]=='independent'
    w.online_source_mode.setCurrentIndex(w.online_source_mode.findData('live'));w.start_online()
    assert launched[-1][0] is True
    w.close()


def test_pause_remains_accessible_while_replay_running(app,tmp_path):
    task=tmp_path/'fixture';key='example';path=task/key/'model.pt'
    write_json(task/'metadata.json',dict(models={key:dict(status='completed',architecture='tcn',signal='by',fold=0,candidate=dict(window_size=900,spec=dict(revision='configurable-tcn-v1')))},config=dict(cv_folds=1,tcn_window=900),split_plan=dict(cv=[dict(val=[0])]),sources=[dict(path=str(tmp_path/'val.csv'))]))
    w=TCNTrainerWindow(tmp_path/'empty');wait(app,lambda:not w.desktop_jobs.running());w.online_experiment.addItem('fixture',str(task));wait(app,lambda:w.online_model.currentData()==str(path));w._set_online_paths([str(tmp_path/'val.csv')])
    w._launch_online()
    try:
        assert w.online_more.isEnabled() and w.online_pause.isEnabled()
        assert not w.online_source_mode.isEnabled()
    finally:
        w.stop_online();w.online_worker.wait(5000);app.processEvents();w.close()


def test_failed_policy_save_does_not_start_formal_test(app,tmp_path,monkeypatch):
    task=tmp_path/'fixture';key='example';path=task/key/'model.pt';path.parent.mkdir(parents=True);path.write_bytes(b'weight')
    write_json(task/'metadata.json',dict(models={key:dict(status='completed',architecture='tcn',signal='by',fold=0,candidate=dict(window_size=900,spec=dict(revision='configurable-tcn-v1')))},config=dict(cv_folds=1,tcn_window=900)))
    w=TCNTrainerWindow(tmp_path/'empty');wait(app,lambda:not w.desktop_jobs.running());w.online_experiment.addItem('fixture',str(task))
    monkeypatch.setattr(W.QFileDialog,'getSaveFileName',lambda *a,**k:(str(tmp_path/'missing'/'policy.json'),''))
    wait(app,lambda:w.online_model.currentData()==str(path))
    launched=[];monkeypatch.setattr(w,'evaluate_online_holdout',lambda:launched.append(True))
    w.freeze_and_evaluate_online()
    wait(app,lambda:not w.desktop_jobs.running())
    assert w.online_policy is None and not launched
    assert w.online_probability.isEnabled()
    w.close()


def test_policy_hash_runs_in_background_and_stale_selection_is_ignored(app,tmp_path,monkeypatch):
    import time,threading
    from PyQt5 import QtCore as C
    import cloth_trainer.online_ui as module
    w=TCNTrainerWindow(tmp_path/'models');wait(app,lambda:not w.desktop_jobs.running())
    w.online_model.addItem('fixture',str(tmp_path/'weight.pt'))
    monkeypatch.setattr(W.QFileDialog,'getSaveFileName',lambda *a,**k:(str(tmp_path/'policy.json'),''))
    main=threading.get_ident();threads=[];ticks=[];timer=C.QTimer();timer.setInterval(10);timer.timeout.connect(lambda:ticks.append(1));timer.start()
    def slow_freeze(path,cfg,**kwargs):
        threads.append(threading.get_ident());time.sleep(.2);return dict(fingerprint='f'*64)
    monkeypatch.setattr(module,'freeze_policy',slow_freeze)
    w.freeze_online_policy();wait(app,lambda:bool(threads));w.unlock_online_policy()
    wait(app,lambda:not w.desktop_jobs.running());timer.stop()
    assert threads and threads[0]!=main and len(ticks)>5
    assert w.online_policy is None
    w.close()
