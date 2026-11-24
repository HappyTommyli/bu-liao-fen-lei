from pathlib import Path
from PyQt5 import QtCore as C
from test_ui import app,wait
from cloth_trainer.storage import write_json


def test_catalog_worker_only_returns_compact_metadata_and_stale_job_is_ignored(app,tmp_path):
    from cloth_trainer.desktop_jobs import DesktopJobs
    root=tmp_path/'models';p=root/'a'/'metadata.json'
    write_json(p,dict(name='a',status='completed',created_at='2026-11-23',config={},models={},screening=dict(padding='x'*12000000)))
    jobs=DesktopJobs();ticks=[];timer=C.QTimer();timer.setInterval(10);timer.timeout.connect(lambda:ticks.append(1));timer.start()
    received=[];jobs.submit('selection','catalog',dict(root=str(root)),lambda x:received.append(('old',x)))
    jobs.submit('selection','metadata',dict(path=str(p)),lambda x:received.append(('new',x)))
    wait(app,lambda:not jobs.running())
    timer.stop()
    assert [x[0] for x in received]==['new']
    assert 'padding' not in str(received[0][1]) and len(ticks)>=2
    assert '正在' in jobs.last_stage or '完成' in jobs.last_stage


def test_cancel_desktop_job_does_not_deliver_late_result(app,tmp_path):
    from cloth_trainer.desktop_jobs import DesktopJobs
    p=tmp_path/'metadata.json';write_json(p,dict(name='a',models={},screening=dict(padding='x'*12000000)))
    jobs=DesktopJobs();got=[];jobs.submit('load','metadata',dict(path=str(p)),got.append);jobs.cancel_all()
    wait(app,lambda:not jobs.running());assert not got


def test_dataset_save_load_cancels_without_replacing_ui_state(app,tmp_path):
    from cloth_trainer.tcn_ui import TCNTrainerWindow
    w=TCNTrainerWindow(tmp_path/'models')
    before=list(w.records);w.load_dataset_version(tmp_path/'missing');w.desktop_jobs.cancel('dataset')
    wait(app,lambda:not w.desktop_jobs.running());assert w.records==before and w.snapshot_path is None
    w.close()


def test_interrupted_task_update_preserves_full_frozen_audit(tmp_path):
    from cloth_trainer.desktop_worker import execute
    from cloth_trainer.storage import read_json
    p=tmp_path/'metadata.json';write_json(p,dict(status='running',models={},admission=dict(original='retained'),screening=dict(original='retained')))
    result=execute('finish-task',dict(path=str(p),cancelled=True,error='cancel'))
    assert result['status']=='cancelled' and read_json(p)['admission']==dict(original='retained')


def test_thread_callback_can_start_next_job_and_errors_are_reported(app):
    from cloth_trainer.desktop_jobs import DesktopJobs
    jobs=DesktopJobs();seen=[];errors=[];jobs.failed.connect(errors.append)
    def done(value):
        assert 'dataset' not in jobs.active
        seen.append(value)
        raise ValueError('callback failure')
    jobs.submit_thread('dataset',lambda c:1,done,'保存版本')
    wait(app,lambda:not jobs.running())
    assert seen==[1] and errors and 'callback failure' in errors[0]


def test_finished_replay_keeps_event_drain_bounded(app,tmp_path,monkeypatch):
    from cloth_trainer.tcn_ui import TCNTrainerWindow
    w=TCNTrainerWindow(tmp_path/'models');wait(app,lambda:not w.desktop_jobs.running())
    events=[];monkeypatch.setattr(w,'online_receive_event',events.append)
    w._online_display_queue=[dict(type='fixture') for _ in range(1000)]
    w._online_finished()
    assert len(events)<=150 and w._online_display_queue
    assert not w.online_start.isEnabled()
    wait(app,lambda:not w._online_display_queue and w.online_start.isEnabled())
    assert len(events)==1000
    w.close()


def test_project_snapshot_applies_selection_after_async_load(app,tmp_path):
    from test_window_pool_upgrade import make_snapshot
    from cloth_trainer.tcn_ui import TCNTrainerWindow,focused_config
    p,records,_=make_snapshot(tmp_path)
    chosen={records[0].path,records[2].path}
    project=tmp_path/'project.json'
    write_json(project,dict(format='cloth-project-v3',config=focused_config().to_dict(),store=str(tmp_path/'models'),snapshot=str(p),sources=[dict(path=r.path,selected=r.path in chosen) for r in records]))
    w=TCNTrainerWindow(tmp_path/'models');w.read_project(project)
    wait(app,lambda:not w.desktop_jobs.running())
    assert {r.path for r in w.selected_records()}==chosen
    w.close()
