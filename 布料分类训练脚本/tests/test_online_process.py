from dataclasses import replace
from PyQt5 import QtCore as C
from test_ui import app,wait
from test_prefix_cera import config
from test_data import fixture_csv
from cloth_trainer.engine import train_request
from cloth_trainer.storage import read_json
from cloth_trainer.online import OnlineConfig


def test_prefix_replay_in_process_streams_trace_and_locks_without_freezing(app,tmp_path):
    from cloth_trainer.online_process import OnlineProcess
    cfg=config();paths=[str(fixture_csv(tmp_path/'raw',c,i,speed=100)) for c in ('布A','布B') for i in range(3)]
    task=train_request(dict(paths=paths,config=cfg.to_dict(),output=str(tmp_path/'models')));meta=read_json(task/'metadata.json');key=next(iter(meta['models']))
    val=[meta['sources'][i]['path'] for i in meta['split_plan']['cv'][0]['val']]
    worker=OnlineProcess(task/key/'model.pt',val,OnlineConfig(window=128,stride=128,probability=0,margin=0,stable_seconds=0,consecutive=2,lock_on_confirm=True),source='validation-tuning')
    events=[];rows=[];errors=[];worker.observed.connect(events.append);worker.record.connect(rows.append);worker.failed.connect(errors.append)
    ticks=[];timer=C.QTimer();timer.setInterval(10);timer.timeout.connect(lambda:ticks.append(1));timer.start();worker.start();wait(app,lambda:not worker.isRunning());app.processEvents();timer.stop()
    assert not errors and len(rows)==2 and len(ticks)>10
    assert any(e['type']=='trace' for e in events) and any(e.get('status')=='locked' for e in events)
    assert all(r['completion'] for r in rows)
