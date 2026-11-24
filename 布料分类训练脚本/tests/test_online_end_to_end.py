from test_ui import app,wait,dispose_test_windows
from test_data import fixture_csv
from cloth_trainer.modern_config import ModernConfig
from cloth_trainer.tcn_models import default_spec
from cloth_trainer.engine import train_request
from cloth_trainer.storage import read_json
from cloth_trainer.online import OnlineConfig
from cloth_trainer.online_ui import OnlineWorker


def test_single_overlap_tcn_best_weight_replays_validation_without_final_retrain(app,tmp_path,monkeypatch):
    paths=[fixture_csv(tmp_path/'raw',c,i,speed=100) for c in ('布A','布B') for i in range(3)]
    cfg=ModernConfig(device='cpu',workers=0,epochs=1,batch_size=8,micro_batch=4,architectures=('tcn',),signals=('by',),tcn_spec=dict(default_spec(16,1),pooling='mean_std'),tcn_window=128,tcn_stride=64,cv_folds=1)
    task=train_request(dict(paths=[str(p) for p in paths],config=cfg.to_dict(),output=str(tmp_path/'models')))
    meta=read_json(task/'metadata.json');before=(task/'metadata.json').read_bytes()
    assert len(meta['models'])==1 and meta['final_model'] is None
    key=next(iter(meta['models']));validation=[meta['sources'][i]['path'] for i in meta['split_plan']['cv'][0]['val']]
    worker=OnlineWorker(task/key/'model.pt',validation,OnlineConfig(window=128,stride=64,probability=0,margin=0,stable_seconds=0),source='validation-tuning')
    records=[];events=[];errors=[]
    worker.record.connect(records.append);worker.observed.connect(events.append);worker.failed.connect(errors.append)
    worker.start();wait(app,lambda:not worker.isRunning());app.processEvents()
    assert not errors and len(records)==2
    assert all(r['source']=='validation-tuning' and r['direction']=='out' for r in records)
    assert all(r['first_confirmation'] and r['confirmation_elapsed_s']>0 for r in records)
    assert len([e for e in events if e['type']=='prediction'])>=4
    assert (task/'metadata.json').read_bytes()==before
    worker.deleteLater()

    # Actual acquisition publisher and frame schema, ephemeral loopback only.
    import importlib.util,csv,time,pytest
    from pathlib import Path
    from types import SimpleNamespace
    from cloth_trainer.online import tcp_packets
    source=Path(__file__).resolve().parents[2]/'xz-control/xz_workbench/realtime.py'
    if not source.exists():pytest.skip('采集程序源码未包含在独立训练工作台发行版中')
    spec=importlib.util.spec_from_file_location('tested_acquisition_realtime',source);realtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(realtime)
    publisher=realtime.Publisher(port=0);assert publisher.start()
    monkeypatch.setattr('cloth_trainer.online_ui.tcp_packets',lambda cancelled:tcp_packets(port=publisher.address[1],cancelled=cancelled))
    live=OnlineWorker(task/key/'model.pt',[],OnlineConfig(window=128,stride=64,probability=0,margin=0,stable_seconds=0),live=True,live_truth='布A')
    live_records=[];live_errors=[];live.record.connect(live_records.append);live.failed.connect(live_errors.append)
    try:
        live.start();wait(app,lambda:publisher.clients>0 or bool(live_errors));assert not live_errors
        frames=realtime.FrameStream()
        with paths[0].open(encoding='utf-8-sig',newline='') as f:
            for i,row in enumerate(csv.DictReader(f)):
                if row['stage']!='step_003':continue
                frame=SimpleNamespace(by=float(row['By_uT']),unwrapped_us=round(float(row['sensor_t_s'])*1e6),segment=0)
                packets=frames.frame(frame,capture_id='synthetic',stage='step_003',status=dict(x=float(row['motor_x_pulses']),fault='NONE'),x_ppmm=250,baseline=(0,0,0),active=True)
                for packet in packets:publisher.publish(packet)
                if i%16==0:app.processEvents();time.sleep(.01)
        publisher.publish(frames.end('complete'))
        wait(app,lambda:bool(live_records) or bool(live_errors));assert not live_errors
        assert live_records[0]['true_label']=='布A' and live_records[0]['first_confirmation']
    finally:
        publisher.stop();live.stop();wait(app,lambda:not live.isRunning());live.deleteLater()
