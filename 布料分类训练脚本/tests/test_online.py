"""Online decisions must be functions of received prefixes and elapsed time."""
import json
import numpy as np
import pytest
from cloth_trainer.online import OnlineConfig, OnlinePredictor, csv_packets, decode_packet, summarize_records


def predictor(**kw):
    return OnlinePredictor(labels=['a','b'],predict_proba=lambda x: np.array([.98,.02]),
                           config=OnlineConfig(window=4,stride=2,stable_seconds=.012,**kw))


def packet(i,**kw):
    return dict(type='sample',capture_id='one',stage='step_003',t_s=i/333,x_mm=60,
                by_uT=5.,baseline_by_uT=2.,valid=True,motor_fault=False,**kw)


def feed(p,n,start=0):
    return [e for i in range(start,n) for e in p.feed(packet(i)) if e['type']=='prediction']


def test_no_padding_stride_and_elapsed_stability():
    p=predictor();rows=feed(p,13)
    assert [r['sample_count'] for r in rows]==[4,6,8,10,12]
    assert rows[0]['status']=='judging'
    assert rows[1]['status']=='judging'
    assert rows[2]['status']=='confirmed'
    assert rows[2]['t_s']-rows[0]['t_s']>=.012
    assert rows[0]['probabilities']==pytest.approx([.98,.02])


def test_prefix_invariance_and_label_independence():
    first=predictor();prefix=feed(first,9)
    other=predictor();full=feed(other,25)
    assert prefix==full[:len(prefix)]
    a=predictor();b=predictor();a.true_label='a';b.true_label='b'
    assert feed(a,14)==feed(b,14)


@pytest.mark.parametrize('bad', [dict(t_s=.2),dict(t_s=0.),dict(valid=False),dict(motor_fault=True)])
def test_faults_and_strict_time_gaps_reset_warmup(bad):
    p=predictor();feed(p,9)
    q=packet(9);q.update(bad)
    events=p.feed(q)
    assert any(e['status']=='error' for e in events)
    assert p.sample_count==0
    assert p.confirmed_label is None


def test_three_point_spike_rejected_only_after_arrival():
    p=predictor();feed(p,3)
    q=packet(3);q['by_uT']=2005
    assert not any(e['status']=='error' for e in p.feed(q))
    assert any(e['status']=='error' for e in p.feed(packet(4)))
    assert p.sample_count==0


def test_reversals_continue_after_confirmation():
    p=predictor();feed(p,13)
    assert p.confirmed_label=='a'
    p.predict_proba=lambda x:np.array([.01,.99])
    rows=feed(p,27,13)
    assert rows[0]['status']=='judging'
    assert rows[-1]['confirmed_label']=='b'
    assert p.flip_count==1


def test_wire_validation_and_end_without_confirmation():
    assert decode_packet(json.dumps(packet(1)).encode())['by_uT']==5
    with pytest.raises(ValueError):decode_packet(b'{bad')
    with pytest.raises(ValueError):decode_packet(b'{"type":"command"}')
    p=predictor();feed(p,3)
    assert p.feed(dict(type='end'))[-1]['status']=='end-unconfirmed'


def test_csv_baseline_prefix_and_metadata_roles(tmp_path):
    (tmp_path/'metadata.json').write_text(json.dumps({'label':'truth','baseline_B0_uT':{'By_uT':2},'context':{'config':{'x_ppmm':250}}}))
    path=tmp_path/'a.csv';path.write_text('stage,sensor_t_s,motor_x_pulses,By_uT,dBy_uT,sample_valid\nstep_003,0,15000,5,3,1\nstep_003,0.003,15000,6,4,1\n')
    rows=list(csv_packets(path))
    assert rows[0]['baseline_by_uT']==2
    assert rows[0]['by_uT']==5
    assert rows[0]['x_mm']==60
    assert rows[-1]['type']=='end'
    assert 'true_label' not in rows[0]
    summary=summarize_records([{'true_label':'a','prediction':'a','status':'correct'}, {'true_label':'a','prediction':None,'status':'abstain'}])
    assert summary['counts']=={'correct':1,'error':0,'abstain':1}
    assert summary['classes']['a']['count']==2


def test_worker_replay_produces_direction_evaluation_without_truth_inference(tmp_path,monkeypatch):
    from cloth_trainer.online_ui import OnlineWorker
    from cloth_trainer import modern_engine
    import torch
    class Fixed(torch.nn.Module):
        def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1))
        def forward(self,x,lengths):return torch.tensor([[5.,0.]])+self.anchor*0
    monkeypatch.setattr(modern_engine,'load_online_bundle',lambda path:dict(_model=Fixed(),labels=['a','b'],scaler={'mean':[0],'std':[1]},config={}))
    (tmp_path/'metadata.json').write_text(json.dumps({'label':'a','baseline_B0_uT':{'By_uT':2}}))
    path=tmp_path/'a.csv'
    path.write_text('stage,sensor_t_s,motor_x_pulses,By_uT\n'+''.join(f'step_003,{i/333},15000,5\n' for i in range(14)))
    worker=OnlineWorker('model',[str(path)],OnlineConfig(window=4,stride=2,stable_seconds=.012))
    rows=[];issues=[];worker.record.connect(rows.append);worker.failed.connect(issues.append);worker.run()
    assert not issues
    assert rows[0]['status']=='correct'
    assert rows[0]['first_confirmation']['label']=='a'
    assert rows[0]['direction']=='out'


@pytest.mark.parametrize('settings',[{'stable_seconds':float('nan')},{'window':2.5},{'stride':True},{'probability':float('inf')}])
def test_gate_rejects_nonfinite_and_noninteger_settings(settings):
    with pytest.raises(ValueError):OnlineConfig(**settings)


def test_quality_error_cannot_leave_stale_confirmation_at_end():
    p=predictor();feed(p,13);assert p.confirmed_label=='a'
    q=packet(13);q['valid']=False;p.feed(q)
    end=p.feed({'type':'end'})[-1]
    assert end['status']=='end-unconfirmed'
    assert end['prediction'] is None
    assert end['first_confirmation']['label']=='a'


def test_csv_other_axis_spike_is_restored_but_unknown_flag_is_not(tmp_path):
    (tmp_path/'metadata.json').write_text(json.dumps({'baseline_B0_uT':{'By_uT':2}}))
    p=tmp_path/'a.csv';p.write_text('stage,sensor_t_s,motor_x_pulses,By_uT,sample_valid,invalid_reason\nstep_003,0,15000,5,0,isolated_spike\nstep_003,0.003,15000,5,0,transport_error\n')
    rows=list(csv_packets(p));assert rows[0]['valid'];assert not rows[1]['valid']


def test_frozen_policy_binds_model_hash_and_settings(tmp_path):
    from cloth_trainer.online import freeze_policy,validate_policy
    model=tmp_path/'model.pt';model.write_bytes(b'frozen model')
    policy=freeze_policy(model,OnlineConfig())
    assert validate_policy(policy,model)==OnlineConfig()
    altered=dict(policy);altered['settings']=dict(policy['settings'],probability=.2)
    with pytest.raises(ValueError):validate_policy(altered,model)
    model.write_bytes(b'changed')
    with pytest.raises(ValueError):validate_policy(policy,model)


def test_crop_exit_preserves_last_valid_confirmation():
    p=predictor();feed(p,13)
    q=packet(13);q['x_mm']=145;p.feed(q)
    assert p.feed({'type':'end'})[-1]['prediction']=='a'


def test_model_window_mismatch_is_blocked():
    with pytest.raises(ValueError):OnlinePredictor({'labels':['a','b'],'candidate':{'window_size':450}},config=OnlineConfig(),predict_proba=lambda x:[1,0])


def test_first_confirmation_metrics_include_abstentions():
    summary=summarize_records([{'true_label':'a','prediction':'b','status':'error','first_confirmation':{'label':'a'}},
                               {'true_label':'a','prediction':None,'status':'abstain','first_confirmation':None}])
    assert summary['first_confirmation_counts']=={'correct':1,'error':0,'abstain':1,'unlabelled':0}


def test_worker_finalizes_outgoing_before_live_stage_reset(monkeypatch):
    from cloth_trainer.online_ui import OnlineWorker
    from cloth_trainer.online import OnlineConfig
    import torch
    class Fixed(torch.nn.Module):
        def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1))
        def forward(self,x,lengths):return torch.tensor([[5.,0.]])+self.anchor*0
    worker=OnlineWorker('unused',[],OnlineConfig(window=4,stride=2,stable_seconds=.012),include_return=True)
    packets=[packet(i) for i in range(14)]+[dict(type='reset',capture_id='one',stage='step_004',reason='stage_changed')]
    packets += [dict(packet(i),stage='step_004') for i in range(14,28)]+[dict(type='end')]
    records=[];worker.record.connect(records.append)
    worker._stream(dict(_model=Fixed(),labels=['a','b'],scaler={'mean':[0],'std':[1]},config={}),packets,'live','a')
    assert len(records)==2
    assert [r['status'] for r in records]==['correct','correct']


def test_tcp_ndjson_stream_uses_same_predictor_and_handles_partial_packets():
    import socket
    import threading
    from cloth_trainer.online import tcp_packets
    listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1);port=listener.getsockname()[1]
    def publish():
        with listener:
            connection,_=listener.accept()
            with connection:
                content=b''.join(json.dumps(packet(i)).encode()+b'\n' for i in range(14))
                connection.sendall(content[:23]);connection.sendall(content[23:]);connection.sendall(b'{"type":"end"}\n')
    thread=threading.Thread(target=publish);thread.start()
    p=predictor();rows=[]
    for q in tcp_packets(port=port):rows.extend(p.feed(q))
    thread.join(2)
    assert any(r['status']=='confirmed' for r in rows)
    assert rows[-1]['prediction']=='a'


def test_online_page_build_and_freeze_controls(tmp_path):
    from PyQt5 import QtWidgets as W
    from cloth_trainer.online_ui import OnlineMixin
    app=W.QApplication.instance() or W.QApplication([])
    class Surface(OnlineMixin,W.QWidget):
        def __init__(self):super().__init__();self.store=tmp_path
    surface=Surface();page=surface.build_online();page.resize(900,700);page.show();app.processEvents()
    assert surface.online_stride.currentData()==900
    assert surface.online_probability.value()==.95
    assert surface.online_stability.value()==2.7
    assert not surface.online_smoothing.isChecked()
    surface._lock_online_policy(True)
    assert not surface.online_probability.isEnabled()
    assert not surface.online_model.isEnabled()
    page.close();surface.close()


def test_worker_outgoing_only_default_and_optional_return():
    from cloth_trainer.online_ui import OnlineWorker
    import torch
    class Fixed(torch.nn.Module):
        def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1))
        def forward(self,x,lengths):return torch.tensor([[5.,0.]])+self.anchor*0
    worker=OnlineWorker('unused',[],OnlineConfig(window=4,stride=2,stable_seconds=.012))
    packets=[packet(i) for i in range(14)]+[dict(packet(i),stage='step_004') for i in range(14,28)]+[dict(type='end')]
    rows=[];worker.record.connect(rows.append)
    worker._stream(dict(_model=Fixed(),labels=['a','b'],scaler={'mean':[0],'std':[1]},config={}),packets,'live','a')
    assert [r['direction'] for r in rows]==['out']


def test_holdout_worker_requires_frozen_policy_and_detects_renamed_holdout(tmp_path,monkeypatch):
    from cloth_trainer.online_ui import OnlineWorker
    from cloth_trainer.online import freeze_policy
    from cloth_trainer import modern_engine
    from cloth_trainer.data import file_hash,metadata_hash
    import torch
    class Fixed(torch.nn.Module):
        def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1))
        def forward(self,x,lengths):return torch.tensor([[5.,0.]])+self.anchor*0
    (tmp_path/'metadata.json').write_text(json.dumps({'label':'a','baseline_B0_uT':{'By_uT':2}}))
    path=tmp_path/'holdout.csv';path.write_text('stage,sensor_t_s,motor_x_pulses,By_uT\n'+''.join(f'step_003,{i/333},15000,5\n' for i in range(14)))
    model=tmp_path/'model.pt';model.write_bytes(b'model')
    bundle=dict(_model=Fixed(),labels=['a','b'],scaler={'mean':[0],'std':[1]},config={},sources=[dict(path=str(path),sha256=file_hash(path),metadata_sha256=metadata_hash(path))],split_audit=dict(test=[0],val=[]))
    monkeypatch.setattr(modern_engine,'load_online_bundle',lambda p:bundle)
    renamed=tmp_path/'renamed.csv';renamed.write_bytes(path.read_bytes())
    cfg=OnlineConfig(window=4,stride=2,stable_seconds=.012)
    explorer=OnlineWorker(model,[str(renamed)],cfg);issues=[];explorer.failed.connect(issues.append);explorer.run()
    assert issues and '留出' in issues[0]
    worker=OnlineWorker(model,[],cfg,source='validation-tuning',heldout=True,policy=freeze_policy(model,cfg));rows=[];issues=[]
    worker.record.connect(rows.append);worker.failed.connect(issues.append);worker.run()
    assert not issues
    assert rows[0]['source']=='holdout-frozen'
    assert rows[0]['status']=='correct'
    assert rows[0]['policy_fingerprint']
    assert file_hash(path)==bundle['sources'][0]['sha256']


def test_no_future_interpolation_is_used_for_grid_values():
    seen=[]
    p=OnlinePredictor(labels=['a','b'],predict_proba=lambda values:(seen.append(values.copy()) or np.array([1.,0.])),config=OnlineConfig(window=4,stride=4,stable_seconds=0))
    for t,by in [(0.,2.),(.009,12.),(.012,22.),(.015,32.)]:
        p.feed(dict(packet(0),t_s=t,by_uT=by))
    assert seen[0][:,0].tolist()==[0.,0.,0.,10.]


@pytest.mark.parametrize('override',[{'signal':'bx'},{'architecture':'gru'},{'config':{'data':{'preprocess_version':'legacy-v2'}}}])
def test_online_rejects_incompatible_model_signal_or_preprocessing(override):
    bundle={'labels':['a','b'],'signal':'by','architecture':'tcn','config':{}}
    bundle.update(override)
    with pytest.raises(ValueError):OnlinePredictor(bundle,predict_proba=lambda x:[1.,0.])


def test_worker_rejects_changed_settings_under_frozen_policy(tmp_path,monkeypatch):
    from cloth_trainer.online_ui import OnlineWorker
    from cloth_trainer.online import freeze_policy
    from cloth_trainer import modern_engine
    model=tmp_path/'model.pt';model.write_bytes(b'frozen')
    monkeypatch.setattr(modern_engine,'load_online_bundle',lambda p:{'labels':['a','b']})
    worker=OnlineWorker(model,[],OnlineConfig(probability=.1),policy=freeze_policy(model,OnlineConfig()))
    issues=[];worker.failed.connect(issues.append);worker.run()
    assert issues and '策略' in issues[0]


@pytest.mark.parametrize('delta',['99','nan'])
def test_csv_by_provenance_error_cannot_be_restored_as_other_axis_spike(tmp_path,delta):
    (tmp_path/'metadata.json').write_text(json.dumps({'baseline_B0_uT':{'By_uT':2}}))
    path=tmp_path/'bad.csv';path.write_text('stage,sensor_t_s,motor_x_pulses,By_uT,dBy_uT,sample_valid,invalid_reason\nstep_003,0,15000,5,'+delta+',0,isolated_spike\n')
    row=next(csv_packets(path))
    assert row['valid'] is False
    assert row['invalid_reason']=='by_baseline_mismatch'
    assert predictor().feed(row)[0]['status']=='error'


@pytest.mark.parametrize('changed',['csv','metadata'])
def test_validation_tuning_rejects_changed_frozen_source(tmp_path,monkeypatch,changed):
    from cloth_trainer.online_ui import OnlineWorker
    from cloth_trainer import modern_engine
    from cloth_trainer.data import file_hash,metadata_hash
    meta=tmp_path/'metadata.json';meta.write_text(json.dumps({'baseline_B0_uT':{'By_uT':2}}))
    path=tmp_path/'val.csv';path.write_text('stage,sensor_t_s,motor_x_pulses,By_uT\nstep_003,0,15000,5\n')
    source=dict(path=str(path),sha256=file_hash(path),metadata_sha256=metadata_hash(path))
    bundle=dict(labels=['a','b'],sources=[source],split_audit=dict(test=[],val=[0]))
    monkeypatch.setattr(modern_engine,'load_online_bundle',lambda p:bundle)
    if changed=='csv':path.write_text(path.read_text().replace('15000,5','15000,6'))
    else:meta.write_text(json.dumps({'baseline_B0_uT':{'By_uT':3}}))
    worker=OnlineWorker('unused',[str(path)],OnlineConfig(),source='validation-tuning');issues=[];rows=[]
    worker.failed.connect(issues.append);worker.record.connect(rows.append);worker.run()
    assert issues and 'CSV或元数据已改变' in issues[0]
    assert rows==[]


def test_holdout_first_policy_lock_persists_across_workers(tmp_path,monkeypatch):
    from cloth_trainer.online_ui import OnlineWorker
    from cloth_trainer.online import freeze_policy
    from cloth_trainer import modern_engine
    model=tmp_path/'model.pt';model.write_bytes(b'model')
    heldout=tmp_path/'test.csv'
    bundle=dict(labels=['a','b'],sources=[dict(path=str(heldout),sha256='a'*64,metadata_sha256='b'*64)],split_audit=dict(test=[0],val=[]))
    monkeypatch.setattr(modern_engine,'load_online_bundle',lambda p:bundle)
    cfg=OnlineConfig();policy=freeze_policy(model,cfg)
    first=OnlineWorker(model,[],cfg,heldout=True,policy=policy);issues=[];first.failed.connect(issues.append);first.run()
    lock=tmp_path/'online_holdout_lock.json'
    assert lock.is_file()
    saved=lock.read_bytes()
    assert json.loads(saved)['policy_fingerprint']==policy['fingerprint']
    # First attempt binds before opening even a missing heldout file; retry keeps the lock.
    retry=OnlineWorker(model,[],cfg,heldout=True,policy=policy);retry.run();assert lock.read_bytes()==saved
    changed=OnlineConfig(probability=.99)
    worker=OnlineWorker(model,[],changed,heldout=True,policy=freeze_policy(model,changed));issues=[];worker.failed.connect(issues.append);worker.run()
    assert issues and '首次留出' in issues[0]
    assert lock.read_bytes()==saved
    bundle['sources'][0]['sha256']='c'*64
    stale=OnlineWorker(model,[],cfg,heldout=True,policy=policy);issues=[];stale.failed.connect(issues.append);stale.run()
    assert issues and '首次留出' in issues[0]
    lock.write_text('{bad')
    malformed=OnlineWorker(model,[],cfg,heldout=True,policy=policy);issues=[];malformed.failed.connect(issues.append);malformed.run()
    assert issues and '留出锁' in issues[0]


def test_live_manual_idle_is_waiting_not_quality_failure_and_does_not_flood():
    p=predictor();events=[]
    for i in range(20):
        q=packet(i);q.update(stage='manual_idle',x_mm=0.,valid=False,invalid_reason='stage_not_classifiable')
        events.extend(p.feed(q))
    assert len(events)==1 and events[0]['type']=='waiting' and events[0]['status']=='waiting'
    assert p.quality_resets==0 and p.sample_count==0
    predictions=feed(p,34,start=20)
    assert predictions and predictions[-1]['confirmed_label']=='a'


def test_not_recording_waits_but_actual_faults_remain_errors():
    p=predictor();q=packet(0);q.update(valid=False,invalid_reason='not_recording')
    assert p.feed(q)[0]['status']=='waiting'
    q=packet(1);q.update(stage='manual_idle',valid=False,invalid_reason='motor_fault',motor_fault=True)
    assert p.feed(q)[0]['status']=='error'
