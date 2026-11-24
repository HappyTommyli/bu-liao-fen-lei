"""Fault injection runs only on tiny synthetic signals, never acquisition data."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from cloth_trainer import engine
from cloth_trainer.classifiers import SegmentClassifier
from cloth_trainer.config import TrainConfig
from cloth_trainer.data import Trial
from cloth_trainer.storage import read_json,write_json
from cloth_trainer.recovery import load_recovery


@pytest.fixture
def signals():
    torch.set_num_threads(1)
    def trial(i):
        t=np.arange(24)/333
        x=(np.sin(t*90+i)*2+i*3+10).astype(np.float32)[:,None]
        return Trial(str(i),f'/synthetic/capture_{i//2}.csv',f'hash{i//2}',f'group{i//2}','A' if i<4 else 'B','out' if i%2==0 else 'back',x,t,t,t,x,(0,24))
    return [trial(i) for i in range(8)]


def fit(folder,signals,emit=lambda e:None,**kwargs):
    folder.mkdir(exist_ok=True)
    cfg=kwargs.pop('cfg',TrainConfig(epochs=4,batch_size=2,hidden_size=32,dropout=.2,scheduler_patience=1))
    return engine._fit(signals[:6],signals[6:],['A','B'],cfg,kwargs.pop('architecture','tcn'),kwargs.pop('readout','mean'),42,emit,lambda:False,folder,**kwargs)


def test_bad_gradient_stops_before_optimizer_and_writes_context(tmp_path,signals,monkeypatch):
    updates=[];step=torch.optim.AdamW.step;forward=SegmentClassifier.forward
    def register(self,x,lengths):
        logits=forward(self,x,lengths)
        if self.training:logits.register_hook(lambda g:torch.full_like(g,float('inf')))
        return logits
    def observed(self,*args,**kwargs):updates.append(True);return step(self,*args,**kwargs)
    monkeypatch.setattr(SegmentClassifier,'forward',register);monkeypatch.setattr(torch.optim.AdamW,'step',observed)
    with pytest.raises(ValueError,match='非有限'):fit(tmp_path,signals)
    assert not updates
    report=read_json(tmp_path/'numerical_failure.json')
    assert report['stage']=='gradients' and report['epoch']==1 and report['batch']==1
    assert report['trials'][0]['path'].startswith('/synthetic/') and report['input']['nonfinite']==0
    assert report['nonfinite_tensors'] and report['recovery_epoch']==0
    assert report['scaler']['std'][0]>0 and report['gradient_norm'] is None


@pytest.mark.parametrize('boundary',['input','logits','optimizer','validation'])
def test_numerical_boundaries_preserve_last_good_recovery(tmp_path,signals,monkeypatch,boundary):
    if boundary=='input':signals[0].values[0,0]=np.nan
    elif boundary=='optimizer':
        step=torch.optim.AdamW.step
        def corrupt(self,*args,**kwargs):
            result=step(self,*args,**kwargs)
            with torch.no_grad():self.param_groups[0]['params'][0].flatten()[0]=float('nan')
            return result
        monkeypatch.setattr(torch.optim.AdamW,'step',corrupt)
    else:
        forward=SegmentClassifier.forward
        def corrupt(self,x,lengths):
            logits=forward(self,x,lengths)
            if boundary=='logits' or not self.training:return logits*float('nan')
            return logits
        monkeypatch.setattr(SegmentClassifier,'forward',corrupt)
    with pytest.raises(ValueError,match='非有限'):fit(tmp_path,signals)
    report=read_json(tmp_path/'numerical_failure.json')
    assert report['stage']=={'input':'input','logits':'logits','optimizer':'optimizer','validation':'validation_logits'}[boundary]
    assert report['trials'] and report['epoch']==1
    assert report['recovery_epoch']==(None if boundary=='input' else 0)
    if boundary!='input':
        assert report['recovery_epoch']==0
        assert (tmp_path/'recovery.ckpt').is_file()


def compare_state(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:
            if k!='seconds':compare_state(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare_state(x,y)
    else:assert a==b


@pytest.mark.parametrize('architecture',['gru','tcn','cera'])
@pytest.mark.parametrize('readout',['mean','last'])
def test_epoch_resume_matches_continuous_training(tmp_path,signals,monkeypatch,architecture,readout):
    # Real validation forward passes, fixed loss solely to exercise scheduler decay.
    evaluate=engine.evaluate
    def plateau(*args,**kwargs):rows,_=evaluate(*args,**kwargs);return rows,1.
    monkeypatch.setattr(engine,'evaluate',plateau)
    direct=fit(tmp_path/'direct',signals,architecture=architecture,readout=readout)
    def interrupt(e):
        if e['type']=='epoch' and e['epoch']==2:raise engine.Cancelled('synthetic interruption')
    with pytest.raises(engine.Cancelled):fit(tmp_path/'interrupted',signals,emit=interrupt,architecture=architecture,readout=readout)
    events=[];resumed=fit(tmp_path/'interrupted',signals,emit=events.append,architecture=architecture,readout=readout)
    assert [e['epoch'] for e in events if e['type']=='epoch']==[3,4]
    restore=next(e for e in events if e['type']=='restored')
    assert restore['epoch']==2 and len(restore['history'])==2
    compare_state(direct[0].state_dict(),resumed[0].state_dict());compare_state(direct[2],resumed[2]);assert direct[3]==resumed[3]
    # Compare current epoch optimizer/RNG/best state too, not just exported best.
    import io
    def state(folder):return torch.load(io.BytesIO((folder/'recovery.ckpt').read_bytes().split(b'\n',1)[1]),weights_only=True)['state']
    a=state(tmp_path/'direct');b=state(tmp_path/'interrupted');compare_state(a,b)
    assert a['optimizer']['param_groups'][0]['lr']<.001


def test_resume_refuses_config_or_input_change_and_corrupt_state(tmp_path,signals):
    fit(tmp_path,signals)
    with pytest.raises(ValueError,match='关联|匹配'):fit(tmp_path,signals,cfg=TrainConfig(epochs=4,batch_size=2,hidden_size=32,dropout=.2,lr=.0003))
    changed=list(signals);changed[0]=replace(signals[0],values=signals[0].values+1)
    with pytest.raises(ValueError,match='关联|匹配'):fit(tmp_path,changed)
    p=tmp_path/'recovery.ckpt';p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='校验|损坏'):fit(tmp_path,signals)


@pytest.mark.parametrize('early_stop',[False,True])
def test_finished_epoch_resume_does_not_add_updates(tmp_path,signals,early_stop):
    cfg=TrainConfig(epochs=4,batch_size=2,hidden_size=32,lr=1e-20,early_stopping=early_stop,patience=1)
    first=fit(tmp_path,signals,cfg=cfg);events=[];again=fit(tmp_path,signals,cfg=cfg,emit=events.append)
    assert len(first[2])==(2 if early_stop else 4)
    assert not any(e['type']=='epoch' for e in events)
    compare_state(first[0].state_dict(),again[0].state_dict())


def test_legacy_missing_checkpoint_restarts_with_notice_and_preserves_history(tmp_path,signals):
    old=[dict(epoch=99,loss=2.)];write_json(tmp_path/'history.json',old);events=[]
    fit(tmp_path,signals,emit=events.append)
    assert read_json(tmp_path/'history-before-restart.json')==old
    assert any(e['type']=='restarted' and '第1轮' in e['message'] for e in events)


def test_finite_gradients_with_overflowing_norm_update_safely(tmp_path,signals,monkeypatch):
    forward=SegmentClassifier.forward;step=torch.optim.AdamW.step;norms=[]
    def enormous(self,x,lengths):
        logits=forward(self,x,lengths)
        if self.training:logits.register_hook(lambda g:torch.full_like(g,1e30))
        return logits
    def observed(self,*args,**kwargs):
        grads=[p.grad for group in self.param_groups for p in group['params'] if p.grad is not None]
        norms.append(float(torch.stack([g.double().norm() for g in grads]).norm()))
        return step(self,*args,**kwargs)
    monkeypatch.setattr(SegmentClassifier,'forward',enormous);monkeypatch.setattr(torch.optim.AdamW,'step',observed)
    model,_,history,_=fit(tmp_path,signals)
    assert norms and all(0<n<=1+1e-6 for n in norms)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    assert not (tmp_path/'numerical_failure.json').exists()
    assert all(row['gradient_norm_max']>1e19 and row['gradient_clip_fraction']==1 for row in history)
    assert all(0<row['gradient_clip_coefficient_min']<1 and row['gradient_clipped_norm_max']<=1+1e-6 for row in history)


def test_second_epoch_failure_keeps_checkpoint_and_resumes_exactly(tmp_path,signals,monkeypatch):
    direct=fit(tmp_path/'direct',signals)
    forward=SegmentClassifier.forward;saved={}
    def corrupt(self,x,lengths):
        logits=forward(self,x,lengths)
        if self.training:logits.register_hook(lambda g:torch.full_like(g,float('nan')))
        return logits
    def inject(e):
        if e['type']=='epoch' and e['epoch']==1:
            saved['bytes']=(tmp_path/'retry/recovery.ckpt').read_bytes()
            monkeypatch.setattr(SegmentClassifier,'forward',corrupt)
    with pytest.raises(ValueError,match='非有限'):fit(tmp_path/'retry',signals,emit=inject)
    assert (tmp_path/'retry/recovery.ckpt').read_bytes()==saved['bytes']
    report=read_json(tmp_path/'retry/numerical_failure.json');assert report['epoch']==2 and report['recovery_epoch']==1
    monkeypatch.setattr(SegmentClassifier,'forward',forward)
    resumed=fit(tmp_path/'retry',signals)
    compare_state(direct[0].state_dict(),resumed[0].state_dict());compare_state(direct[2],resumed[2])
    # Failure evidence remains after a successful continuation.
    assert read_json(tmp_path/'retry/numerical_failure.json')==report


def test_invalid_scaler_still_produces_valid_json(tmp_path,signals,monkeypatch):
    monkeypatch.setattr(engine,'fit_scaler',lambda trials:dict(mean=[float('inf')],std=[float('nan')]))
    with pytest.raises(ValueError,match='非有限'):fit(tmp_path,signals)
    text=(tmp_path/'numerical_failure.json').read_text()
    assert 'Infinity' not in text and 'NaN' not in text
    assert json.loads(text)['stage']=='scaler'


def test_repeated_failure_retains_each_original_report(tmp_path,signals,monkeypatch):
    forward=SegmentClassifier.forward;fault={'mode':'logits'};events=[]
    def corrupt(self,x,lengths):
        logits=forward(self,x,lengths)
        if self.training:
            if fault['mode']=='logits':return logits*float('nan')
            logits.register_hook(lambda g:torch.full_like(g,float('inf')))
        return logits
    monkeypatch.setattr(SegmentClassifier,'forward',corrupt)
    with pytest.raises(ValueError,match='非有限'):fit(tmp_path,signals,emit=events.append)
    from pathlib import Path
    first=Path(next(e['diagnostic'] for e in events if e['type']=='numerical_failure'));original=first.read_bytes()
    fault['mode']='gradients'
    with pytest.raises(ValueError,match='非有限'):fit(tmp_path,signals,emit=events.append)
    failures=[e for e in events if e['type']=='numerical_failure']
    assert failures[0]['diagnostic']!=failures[1]['diagnostic']
    assert first.read_bytes()==original and read_json(first)['stage']=='logits'
    second=Path(failures[1]['diagnostic'])
    assert read_json(second)['stage']=='gradients'
    assert read_json(tmp_path/'numerical_failure.json')==read_json(second)
