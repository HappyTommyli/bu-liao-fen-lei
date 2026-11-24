import pytest
import torch
from cloth_trainer.tcn_study import window_slices,split_batch,make_candidate,validate_candidate,aggregate_log_probs
from cloth_trainer.tcn_models import default_spec


def test_overlap_stops_after_first_tail_and_preserves_coverage():
    assert window_slices(1800,900,450)==[(0,900),(450,1350),(900,1800)]
    assert window_slices(1900,900,450)[-1]==(1350,1900)
    assert window_slices(400,900,450)==[(0,400)]
    with pytest.raises(ValueError):window_slices(1800,900,901)


def test_parent_weights_and_probabilities_remain_normalized_with_overlap():
    x=torch.randn(2,1900,1)
    windows,lens,parents,weights,starts=split_batch(x,torch.tensor([1900,900]),900,450)
    assert list(starts[parents==0])==[0,450,900,1350]
    assert len(windows[parents==1])==1
    for parent in (0,1):assert weights[parents==parent].sum()==pytest.approx(1)
    logp=aggregate_log_probs(torch.randn(len(windows),3),parents,weights,2)
    assert torch.allclose(logp.exp().sum(1),torch.ones(2),atol=1e-6)


def test_candidate_revision_keeps_legacy_dicts_and_freezes_overlap_stride():
    old=make_candidate('old',default_spec(128),900)
    assert 'window_stride' not in old and validate_candidate(old)==old
    new=make_candidate('new',default_spec(128),900,450)
    assert new['window_stride']==450 and new['training_revision']!=old['training_revision']
    assert validate_candidate(new)==new
    with pytest.raises(ValueError):validate_candidate(dict(new,window_stride=901))


def test_overlap_model_really_trains_and_evaluates_with_normalized_scores(tmp_path):
    from test_data import fixture_csv
    from cloth_trainer.data import load_record,prepare_trials
    from cloth_trainer.modern_config import ModernConfig
    from cloth_trainer.modern_training import fit,evaluate
    from cloth_trainer.tcn_study import candidates
    rr=[load_record(fixture_csv(tmp_path,c,i,speed=100),signals=('by',)) for c in ('布A','布B') for i in range(2)]
    cfg=ModernConfig(device='cpu',workers=0,epochs=1,batch_size=8,micro_batch=4,architectures=('tcn',),signals=('by',),tcn_spec=default_spec(16,1),tcn_window=128,tcn_stride=64,cv_folds=1)
    train=prepare_trials(rr,'by');candidate=candidates(cfg)[0]
    result=fit(train,train,['布A','布B'],cfg,'tcn','mean',42,tmp_path/'fit',candidate=candidate)
    predictions,_=evaluate(result['model'],train,result['scaler'],['布A','布B'],cfg)
    assert predictions and result['history'][0]['optimizer_updates']==1
    for p,t in zip(predictions,train):
        assert len(p['windows'])==len(window_slices(len(t.values),128,64))
        assert sum(p['scores'])==pytest.approx(1,abs=1e-5)


def test_legacy_recovery_keeps_exact_frozen_config(tmp_path):
    import io,json
    from test_data import fixture_csv
    from cloth_trainer.data import load_record,prepare_trials
    from cloth_trainer.modern_config import ModernConfig
    from cloth_trainer.modern_training import fit
    cfg=ModernConfig(device='cpu',workers=0,epochs=1,batch_size=8,architectures=('gru',),hidden_size=32)
    old=json.loads(json.dumps(cfg.to_dict()));old.pop('cv_folds');old.pop('tcn_stride');old['data'].pop('sample_quality');old['data'].pop('contact_rule')
    # Real old recovery stores the original JSON, not newly supplied defaults.
    recovered=ModernConfig.from_dict(old)
    trials=prepare_trials([load_record(fixture_csv(tmp_path,c,0,speed=100),signals=('by',)) for c in ('布A','布B')],'by')
    folder=tmp_path/'fit'
    first=fit(trials,trials,['布A','布B'],recovered,'gru','mean',42,folder,frozen_config=old)
    raw=(folder/'recovery.ckpt').read_bytes().split(b'\n',1)[1]
    assert torch.load(io.BytesIO(raw),weights_only=True)['binding']['config']==old
    second=fit(trials,trials,['布A','布B'],recovered,'gru','mean',42,folder,frozen_config=old)
    assert second['history']==first['history']


def test_overlap_comparison_has_paired_windows_and_contrast():
    from dataclasses import replace
    from test_tcn_helpers import legacy_focused_config as focused_config
    from cloth_trainer.tcn_study import candidates,study_definition,study_description,contrasts
    cfg=replace(focused_config(),tcn_mode='overlap_pair',tcn_stride=450).validate()
    a,b=candidates(cfg)
    assert a['window_size']==b['window_size']==900
    assert a['spec']==b['spec'] and 'window_stride' not in a and b['window_stride']==450
    assert len(study_description(study_definition(cfg))['groups'])==2
    meta=dict(study=study_definition(cfg),models={str(i):dict(candidate=c,seed=42,fold=0,val_metrics=dict(macro_f1=v)) for i,(c,v) in enumerate(((a,.7),(b,.8)))})
    assert contrasts(meta)['B-A']['mean']==pytest.approx(.1)
    with pytest.raises(ValueError):replace(cfg,tcn_stride=900).validate()
    with pytest.raises(ValueError):replace(cfg,tcn_stride=0).validate()
    with pytest.raises(ValueError):replace(cfg,tcn_window=0,tcn_stride=0).validate()


def test_overlap_comparison_really_trains_two_models_once(tmp_path):
    from dataclasses import replace
    from test_data import fixture_csv
    from cloth_trainer.data import load_record
    from test_tcn_helpers import legacy_focused_config as focused_config
    from cloth_trainer.engine import train_request
    from cloth_trainer.storage import read_json
    cfg=replace(focused_config(),device='cpu',workers=0,epochs=1,batch_size=8,tcn_mode='overlap_pair',tcn_window=128,tcn_stride=64,tcn_spec=default_spec(16,1))
    rr=[load_record(fixture_csv(tmp_path/'raw',c,i,speed=100),signals=('by',)) for c in ('布A','布B') for i in range(3)]
    task=train_request(dict(paths=[r.path for r in rr],sources=[r.manifest() for r in rr],config=cfg.to_dict(),output=str(tmp_path/'models')))
    meta=read_json(task/'metadata.json');rows=list(meta['models'].values())
    assert len(rows)==2 and len(meta['summaries'])==2
    assert {r['candidate'].get('window_stride',128) for r in rows}=={64,128}
    assert len(meta['contrasts']['B-A']['folds'])==1
    assert all(len(r['history'])==1 for r in rows)
