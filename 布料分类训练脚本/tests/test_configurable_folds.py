"""Configurable grouped development splits and safe online model loading."""
import copy
from types import SimpleNamespace
import pytest
from cloth_trainer.modern_config import ModernConfig
from cloth_trainer.modern_split import make_plan, validate_plan


def records(n=20, shared=False):
    return [SimpleNamespace(group=f'{"session" if shared else c}{i}',label=c) for c in 'AB' for i in range(n)]


@pytest.mark.parametrize('count', [1,2,3,5,10])
@pytest.mark.parametrize('shared', [False,True])
def test_grouped_counts_keep_fixed_holdout_and_disjoint_development(count,shared):
    rr=records(shared=shared);plan=make_plan(rr,42,cv_folds=count)
    assert plan['test']==make_plan(rr,42)['test']
    assert len(plan['cv'])==count
    assert len(plan['dev'])==32 and len(plan['test'])==8
    validate_plan(rr,plan)
    for fold in plan['cv']:
        assert not {rr[i].group for i in fold['train']}&{rr[i].group for i in fold['val']}
        assert not set(fold['train']+fold['val'])&set(plan['test'])
    if count==1:assert len(plan['cv'][0]['train'])==24 and len(plan['cv'][0]['val'])==8
    else:assert sorted(i for f in plan['cv'] for i in f['val'])==plan['dev']


def test_single_split_supports_three_independent_groups_per_class():
    plan=make_plan(records(3),cv_folds=1)
    assert len(plan['test'])==2 and len(plan['cv'][0]['train'])==2 and len(plan['cv'][0]['val'])==2


@pytest.mark.parametrize('count',[0,11,-1,True,2.5])
def test_invalid_cv_counts_rejected(count):
    with pytest.raises(ValueError):ModernConfig(cv_folds=count).validate()
    with pytest.raises(ValueError):make_plan(records(),cv_folds=count)


def test_legacy_missing_fold_setting_stays_five_and_jobs_follow_configuration():
    from cloth_trainer.engine import _jobs
    old=ModernConfig(architectures=('gru',)).to_dict();old.pop('cv_folds',None)
    assert len(_jobs(ModernConfig.from_dict(old)))==5
    assert len(_jobs(ModernConfig(cv_folds=1,architectures=('gru',))))==1
    assert len(_jobs(ModernConfig(cv_folds=3,architectures=('gru',))))==3


@pytest.mark.parametrize('stride',[0,1,900])
def test_stride_validated_and_serialized(stride):
    cfg=ModernConfig(tcn_stride=stride)
    assert ModernConfig.from_dict(cfg.to_dict()).tcn_stride==stride


@pytest.mark.parametrize('stride',[-1,901,True,1.5])
def test_invalid_stride_rejected(stride):
    with pytest.raises(ValueError):ModernConfig(tcn_stride=stride).validate()


def test_plan_validation_rejects_duplicate_single_indices_and_declared_count_mismatch():
    rr=records();p=make_plan(rr,cv_folds=1)
    bad=copy.deepcopy(p);bad['cv'][0]['train'][0]=bad['cv'][0]['train'][1]
    with pytest.raises(ValueError):validate_plan(rr,bad)
    bad=copy.deepcopy(p);bad['cv_folds']=2
    with pytest.raises(ValueError):validate_plan(rr,bad)


@pytest.mark.parametrize('count,groups',[(1,3),(2,3)])
def test_training_uses_requested_folds_and_online_loader_preserves_formal_guard(tmp_path,count,groups):
    from test_data import fixture_csv
    from cloth_trainer.data import load_record,file_hash
    from cloth_trainer.engine import train_request,load_bundle,test_request as formal_test
    from cloth_trainer import modern_engine
    from cloth_trainer.storage import read_json,write_json
    rr=[load_record(fixture_csv(tmp_path/'raw',c,i,speed=100)) for c in ('布A','布B') for i in range(groups)]
    cfg=ModernConfig(device='cpu',workers=0,epochs=1,hidden_size=32,architectures=('gru',),batch_size=8,cv_folds=count)
    task=train_request(dict(paths=[r.path for r in rr],sources=[r.manifest() for r in rr],overrides={r.path:r.label for r in rr},config=cfg.to_dict(),output=str(tmp_path/'models')))
    meta=read_json(task/'metadata.json');assert len(meta['models'])==count and not meta['final_model']
    assert len(read_json(task/'dataset/dataset.json')['folds'])==2
    key=next(iter(meta['models']));weight=task/key/'model.pt'
    bundle=modern_engine.load_online_bundle(weight)
    assert bundle['_model'].training is False and bundle['phase']=='cv'
    assert bundle['model_source']['weight_sha256']==file_hash(weight)
    assert bundle['split_audit']['val']==meta['split_plan']['cv'][0]['val']
    with pytest.raises(ValueError,match='最终'):load_bundle(weight)
    with pytest.raises(ValueError,match='最终'):formal_test(dict(task=str(task)))
    import torch
    forged=torch.load(weight,weights_only=True);forged['scaler']['std']=[-1.]*len(forged['scaler']['mean']);torch.save(forged,weight)
    meta['models'][key]['weight_sha256']=file_hash(weight);write_json(task/'metadata.json',meta)
    with pytest.raises(ValueError,match='标准化|scaler'):modern_engine.load_online_bundle(weight)
