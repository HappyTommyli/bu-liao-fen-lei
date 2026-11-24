import shutil
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from cloth_trainer.config import TrainConfig
from cloth_trainer.data import load_record,prepare_trials
from cloth_trainer.storage import read_json,export_task,import_task
from cloth_trainer.engine import train_request,test_request as heldout_test,predict_request,load_bundle,Cancelled
from test_data import fixture_csv

@pytest.fixture(scope='module')
def source(tmp_path_factory):
    root=tmp_path_factory.mktemp('v3_source')
    return [load_record(fixture_csv(root,c,i,speed=100),signals=('by','bxby')) for c in ('布A','布B') for i in range(5)]


def request(source,out,**kwargs):
    cfg=TrainConfig(epochs=2,architectures=('tcn',),batch_size=8,**kwargs)
    return dict(paths=[r.path for r in source],sources=[r.manifest() for r in source],overrides={r.path:r.label for r in source},config=cfg.to_dict(),output=str(out))


def test_fivefold_full_budget_train_scalers_and_once_oof(source,tmp_path):
    events=[];task=train_request(request(source,tmp_path/'out',patience=1),events.append);m=read_json(task/'metadata.json')
    assert m['format']=='cloth-classification-v3' and len(m['models'])==5
    assert all(len(r['history'])==2 and r['test_metrics'] is None for r in m['models'].values())
    for key,r in m['models'].items():
        b=load_bundle(task/key/'model.pt');idx=m['partitions'][r['fold']]['train']
        expected=np.concatenate([t.values for t in prepare_trials([source[i] for i in idx],'by')]).mean(0)
        assert np.allclose(b['scaler']['mean'],expected,rtol=1e-5)
        assert b['dataset_id']==m['dataset_id'] and b['fold']==r['fold']
    heldout_test({'task':str(task)})
    m=read_json(task/'metadata.json');summary=m['summaries']['tcn_by_mean']['by_seed']['42']
    assert summary['oof_metrics']['total']==20
    assert len(summary['fold_metrics'])==5
    rows=[x for r in m['models'].values() for x in r['test_predictions']]
    assert len({r['id'] for r in rows})==20
    assert set(summary['quality_metrics'])=={'normal','suggest_review','suggest_recollect'}
    assert all(e.get('fold') is not None for e in events if e['type']=='epoch')


def test_cancel_resume_uses_immutable_snapshot_and_finished_weights(source,tmp_path):
    trigger={'stop':False}
    def emit(e):
        if e['type']=='model_done':trigger['stop']=True
    with pytest.raises(Cancelled):train_request(request(source,tmp_path/'out'),emit,lambda:trigger['stop'])
    task=next((tmp_path/'out').glob('*'));before=read_json(task/'metadata.json');assert before['status']=='cancelled' and len(before['models'])==1
    original=next(iter(before['models'].values()))['weight_sha256']
    events=[];train_request({'resume':str(task)},events.append)
    after=read_json(task/'metadata.json');assert after['status']=='completed' and len(after['models'])==5
    assert next(iter(after['models'].values()))['weight_sha256']==original
    assert len([e for e in events if e['type']=='phase'])==4
    package=tmp_path/'package.zip';export_task(task,package);moved=import_task(package,tmp_path/'import')
    key=next(iter(after['models']));b=load_bundle(moved/key/'model.pt');assert b['model_id']==after['models'][key]['model_id']
    # Testing uses the saved processed snapshot, not a fresh scan of raw folders.
    heldout_test({'task':str(moved)})
    dest=tmp_path/'prediction.json';predict_request(dict(models=[str(moved/key/'model.pt')],paths=[source[0].path],destination=str(dest)))
    result=read_json(dest)['results'][0]
    assert len(result['predictions'])==2 and result['evaluation']['replay']['total']==2


def test_corrupt_snapshot_blocks_resume_and_pre_registered_configs(source,tmp_path):
    state={'stop':False}
    with pytest.raises(Cancelled):
        train_request(request(source,tmp_path/'out'),lambda e:state.update(stop=True) if e['type']=='model_done' else None,lambda:state['stop'])
    task=next((tmp_path/'out').glob('*'));a=next((task/'dataset/arrays').glob('*.npz'));a.write_bytes(a.read_bytes()+b'altered')
    with pytest.raises(ValueError,match='校验'):train_request({'resume':str(task)})


def test_bundle_fold_identity_and_unconfirmed_truth_are_guarded(source,tmp_path):
    import json
    from cloth_trainer.storage import write_json
    task=train_request(request(source,tmp_path/'out'));m=read_json(task/'metadata.json');key=next(iter(m['models']))
    m['models'][key]['fold']=1;write_json(task/'metadata.json',m)
    with pytest.raises(ValueError,match='折|关联'):load_bundle(task/key/'model.pt')
    m['models'][key]['fold']=0;write_json(task/'metadata.json',m)
    original=Path(source[0].path);folder=tmp_path/'unlabeled';folder.mkdir();target=folder/'布A_999.csv';shutil.copy(original,target)
    md=read_json(original.parent/'metadata.json');md.pop('label');(folder/'metadata.json').write_text(json.dumps(md))
    dest=tmp_path/'prediction.json';predict_request(dict(models=[str(task/key/'model.pt')],paths=[str(target)],destination=str(dest)))
    result=read_json(dest)['results'][0]
    assert all(r['true']==-1 for r in result['predictions']) and result['evaluation']['unknown_labels']==2


@pytest.mark.parametrize('signal',['bxby','xy'])
def test_by_snapshot_rejects_missing_axes_before_task_creation(source,tmp_path,signal):
    from cloth_trainer.dataset import save_dataset
    records=[load_record(r.path,signals=('by',)) for r in source]
    cfg=TrainConfig(epochs=1,architectures=('tcn',),signals=(signal,))
    snap=save_dataset(tmp_path/'snap',records,cfg.data)
    with pytest.raises(ValueError,match='通道'):train_request(dict(snapshot=str(snap),config=cfg.to_dict(),output=str(tmp_path/'out')))
    assert not list((tmp_path/'out').glob('CLOTH5_*'))


def test_frozen_screening_report_preserves_original_exclusion_audit(source,tmp_path):
    req=request(source,tmp_path/'out');req['screening']=dict(candidate_count=12,ignored_count=1,ignored=[dict(path='trash.csv',reason='回收站')],duplicates=[dict(path='copy.csv')],excluded=[dict(path='bad.csv',reason='用户明确不纳入')])
    task=train_request(req);meta=read_json(task/'metadata.json');screen=read_json(task/'导入筛查.json')
    assert meta['screening']['excluded']==req['screening']['excluded']
    assert screen['ignored']==req['screening']['ignored'] and screen['duplicates']==req['screening']['duplicates']
    assert r'\|ΔBy\|' in (task/'报告.md').read_text()


def test_narrowing_frozen_snapshot_requires_new_processing_version(source,tmp_path):
    from cloth_trainer.dataset import save_dataset
    cfg=TrainConfig(epochs=1,architectures=('tcn',))
    snap=save_dataset(tmp_path/'joint',source,cfg.data)
    with pytest.raises(ValueError,match='通道|有效点'):train_request(dict(snapshot=str(snap),config=cfg.to_dict(),output=str(tmp_path/'out')))


def test_joint_training_by_scalers_and_inference_do_not_depend_on_bx(tmp_path):
    from test_r3_data import rewrite
    records=[load_record(fixture_csv(tmp_path/'raw',c,i,speed=100),signals=('by','bxby')) for c in ('布A','布B') for i in range(5)]
    p=Path(records[0].path);rewrite(p,modify=lambda row:row.update(Bx_uT='nan',dBx_uT='nan') if row['stage']=='step_003' and .8<float(row['sensor_t_s'])<.804 else None);records[0]=load_record(p,signals=('by','bxby'))
    task=train_request(request(records,tmp_path/'out',signals=('by','bxby')));m=read_json(task/'metadata.json')
    for key,r in m['models'].items():
        if r['signal']!='by':continue
        b=load_bundle(task/key/'model.pt');fresh=[load_record(m['sources'][i]['path'],signals=('by',)) for i in m['partitions'][r['fold']]['train']]
        expected=np.concatenate([t.values for t in prepare_trials(fresh,'by')]).astype(np.float64).mean(0)
        assert np.allclose(b['scaler']['mean'],expected,rtol=0,atol=1e-10)
    key=next(k for k,r in m['models'].items() if r['signal']=='by');folder=tmp_path/'by_input';folder.mkdir();q=folder/'input.csv';shutil.copy(p,q);shutil.copy(p.parent/'metadata.json',folder/'metadata.json');rewrite(q,drop=('Bx_uT','dBx_uT'))
    dest=tmp_path/'p.json';predict_request(dict(models=[str(task/key/'model.pt')],paths=[str(p)],destination=str(dest)));before=read_json(dest)['results'][0]['predictions']
    predict_request(dict(models=[str(task/key/'model.pt')],paths=[str(q)],destination=str(dest)));after=read_json(dest)['results'][0]['predictions']
    assert len(after)==2 and all(np.allclose(a['scores'],b['scores'],atol=1e-7) for a,b in zip(before,after))


def test_worker_excludes_bad_capture_but_freezes_complete_review(tmp_path):
    from test_r3_data import rewrite
    paths=[fixture_csv(tmp_path/'raw',c,i,speed=100) for c in ('布A','布B') for i in range(6 if c=='布A' else 5)]
    rewrite(paths[0],modify=lambda row:row.update(dBy_uT=str(float(row['dBy_uT'])*2),By_uT=str(float(row['By_uT'])*2)) if row['stage']=='step_004' else None)
    cfg=TrainConfig(epochs=1,architectures=('tcn',))
    task=train_request(dict(paths=[str(p) for p in paths],config=cfg.to_dict(),output=str(tmp_path/'out')))
    m=read_json(task/'metadata.json');snap=read_json(task/'dataset/dataset.json')
    assert m['records']==10 and len(snap['admission']['batches'][0]['entries'])==11
    assert all(s['path']!=str(paths[0]) for s in m['sources'])
    assert snap['admission']['batches'][0]['entries'][0]['audit']['eligible'] is False


def test_worker_rejects_old_snapshot_and_changed_full_batch(tmp_path):
    from cloth_trainer.data import import_dataset,file_hash
    from cloth_trainer.dataset import save_dataset
    from cloth_trainer.storage import write_json
    cfg=TrainConfig(epochs=1,architectures=('tcn',))
    paths=[fixture_csv(tmp_path/'raw',c,i,speed=100) for c in ('布A','布B') for i in range(5)]
    records,report=import_dataset(paths,cfg.data,cfg.signals)
    snap=save_dataset(tmp_path/'snap',records,cfg.data,screening=report)
    m=read_json(snap/'dataset.json');m.pop('admission');write_json(snap/'dataset.json',m);(snap/'dataset.sha256').write_text(file_hash(snap/'dataset.json'))
    with pytest.raises(ValueError,match='旧|审查'):train_request(dict(snapshot=str(snap),config=cfg.to_dict(),output=str(tmp_path/'out')))
    paths[0].write_text(paths[0].read_text(encoding='utf-8-sig')+'\n',encoding='utf-8-sig')
    with pytest.raises(ValueError,match='批次|改变'):train_request(dict(paths=[str(p) for p in paths[1:]],review_batches=report['review_batches'],config=cfg.to_dict(),output=str(tmp_path/'out')))
    assert not list((tmp_path/'out').glob('*'))


def test_resume_inside_fold_preserves_state_and_reports_saved_epoch(source,tmp_path):
    trigger={'stop':False}
    def emit(e):
        if e['type']=='epoch' and e['epoch']==1:trigger['stop']=True
    with pytest.raises(Cancelled):train_request(request(source,tmp_path/'out'),emit,lambda:trigger['stop'])
    task=next((tmp_path/'out').glob('*'));before=read_json(task/'metadata.json');assert before['models']=={}
    key='tcn_by_mean_s42_f1';checkpoint=(task/key/'recovery.ckpt').read_bytes()
    assert checkpoint and before['status']=='cancelled'
    events=[];train_request({'resume':str(task)},events.append)
    restored=next(e for e in events if e['type']=='restored')
    assert restored['model']==key and restored['fold']==0 and restored['epoch']==1
    assert [e['epoch'] for e in events if e['type']=='epoch' and e['model']==key]==[2]
    meta=read_json(task/'metadata.json');assert len(meta['models'])==5
    assert meta['training_runtime']=='numerical-safety-v2'


def test_final_train_evaluation_failure_has_diagnostic_and_keeps_recovery(source,tmp_path,monkeypatch):
    import torch
    from cloth_trainer import engine
    evaluate=engine.evaluate;calls=[]
    def fail_final(model,*args,**kwargs):
        calls.append(True)
        if len(calls)==4:
            forward=model.forward
            model.forward=lambda *a,**k:forward(*a,**k)*float('nan')
        return evaluate(model,*args,**kwargs)
    monkeypatch.setattr(engine,'evaluate',fail_final)
    with pytest.raises(ValueError,match='非有限'):train_request(request(source,tmp_path/'out'))
    task=next((tmp_path/'out').glob('*'));folder=task/'tcn_by_mean_s42_f1'
    report=read_json(folder/'numerical_failure.json')
    assert report['stage']=='validation_logits' and report['phase']=='final_train_evaluation'
    assert report['recovery_epoch']==2 and report['job']['key']=='tcn_by_mean_s42_f1'
    assert (folder/'recovery.ckpt').exists() and not (folder/'model.pt').exists()
    assert read_json(task/'metadata.json')['models']=={}


def test_old_runtime_continues_in_one_fold_validation_copy(source,tmp_path,monkeypatch):
    import hashlib,io,torch
    from cloth_trainer import engine
    from cloth_trainer.recovery import save_recovery
    from cloth_trainer.storage import write_json
    trigger={'stop':False}
    with pytest.raises(Cancelled):
        train_request(request(source,tmp_path/'out'),lambda e:trigger.update(stop=True) if e['type']=='epoch' else None,lambda:trigger['stop'])
    parent=next((tmp_path/'out').glob('*'));key='tcn_by_mean_s42_f1';p=parent/key/'recovery.ckpt'
    envelope=torch.load(io.BytesIO(p.read_bytes().split(b'\n',1)[1]),weights_only=True)
    envelope['binding']['runtime']='numerical-safety-v1'
    for row in envelope['state']['history']:
        for k in list(row):
            if k.startswith('gradient_'):row.pop(k)
    save_recovery(p.parent,envelope['state'],envelope['binding'])
    meta=read_json(parent/'metadata.json');meta['training_runtime']='numerical-safety-v1';write_json(parent/'metadata.json',meta)
    fingerprint=lambda root:{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*') if f.is_file()}
    before=fingerprint(parent);events=[];derived=train_request({'resume':str(parent)},events.append)
    assert derived!=parent and fingerprint(parent)==before
    m=read_json(derived/'metadata.json')
    assert m['status']=='completed' and m['purpose']=='quick' and m['validation_kind']=='runtime_upgrade'
    assert m['parent_task']==str(parent.resolve()) and m['validation_target_key']==key and len(m['models'])==1
    assert m['training_runtime']=='numerical-safety-v2' and m['runtime_upgrade']['from']=='numerical-safety-v1'
    assert len([e for e in events if e['type']=='phase'])==1
    assert 'gradient_norm_max' not in m['models'][key]['history'][0] and 'gradient_norm_max' in m['models'][key]['history'][1]
    assert [e['epoch'] for e in events if e['type']=='epoch']==[2]
    assert any(e['type']=='runtime_upgraded' for e in events)
    assert (derived/key/'recovery-v1-original.ckpt').read_bytes()==p.read_bytes()
    with pytest.raises(ValueError,match='快速|验证|正式'):load_bundle(derived/key/'model.pt')
    with pytest.raises(ValueError,match='快速|验证|正式'):heldout_test({'task':str(derived)})


def test_interrupted_validation_copy_is_never_published_as_original(tmp_path,monkeypatch):
    from cloth_trainer import runtime_validation
    from cloth_trainer.storage import write_json,list_tasks
    source=tmp_path/'original';meta=dict(id='original',name='原失败实验',status='failed',purpose='formal',training_runtime='numerical-safety-v1',models={})
    write_json(source/'metadata.json',meta)
    original=runtime_validation.shutil.copytree
    def interrupted(*args,**kwargs):
        original(*args,**kwargs)
        raise KeyboardInterrupt('simulate forced termination after source metadata copy')
    monkeypatch.setattr(runtime_validation.shutil,'copytree',interrupted)
    with pytest.raises(KeyboardInterrupt):runtime_validation.fork_runtime_validation(source,meta,'gru_by_mean_s42_f1','numerical-safety-v2')
    visible=list_tasks(tmp_path)
    assert len(visible)==1 and visible[0]['directory']==str(source.resolve())
    assert read_json(source/'metadata.json')==meta
