import json
from dataclasses import replace
import numpy as np
import pytest
from cloth_trainer.modern_config import ModernConfig
from cloth_trainer.tcn_models import default_spec
from cloth_trainer.storage import read_json,write_json
from test_ui import app,wait,dispose_test_windows


def test_new_study_locks_actual_candidates_and_roundtrips():
    from cloth_trainer.tcn_study import candidates,study_definition,study_description,contrasts
    cfg=ModernConfig(architectures=('tcn',),tcn_mode='window_pool',tcn_spec=default_spec(128),epochs=400)
    cfg.validate()
    assert ModernConfig.from_dict(cfg.to_dict()).to_dict()==cfg.to_dict()
    cc=candidates(cfg)
    assert [c['window_size'] for c in cc]==[900,900,450,1800]
    assert [c['spec']['pooling'] for c in cc]==['mean','mean_std','mean','mean']
    assert all(c['spec']['channels']==[128]*6 for c in cc)
    assert len({c['id'] for c in cc})==4
    for override in [dict(tcn_window=450),dict(tcn_spec=default_spec(64))]:
        with pytest.raises(ValueError):replace(cfg,**override).validate()
    desc=study_description(study_definition(cfg))
    assert len(desc['groups'])==4 and desc['groups'][1]['candidate_id']==cc[1]['id']
    modelrows={str(i):dict(candidate=c,seed=42,fold=0,val_metrics={'macro_f1':i/10})for i,c in enumerate(cc)}
    diffs=contrasts(dict(study=study_definition(cfg),models=modelrows))
    assert set(diffs)=={'B-A','C-A','D-A'}
    assert diffs['B-A']['mean']==pytest.approx(.1)


def test_training_events_show_real_validation_and_refit_na(tmp_path):
    from test_tcn_training import trials,config
    from cloth_trainer.modern_training import fit
    from cloth_trainer.tcn_study import candidates
    tt=trials();cfg=config();events=[]
    result=fit(tt[:5],tt[5:],['A','B'],cfg,'tcn','mean',42,tmp_path/'cv',events.append,candidate=candidates(cfg)[0])
    epochs=[e for e in events if e['type']=='epoch']
    for e in epochs:
        assert f'{e["val_f1"]:.2%}' in e['message'] and '验证宏F1' in e['message']
        assert f'{e["best_f1"]:.2%}' in e['message'] and e['best_epoch']>=1
    assert result['stop_reason']=='max_epochs'
    assert events[-1]['type']=='fit_done' and '上限' in events[-1]['message']
    events=[]
    fit(tt[:5],[],['A','B'],cfg,'tcn','mean',42,tmp_path/'final',events.append,fixed_epochs=1,candidate=candidates(cfg)[0])
    epoch=next(e for e in events if e['type']=='epoch')
    assert epoch['val_f1'] is None and '验证F1不适用' in epoch['message']


def make_snapshot(root):
    from test_data import fixture_csv
    from cloth_trainer.data import load_record
    from cloth_trainer.config import DataConfig
    from cloth_trainer.dataset import save_dataset
    from cloth_trainer.modern_split import make_plan
    rr=[load_record(fixture_csv(root/'raw',c,i,speed=100),signals=('by',))for c in ['布A','印花丝绒','布B']for i in range(6)]
    p=root/'parent';save_dataset(p,rr,DataConfig(),signals=('by',))
    return p,rr,make_plan(rr)


def test_derivation_removes_class_and_preserves_all_partitions(tmp_path):
    from cloth_trainer.dataset import derive_dataset,load_dataset
    from cloth_trainer.data import file_hash
    parent,rr,plan=make_snapshot(tmp_path)
    original=file_hash(parent/'dataset.json');raw={r.path:file_hash(r.path)for r in rr}
    target=derive_dataset(parent,tmp_path/'child',['印花丝绒'],split_plan=plan)
    child,m=load_dataset(target);old=read_json(parent/'dataset.json')
    assert len(child)==12 and {r.label for r in child}=={'布A','布B'}
    assert m['dataset_id']!=old['dataset_id'] and m['selection']['parent_dataset_id']==old['dataset_id']
    assert len(list((target/'arrays').glob('*.npz')))==12
    kept=[i for i,r in enumerate(rr)if r.label!='印花丝绒'];mapping={old:new for new,old in enumerate(kept)}
    assert m['folds']==[[mapping[i]for i in f if i in mapping]for f in old['folds']]
    assert m['split_plan']['test']==[mapping[i]for i in plan['test']if i in mapping]
    assert m['split_plan']['cv']==[{k:[mapping[i]for i in f[k]if i in mapping]for k in ('train','val')}for f in plan['cv']]
    assert m['admission']==old['admission']
    assert file_hash(parent/'dataset.json')==original
    assert all(file_hash(p)==h for p,h in raw.items())
    for i,r in zip(kept,child):assert np.array_equal(r.trials[0].values,rr[i].trials[0].values)
    with pytest.raises(ValueError):derive_dataset(parent,tmp_path/'empty',['布A','印花丝绒','布B'],split_plan=plan)


def test_training_consumes_derived_frozen_plan(tmp_path):
    from cloth_trainer.dataset import derive_dataset,load_dataset
    from cloth_trainer.modern_engine import initialize
    from cloth_trainer.config import DataConfig
    parent,rr,plan=make_snapshot(tmp_path)
    child=derive_dataset(parent,tmp_path/'child',['印花丝绒'],split_plan=plan)
    _,manifest=load_dataset(child)
    cfg=ModernConfig(device='cpu',workers=0,architectures=('tcn',),tcn_spec=default_spec(32,2),data=DataConfig.from_dict(manifest['data']))
    task,m,_,_=initialize(dict(snapshot=str(child),output=str(tmp_path/'models'),config=cfg.to_dict()))
    assert m['split_plan']==manifest['split_plan']
    with pytest.raises(ValueError,match='种子|划分'):initialize(dict(snapshot=str(child),output=str(tmp_path/'other'),config=replace(cfg,split_seed=7).to_dict()))


def test_ui_modes_share_actual_explanations_and_preserve_training_settings(app,tmp_path):
    from test_tcn_helpers import TCNTrainerWindow
    w=TCNTrainerWindow(tmp_path/'models')
    w.config=replace(w.config,name='16类（不含印花丝绒）')
    w.sync_controls();assert w.task_name.text()=='16类（不含印花丝绒）'
    w.editor.fields['epochs'].setValue(400)
    mode=w.editor.fields['tcn_mode'];index=mode.findData('window_pool');assert index>=0
    mode.setCurrentIndex(index)
    assert w.controls_config().epochs==400 and w.controls_config().tcn_spec['channels']==[128]*6
    text=w.candidate_preview.toPlainText()
    assert all(s in text for s in ['窗口与汇总','450','1800','mean_std','波动','20'])
    mode.setCurrentIndex(mode.findData('four_way'));assert w.controls_config().epochs==400
    assert '通道与切窗' in w.candidate_preview.toPlainText()
    w.close()


def test_derived_dataset_ui_does_not_resurrect_removed_review_rows(app,tmp_path):
    from cloth_trainer.dataset import derive_dataset
    from test_tcn_helpers import TCNTrainerWindow
    parent,rr,plan=make_snapshot(tmp_path)
    child=derive_dataset(parent,tmp_path/'child',['印花丝绒'],split_plan=plan)
    w=TCNTrainerWindow(tmp_path/'models');w.load_dataset_version(child);wait(app,lambda:len(w.records)==12 and not w.busy())
    assert len(w.records)==12 and not any(r.label=='印花丝绒'for r in w.records)
    assert not any(e.get('label')=='印花丝绒'for e in w.errors)
    manifest=read_json(child/'dataset.json')
    assert w.selected_plan(w.records,w.config.split_seed)==manifest['split_plan']
    w.save_dataset_version(tmp_path/'saved-again');wait(app,lambda:(tmp_path/'saved-again/dataset.json').exists() and not w.busy())
    assert read_json(tmp_path/'saved-again/dataset.json')['split_plan']==manifest['split_plan']
    w.close()


def test_startup_current_project_selection_is_explicit_and_skips_batch_import(tmp_path):
    from train_cloth import startup_project
    active=tmp_path/'outputs/当前训练项目.json';active.parent.mkdir();active.write_text('{}')
    assert startup_project(None,True,tmp_path)==active
    assert startup_project(None,False,tmp_path) is None
    custom=tmp_path/'custom.json'
    assert startup_project(custom,False,tmp_path)==custom


def test_second_derivation_keeps_ancestor_exclusion_and_refuses_repartition(tmp_path):
    from cloth_trainer.dataset import derive_dataset,load_dataset
    from cloth_trainer.modern_split import make_plan
    parent,rr,plan=make_snapshot(tmp_path)
    child=derive_dataset(parent,tmp_path/'child',['印花丝绒'],split_plan=plan)
    records,m=load_dataset(child)
    alternative=make_plan(records)
    assert alternative!=m['split_plan']
    with pytest.raises(ValueError,match='冻结|划分'):derive_dataset(child,tmp_path/'bad',['布A'],split_plan=alternative)
    # A second derivation needs >=2 retained classes, so use a four-class parent.
    from test_data import fixture_csv
    from cloth_trainer.data import load_record
    from cloth_trainer.dataset import save_dataset
    from cloth_trainer.config import DataConfig
    rr=[load_record(fixture_csv(tmp_path/'fourraw',c,i,speed=100),signals=('by',))for c in ['布A','印花丝绒','布B','布C']for i in range(6)]
    p=tmp_path/'fourparent';save_dataset(p,rr,DataConfig(),signals=('by',))
    first=derive_dataset(p,tmp_path/'first',['印花丝绒'],split_plan=make_plan(rr))
    second=derive_dataset(first,tmp_path/'second',['布C'])
    _,m=load_dataset(second)
    assert set(m['selection']['excluded_labels'])=={'印花丝绒','布C'}


def test_invalid_draft_cannot_replace_busy_frozen_task_description(app,tmp_path):
    from PyQt5 import QtCore as C
    from test_tcn_helpers import TCNTrainerWindow
    w=TCNTrainerWindow(tmp_path/'models')
    w.candidate_preview.setPlainText('冻结运行中的对照说明')
    w.process.start('/bin/sleep',['2']);assert w.process.waitForStarted()
    try:
        w.editor.fields['channels'].setText('oops')
        assert w.candidate_preview.toPlainText()=='冻结运行中的对照说明'
    finally:
        w.process.kill();w.process.waitForFinished();w.close()
