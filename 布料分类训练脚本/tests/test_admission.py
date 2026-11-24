from dataclasses import replace
import json
import numpy as np
import pytest
from cloth_trainer.data import Record,Trial,load_record,import_dataset
from cloth_trainer.quality import annotate
from test_data import fixture_csv


def record(out=100.,back=120.,sigma=1.,label='A',group='one',sha='one'):
    trials=[]
    for direction,lo,hi,value in [('out',45,142,out),('back',16,114,back)]:
        x=np.arange(lo,hi+.01,.2);y=np.full(len(x),value,dtype=float)
        t=np.arange(len(x))/333
        trials.append(Trial(sha+direction,sha+'.csv',sha,group,label,direction,y[:,None].copy(),t,t,x,y[:,None],(lo,hi),np.ones(len(x),bool),('y',)))
    return Record(sha+'.csv',sha,'meta',group,label,trials,[],250,6,axes=('y',),diagnostics=dict(baseline_sigma=[0,sigma,0]))


def test_all_four_conditions_exclude_both_legs_without_mutating_values():
    r=record();before=[t.raw_values.copy() for t in r.trials];annotate(r)
    assert r.audit['status']=='excluded' and not r.audit['eligible']
    d=r.diagnostics['direction_quality']
    assert d['full_rate']==pytest.approx(2/11) and d['common_rate']==pytest.approx(2/11)
    assert d['conditions']==dict(rates=True,same_direction=True,amplitude=True,persistent=True)
    assert all(np.array_equal(a,t.raw_values) for a,t in zip(before,r.trials))


@pytest.mark.parametrize('missing',['rates_full','rates_common','same_direction','amplitude','persistent'])
def test_each_missing_condition_cannot_exclude(missing):
    r=record()
    if missing in ('rates_full','same_direction'):
        t=r.trials[0];x=np.r_[t.raw_x,np.full(2000,130.)];y=np.r_[t.raw_values[:,0],np.full(2000,120. if missing=='rates_full' else 140.)]
        t.raw_x=x;t.raw_values=y[:,None];t.raw_valid=np.ones(len(x),bool)
    elif missing=='rates_common':
        t=r.trials[1];t.raw_values[:]=105.;t.raw_x=np.r_[t.raw_x,np.full(2000,30.)];t.raw_values=np.r_[t.raw_values[:,0],np.full(2000,180.)][:,None];t.raw_valid=np.ones(len(t.raw_x),bool)
    elif missing=='amplitude':
        r=record(out=10,back=11.5)
    else:
        t=r.trials[1];t.raw_values[(t.raw_x>=46)&(t.raw_x<68)]=80.
    annotate(r)
    assert r.audit['status']=='suggest_review' and r.audit['eligible']
    assert not r.diagnostics['direction_quality']['marked']


@pytest.mark.parametrize('missing',['noise','bins','zero'])
def test_missing_evidence_never_passes(missing):
    r=record()
    if missing=='noise':r.diagnostics['baseline_sigma']=None
    elif missing=='bins':
        r.trials[1].raw_x[:]=17.
    else:
        for t in r.trials:t.raw_values[:]=0
    annotate(r)
    assert r.audit['status']=='undetermined' and not r.audit['eligible']
    assert r.diagnostics['direction_quality']['status']=='not_assessable'


def test_common_center_uses_samples_not_equal_weight_bin_medians():
    r=record();x=np.concatenate([np.linspace(46+2*i,47.8+2*i,3 if i<6 else 30) for i in range(10)])
    for i,t in enumerate(r.trials):
        y=np.full(len(x),100.) if i==0 else np.where(x<58,110.,130.)
        t.raw_x=x.copy();t.raw_values=y[:,None];t.raw_valid=np.ones(len(x),bool)
    annotate(r);d=r.diagnostics['direction_quality']
    assert d['common_medians']==[100.,130.]
    assert d['common_difference']==30 and d['bins']==10


def test_historical_seven_and_eleven_require_both_fingerprints():
    from cloth_trainer.quality import ADMISSION_ANNOTATIONS
    excluded=[a for a in ADMISSION_ANNOTATIONS if a['decision']=='excluded']
    review=[a for a in ADMISSION_ANNOTATIONS if a['decision']=='suggest_review']
    assert len(excluded)==7 and len(review)==11
    for a in excluded+review:
        r=record(back=100,sha=a['sha256']);r.metadata_sha256=a['metadata_sha256'];annotate(r)
        assert r.audit['status']==a['decision']
        r.metadata_sha256='new metadata';annotate(r)
        assert r.audit['status']=='normal' and r.audit['eligible']
        r.sha256='new csv';r.metadata_sha256=a['metadata_sha256'];annotate(r)
        assert r.audit['status']=='normal'


def test_integrity_report_records_limits_and_observations(tmp_path):
    r=load_record(fixture_csv(tmp_path,speed=100),signals=('by',))
    checks=r.diagnostics['integrity']
    assert checks['limits']['max_gap_s']==.020 and checks['limits']['baseline_tolerance_uT']==.01
    assert checks['stages']['step_003']['max_gap_s']<.004
    assert checks['stages']['step_004']['observed_direction']=='back'
    assert checks['sources_unchanged'] and '撞击' in checks['unverifiable'][0]


def test_missing_baseline_is_visible_undetermined_not_usable(tmp_path):
    p=fixture_csv(tmp_path,speed=100);mp=p.parent/'metadata.json';m=json.loads(mp.read_text());m.pop('baseline_B0_uT');mp.write_text(json.dumps(m))
    records,report=import_dataset([p],signals=('by',))
    assert not records and report['errors'][0]['audit']['status']=='undetermined'
    assert report['errors'][0]['sha256'] and report['errors'][0]['metadata_sha256']


def test_class_gate_counts_independent_groups_and_requires_ten():
    from cloth_trainer.admission import review_batch
    from cloth_trainer.config import DataConfig
    rs=[annotate(record(back=120 if i<8 else 100,sha=str(i),group=str(i))) for i in range(10)]
    report=review_batch(rs,[],[r.path for r in rs],DataConfig(),('by',))
    c=report['classes']['A']
    assert c['abnormal_groups']==8 and c['assessable_groups']==10 and c['disabled']
    assert all(not r.audit['eligible'] for r in rs)
    nine=[annotate(record(sha=str(i),group=str(i))) for i in range(9)]
    report=review_batch(nine,[],[r.path for r in nine],DataConfig(),('by',))
    assert report['classes']['A']['rate']==1 and not report['classes']['A']['disabled']
    duplicate=[annotate(record(sha=str(i),group='same')) for i in range(20)]
    report=review_batch(duplicate,[],[r.path for r in duplicate],DataConfig(),('by',))
    assert report['classes']['A']['assessable_groups']==1 and not report['classes']['A']['disabled']


def test_class_gate_omits_unassessable_group_and_preserves_full_batch():
    from cloth_trainer.admission import review_batch,require_admission
    from cloth_trainer.config import DataConfig
    rs=[annotate(record(back=100,sha=str(i),group=str(i))) for i in range(10)]
    rs[0].diagnostics['baseline_sigma']=None;annotate(rs[0])
    report=review_batch(rs,[],[r.path for r in rs],DataConfig(),('by',))
    c=report['classes']['A'];assert c['assessable_groups']==9 and c['undetermined_groups']==1
    admission=dict(rule=report['rule'],batches=[report])
    require_admission(rs[1:3],admission,DataConfig())
    assert report['classes']['A']['assessable_groups']==9
    with pytest.raises(ValueError,match='准入|审查'):require_admission(rs[:1],admission,DataConfig())
    report['classes']['A']['assessable_groups']=2
    with pytest.raises(ValueError,match='校验'):require_admission(rs[1:3],admission,DataConfig())


def test_new_snapshot_rejects_exclusion_old_snapshot_stays_readonly(tmp_path):
    from cloth_trainer.dataset import save_dataset,load_dataset
    from cloth_trainer.admission import require_admission
    from cloth_trainer.config import DataConfig
    from cloth_trainer.quality import digest
    from cloth_trainer.data import file_hash
    records=[load_record(fixture_csv(tmp_path/'raw',c,i,speed=100),signals=('by',)) for c in ('布A','布B') for i in range(5)]
    snap=save_dataset(tmp_path/'snapshot',records,DataConfig())
    restored,m=load_dataset(snap);require_admission(restored,m['admission'],DataConfig())
    p=snap/'dataset.json';m.pop('admission');p.write_text(json.dumps(m,ensure_ascii=False));(snap/'dataset.sha256').write_text(file_hash(p))
    restored,m=load_dataset(snap)
    with pytest.raises(ValueError,match='旧|审查'):require_admission(restored,m.get('admission'),DataConfig())


def test_import_report_keeps_excluded_records_with_fixed_batch(tmp_path):
    from test_r3_data import rewrite
    p=fixture_csv(tmp_path,speed=100)
    rewrite(p,modify=lambda row:row.update(dBy_uT=str(float(row['dBy_uT'])*2),By_uT=str(float(row['By_uT'])*2)) if row['stage']=='step_004' else None)
    records,report=import_dataset([p],signals=('by',))
    assert len(records)==1 and not report['errors'] and report['eligible_count']==0
    assert records[0].audit['batch_id']==report['review_batches'][0]['batch_id']


def test_nonfinite_noise_stays_visible_and_report_is_standard_json(tmp_path):
    p=fixture_csv(tmp_path,speed=100);mp=p.parent/'metadata.json';m=json.loads(mp.read_text());m['context']['baseline']['sigma'][1]=float('nan');mp.write_text(json.dumps(m))
    records,report=import_dataset([p],signals=('by',))
    assert len(records)==1 and records[0].quality=='undetermined'
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize('label',[123,['A'],{'name':'A'}])
def test_malformed_error_label_does_not_abort_other_imports(tmp_path,label):
    good=fixture_csv(tmp_path/'raw','布A',0,speed=100);bad=fixture_csv(tmp_path/'raw','布B',0,speed=100)
    mp=bad.parent/'metadata.json';m=json.loads(mp.read_text());m['label']=label;mp.write_text(json.dumps(m))
    records,report=import_dataset([good,bad],signals=('by',))
    assert len(records)==1 and len(report['errors'])==1
    assert report['errors'][0]['label']=='' and 'invalid_metadata_label' in report['errors'][0]['diagnostics']


@pytest.mark.parametrize('partial',[False,True])
def test_unverifiable_subtraction_is_not_normal_and_preserves_input(tmp_path,partial):
    from test_r3_data import rewrite
    p=fixture_csv(tmp_path,speed=100)
    before=load_record(p,signals=('by',)).trials[0].values.copy()
    rewrite(p,modify=lambda row:row.update(By_uT='nan') if not partial or row['stage']=='step_003' and .8<float(row['sensor_t_s'])<.804 else None)
    records,report=import_dataset([p],signals=('by',));r=records[0]
    assert r.quality=='undetermined' and not r.audit['eligible']
    assert np.array_equal(r.trials[0].values,before)
    assert r.diagnostics['by_verification']['unverifiable_samples']>0
    assert report['review_batches'][0]['classes']['布A']['assessable_groups']==0


def test_error_label_conflict_reverification_uses_original_overrides(tmp_path):
    from cloth_trainer.admission import verify_raw_batches
    from cloth_trainer.config import DataConfig
    good=fixture_csv(tmp_path/'raw','布A',0,speed=100);bad=fixture_csv(tmp_path/'raw','布B',0,speed=100)
    mp=bad.parent/'metadata.json';m=json.loads(mp.read_text());m['label']='冲突类';mp.write_text(json.dumps(m))
    records,report=import_dataset([good,bad],signals=('by',))
    assert len(records)==1 and len(report['errors'])==1
    fresh=verify_raw_batches(report['review_batches'],DataConfig())
    assert len(fresh)==1 and fresh[0].path==records[0].path


@pytest.mark.parametrize('keys',[('context',),('context','config'),('context','baseline'),('baseline_B0_uT',)])
def test_malformed_metadata_mapping_is_isolated_per_capture(tmp_path,keys):
    good=fixture_csv(tmp_path/'raw','布A',0,speed=100);bad=fixture_csv(tmp_path/'raw','布B',0,speed=100)
    mp=bad.parent/'metadata.json';m=json.loads(mp.read_text());node=m
    for key in keys[:-1]:node=node[key]
    node[keys[-1]]=['invalid'];mp.write_text(json.dumps(m))
    records,report=import_dataset([good,bad],signals=('by',))
    assert len(records)==1 and len(report['errors'])==1 and report['errors'][0]['audit']['status']=='data_error'
