import csv,json,shutil
from dataclasses import replace
import numpy as np
import pytest
from cloth_trainer.config import DataConfig,TrainConfig
from cloth_trainer.data import load_record,prepare_trials,import_dataset,file_hash
from test_data import fixture_csv


def rewrite(p,drop=(),modify=None):
    with p.open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);fields=[k for k in r.fieldnames if k not in drop];rows=list(r)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader()
        for row in rows:
            if modify:modify(row)
            w.writerow(row)


def test_by_only_unused_bx_not_required_and_abs_both(tmp_path):
    p=fixture_csv(tmp_path,speed=100)
    rewrite(p,drop=('Bx_uT','dBx_uT'),modify=lambda r:r.update(dBy_uT=str(-abs(float(r['dBy_uT']))),By_uT=str(-abs(float(r['dBy_uT'])))))
    r=load_record(p,signals=('by',))
    assert r.axes==('y',)
    assert all((t.values>=0).all() and t.values.shape[1]==1 for t in prepare_trials([r],'by'))
    with pytest.raises(ValueError,match='Bx'):load_record(p,signals=('bxby',))


def test_difference_baseline_conflict_or_untraceable_is_visible(tmp_path):
    p=fixture_csv(tmp_path,speed=100)
    rewrite(p,modify=lambda r:r.update(dBy_uT=str(float(r['dBy_uT'])+1)))
    with pytest.raises(ValueError,match='差分|基线'):load_record(p,signals=('by',))
    q=fixture_csv(tmp_path/'missing',speed=100);m=json.loads((q.parent/'metadata.json').read_text());m.pop('baseline_B0_uT')
    (q.parent/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='来源待确认|基线'):load_record(q,signals=('by',))
    legacy=load_record(q,replace(DataConfig(),preprocess_version='legacy-v2'),signals=('by',))
    assert len(legacy.trials)==2


def test_import_screening_ignore_duplicate_and_conflict(tmp_path):
    p=fixture_csv(tmp_path/'good',speed=100)
    for name in ('.workbench-trash','outputs','__pycache__'):
        q=tmp_path/name/'copy.csv';q.parent.mkdir();shutil.copy(p,q)
    records,report=import_dataset([tmp_path],signals=('by',))
    assert len(records)==1 and report['ignored_count']==3 and not report['errors']
    other=tmp_path/'duplicate';other.mkdir();shutil.copy(p,other/'copy.csv');shutil.copy(p.parent/'metadata.json',other/'metadata.json')
    records,report=import_dataset([tmp_path],signals=('by',))
    assert len(records)==1 and len(report['duplicates'])==1
    md=json.loads((other/'metadata.json').read_text());md['label']='冲突布';(other/'metadata.json').write_text(json.dumps(md))
    records,report=import_dataset([tmp_path],signals=('by',))
    assert not records and any('冲突' in e['message'] for e in report['errors'])


def test_history_flags_use_fingerprint_not_new_class_record(tmp_path):
    from cloth_trainer.quality import annotate
    r=load_record(fixture_csv(tmp_path,speed=100),signals=('by',))
    before=r.trials[0].values.copy()
    annotations=[dict(sha256=r.sha256,metadata_sha256=r.metadata_sha256,decision='excluded',reason='历史7次确认')]
    annotate(r,annotations)
    assert r.quality=='excluded' and r.manifest()['sample_weight']==1 and not r.audit['eligible']
    assert np.array_equal(before,r.trials[0].values)
    other=load_record(fixture_csv(tmp_path,i=1,speed=100),signals=('by',));annotate(other,annotations)
    assert other.quality!='excluded'


def test_r3_defaults_and_legacy_config_dispatch():
    c=TrainConfig();assert c.signals==('by',) and not c.early_stopping
    assert c.data.preprocess_version=='abs-v3'
    old=TrainConfig.from_dict({'data':{},'signals':['by']})
    assert old.data.preprocess_version=='legacy-v2'


def test_explicit_motor_fault_in_contact_and_conflicting_baselines(tmp_path):
    p=fixture_csv(tmp_path/'fault',speed=100)
    with p.open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);fields=list(r.fieldnames)+['motor_fault'];rows=list(r)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader()
        for row in rows:
            row['motor_fault']='LIMIT' if row['stage']=='step_003' and 60<float(row['motor_x_pulses'])/250<80 else 'NONE';w.writerow(row)
    with pytest.raises(ValueError,match='故障'):load_record(p,signals=('by',))
    q=fixture_csv(tmp_path/'baseline',speed=100);mp=q.parent/'metadata.json';m=json.loads(mp.read_text());m['context']['baseline']={'baseline':[0,1,0]};mp.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='基线.*冲突'):load_record(q,signals=('by',))


def test_grid_cannot_extend_beyond_last_valid_observation(tmp_path):
    p=fixture_csv(tmp_path,speed=6)
    rewrite(p,modify=lambda row:row.update(sample_valid='0') if row['stage']=='step_003' and float(row['motor_x_pulses'])/250>DataConfig().out_high-1.3 else None)
    with pytest.raises(ValueError,match='最后有效|完整裁剪'):load_record(p,signals=('by',))


def test_optional_bx_only_does_not_require_unused_by(tmp_path):
    p=fixture_csv(tmp_path,speed=100);rewrite(p,drop=('By_uT','dBy_uT'))
    r=load_record(p,signals=('bx',))
    assert r.axes==('x',) and prepare_trials([r],'bx')[0].values.shape[1]==1


def test_quality_rules_do_not_treat_invalid_samples_as_real_observations(tmp_path):
    p=fixture_csv(tmp_path,speed=30)
    def change(row):
        if row['stage']=='step_004' and int(round(float(row['sensor_t_s'])*333))%3:
            row.update(sample_valid='0',By_uT='500',dBy_uT='500')
    rewrite(p,modify=change)
    m=json.loads((p.parent/'metadata.json').read_text());m['context']['baseline']={'baseline':[0,0,0],'sigma':[1,1,1]};(p.parent/'metadata.json').write_text(json.dumps(m))
    r=load_record(p,signals=('by',))
    assert not r.diagnostics['direction_quality'].get('marked',False)


def test_high_rate_filter_keeps_clock_when_invalid_points_are_held(tmp_path):
    def make(root,invalid):
        p=fixture_csv(root,speed=30)
        with p.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['sensor_t_s','stage','motor_x_pulses','dBx_uT','dBy_uT','Bx_uT','By_uT','sample_valid','device_clock_segment'])
            for j,stage in enumerate(('step_003','step_004')):
                previous=0.
                for n in range(int(155/30*500)+1):
                    t=n/500;x=30*t if j==0 else 155-30*t;y=10+np.sin(100*t)
                    bad=n%7==4
                    if bad:y=999 if invalid else previous
                    else:previous=y
                    w.writerow([t+j*10,stage,x*250,0,y,0,y,0 if bad and invalid else 1,0])
        return p
    cfg=replace(DataConfig(),source_fs=500)
    a=load_record(make(tmp_path/'invalid',True),cfg,signals=('by',));b=load_record(make(tmp_path/'held',False),cfg,signals=('by',))
    assert np.allclose(a.trials[0].values,b.trials[0].values,atol=1e-5)


@pytest.mark.parametrize('signal',['bxby','xy'])
def test_prepared_by_only_records_reject_unavailable_axes(tmp_path,signal):
    r=load_record(fixture_csv(tmp_path,speed=100),signals=('by',))
    with pytest.raises(ValueError,match='通道'):prepare_trials([r],signal)


def test_legacy_label_protocol_unchanged(tmp_path):
    p=fixture_csv(tmp_path,speed=100);target=p.parent/'另一种布_999.csv';p.rename(target)
    with pytest.raises(ValueError,match='标签'):load_record(target,signals=('by',))
    assert load_record(target,replace(DataConfig(),preprocess_version='legacy-v2'),signals=('by',)).label=='布A'


def test_joint_comparison_by_view_matches_by_only_import(tmp_path):
    p=fixture_csv(tmp_path,speed=100)
    rewrite(p,modify=lambda row:row.update(Bx_uT='nan',dBx_uT='nan') if row['stage']=='step_003' and .8<float(row['sensor_t_s'])<.804 else None)
    joint=load_record(p,signals=('by','bxby'));by=load_record(p,signals=('by',))
    assert np.array_equal(prepare_trials([joint],'by')[0].values,prepare_trials([by],'by')[0].values)
