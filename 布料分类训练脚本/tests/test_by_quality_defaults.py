import csv
from dataclasses import replace
import pytest
from cloth_trainer.config import DataConfig
from cloth_trainer.data import load_record
from test_data import fixture_csv


def rewrite(path,change):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);rows=list(reader);fields=list(reader.fieldnames)
    for key in ('invalid_reason','Bz_uT'):
        if key not in fields:fields.append(key)
    change(rows)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def test_new_defaults_and_legacy_deserialization():
    cfg=DataConfig()
    assert (cfg.out_low,cfg.out_high,cfg.back_low,cfg.back_high)==(51,140,30,120)
    assert cfg.contact_rule=='fixed-x-v1'
    assert cfg.sample_quality=='selected-axis-v1'
    legacy=DataConfig.from_dict({})
    assert (legacy.out_low,legacy.out_high,legacy.back_low,legacy.back_high)==(45,142,16,114)
    assert legacy.sample_quality=='recorded-v1'
    assert DataConfig.from_dict(vars(cfg))==cfg


def test_only_other_axis_spikes_are_restored_without_changing_sources(tmp_path):
    p=fixture_csv(tmp_path)
    def change(rows):
        for r in rows:
            if r['stage']=='step_003' and 2.5<float(r['sensor_t_s'])<2.6:
                r.update(sample_valid='0',invalid_reason='isolated_spike',Bz_uT='7900')
    rewrite(p,change);before=p.read_bytes()
    cfg=DataConfig()
    r=load_record(p,cfg,signals=('by',))
    assert r.quality=='normal' and p.read_bytes()==before
    assert r.diagnostics['sample_quality']['restored_other_axis_spike']>20
    with pytest.raises(ValueError,match='缺口'):
        load_record(p,replace(cfg,sample_quality='recorded-v1'),signals=('by',))


@pytest.mark.parametrize('reason',['unknown_failure','nonfinite'])
def test_unknown_invalid_reasons_are_not_blindly_restored(tmp_path,reason):
    p=fixture_csv(tmp_path)
    def change(rows):
        for r in rows:
            if r['stage']=='step_003' and 2.5<float(r['sensor_t_s'])<2.6:r.update(sample_valid='0',invalid_reason=reason)
    rewrite(p,change)
    with pytest.raises(ValueError,match='缺口'):load_record(p,signals=('by',))


def test_real_time_gaps_still_fail(tmp_path):
    with pytest.raises(ValueError,match='缺口'):load_record(fixture_csv(tmp_path,gap=True),signals=('by',))


def test_by_spikes_are_checked_even_when_original_flag_is_valid(tmp_path):
    from cloth_trainer.sample_quality import selected_axis_rows
    rows=[dict(By_uT=str(v),sample_valid='1') for v in (0,1600,0)]
    detail={};out=list(selected_axis_rows(rows,('y',),detail))
    assert out[1]['sample_valid']=='0' and detail['selected_axis_spikes']==1
    assert rows[1]['sample_valid']=='1'


def test_startup_uses_fixed_directory_and_respects_explicit_sources(tmp_path):
    from train_cloth import startup_sources
    directory=tmp_path.parent/'xz-control/data/布料采集数据新'
    assert startup_sources(root=tmp_path)==(None,directory)
    assert startup_sources(project=tmp_path/'old.json',root=tmp_path)==(tmp_path/'old.json',None)
    assert startup_sources(directory=tmp_path/'raw',root=tmp_path)==(None,tmp_path/'raw')
    assert startup_sources(screenshot=True,root=tmp_path)==(None,None)


def test_legacy_batch_recheck_keeps_its_original_diagnostics(tmp_path):
    from cloth_trainer.data import import_dataset
    from cloth_trainer.admission import verify_raw_batches
    from cloth_trainer.quality import digest
    cfg=DataConfig.from_dict({'preprocess_version':'abs-v3'})
    p=fixture_csv(tmp_path)
    _,report=import_dataset([p],cfg,signals=('by',))
    from copy import deepcopy
    batch=deepcopy(report['review_batches'][0])
    batch['data'].pop('sample_quality',None)
    for e in batch['entries']:e['diagnostics'].pop('sample_quality',None)
    batch['sha256']=digest({k:v for k,v in batch.items() if k!='sha256'})
    assert len(verify_raw_batches([batch],cfg))==1
