import csv
import json
from dataclasses import replace
import numpy as np
import pytest
from cloth_trainer.config import DataConfig
from cloth_trainer.data import load_record, import_records, prepare_trials, split_records


def fixture_csv(root, label='布A', i=0, gap=False, future=False, speed=30):
    folder=root/f'{label}_{i:03}'
    folder.mkdir(parents=True, exist_ok=True)
    path=folder/(folder.name+'.csv')
    fields=['sensor_t_s','stage','motor_x_pulses','dBx_uT','dBy_uT','Bx_uT','By_uT','sample_valid','device_clock_segment']
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(fields)
        for j,stage in enumerate(['step_003','step_004']):
            for n in range(int(155/speed*333)+1):
                t=n/333;x=t*speed if j==0 else 155-t*speed
                if gap and 2<t<2.1:continue
                by=(1 if j==0 else -1)*(10+i+(0 if label=='布A' else 20)+2*np.sin(20*t))
                if future and t>3:by+=1000
                w.writerow([t+j*10,stage,x*250,2+np.cos(t),by,2+np.cos(t),by,1,0])
    (folder/'metadata.json').write_text(json.dumps({'label':label,'baseline_B0_uT':{'Bx_uT':0,'By_uT':0},'context':{'config':{'x_ppmm':250,'x_speed':speed},'baseline':{'sigma':[.1,1.,.1]}}}))
    return path


def test_crops_signs_and_causal_hold(tmp_path):
    p=fixture_csv(tmp_path);before=p.read_bytes()
    r=load_record(p,DataConfig())
    assert len(r.trials)==2
    a=prepare_trials([r],'by')
    assert all(t.values.shape[1]==1 and (t.values>0).all() for t in a)
    b=prepare_trials([r],'bxby')
    assert b[0].values.shape[1]==2
    assert b[1].values.min()>0
    future=load_record(fixture_csv(tmp_path/'future',future=True),DataConfig())
    assert np.array_equal(r.trials[0].values[:300],future.trials[0].values[:300])
    assert p.read_bytes()==before


def test_long_gap_rejects_whole_record(tmp_path):
    with pytest.raises(ValueError,match='缺口'):
        load_record(fixture_csv(tmp_path,gap=True),DataConfig())


def test_duplicates_and_missing_label_override(tmp_path):
    p=fixture_csv(tmp_path);q=tmp_path/'copy.csv';q.write_bytes(p.read_bytes())
    (tmp_path/'metadata.json').write_bytes((p.parent/'metadata.json').read_bytes())
    records,errors=import_records([p,q],DataConfig())
    assert len(records)==1 and '重复' in errors[0]['message']
    assert load_record(p,DataConfig(),label='新布').label=='新布'


def test_fixed_group_split_no_prefix_leakage(tmp_path):
    records=[load_record(fixture_csv(tmp_path,label,i),DataConfig()) for label in ['布A','布B'] for i in range(10)]
    split=split_records(records,42)
    assert [len(split[k]) for k in ['train','val','test']]==[14,2,4]
    gs=[{records[i].group for i in split[k]} for k in split]
    assert sum(len(g) for g in gs)==len(set.union(*gs))
    assert sorted(sum(split.values(),[]))==list(range(20))
    with pytest.raises(ValueError,match='10'):
        split_records(records[:5]+records[10:15])

def test_group_quotas_for_fifty_and_reproducibility():
    from types import SimpleNamespace
    records=[SimpleNamespace(group=f'{c}:{i}',label=c) for c in ['A','B'] for i in range(50)]
    split=split_records(records,42)
    assert [len(split[k]) for k in ['train','val','test']]==[70,10,20]
    assert split_records(records,42)==split


def test_xy_norm_and_metadata_baseline(tmp_path):
    p=fixture_csv(tmp_path);r=load_record(p,DataConfig())
    xy=prepare_trials([r],'xy')[0].values[:,0]
    assert np.allclose(xy,np.linalg.norm(r.trials[0].values,axis=1))
    md=json.loads((p.parent/'metadata.json').read_text());md.pop('baseline_B0_uT');(p.parent/'metadata.json').write_text(json.dumps(md))
    with pytest.raises(ValueError,match='基线'):load_record(p,DataConfig())

def test_fixed_horizon_tolerates_software_position_registration_delay(tmp_path):
    p=fixture_csv(tmp_path)
    with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f));fields=list(rows[0])
    for row in rows:row['sensor_t_s']=str(float(row['sensor_t_s'])*.97)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    r=load_record(p,DataConfig())
    assert len(r.trials)==2

def test_high_rate_future_timestamps_do_not_change_earlier_inputs(tmp_path):
    def source(folder,later_rate):
        p=fixture_csv(folder)
        with p.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['sensor_t_s','stage','motor_x_pulses','dBx_uT','dBy_uT','Bx_uT','By_uT','sample_valid','device_clock_segment'])
            for j,stage in enumerate(['step_003','step_004']):
                tt=np.r_[np.arange(0,2,1/500),np.arange(2,155/30,1/later_rate)]
                for t in tt:
                    x=t*30 if j==0 else 155-t*30
                    w.writerow([t+j*10,stage,x*250,2+np.sin(1200*t),10+np.sin(1100*t),2+np.sin(1200*t),10+np.sin(1100*t),1,0])
        return p
    cfg=DataConfig(source_fs=500)
    a=load_record(source(tmp_path/'a',500),cfg);b=load_record(source(tmp_path/'b',505),cfg)
    assert np.array_equal(a.trials[0].values[:100],b.trials[0].values[:100])


def test_mixed_capture_groups_allow_approximate_quotas_without_splitting():
    from types import SimpleNamespace
    records=[SimpleNamespace(group=f'{pair}:{i}',label=c) for pair in ['AB','AC','BC'] for i in range(5) for c in pair]
    split=split_records(records,42)
    groups=[{records[i].group for i in v} for v in split.values()]
    assert all(not (a&b) for j,a in enumerate(groups) for b in groups[j+1:])
    assert all({records[i].label for i in v}=={'A','B','C'} for v in split.values())
