import numpy as np
from dataclasses import replace,asdict
import pytest
from cloth_trainer.online import OnlinePredictor,OnlineConfig,freeze_policy,validate_policy
from test_online import packet


def prefix_bundle():
    return dict(architecture='cera',signal='by',candidate=dict(window_size=4,spec=dict(revision='configurable-cera-prefix-v1'),prefix_policy=dict(min_points=4,step=2,samples_per_parent=2)))


def test_prefix_uses_all_history_and_locks_irreversibly():
    seen=[]
    cfg=OnlineConfig(window=4,stride=2,probability=.95,margin=.2,stable_seconds=.012,consecutive=2,lock_on_confirm=True)
    p=OnlinePredictor(prefix_bundle(),labels=['a','b'],predict_proba=lambda x:(seen.append(x.copy()) or np.array([.98,.02])),config=cfg)
    events=[e for i in range(16) for e in p.feed(packet(i))]
    assert [len(x) for x in seen]==[4,6,8]
    assert seen[-1][:4].tolist()==seen[0].tolist()
    confirmed=[e for e in events if e['type']=='prediction' and e['status']=='locked']
    assert len(confirmed)==1 and confirmed[0]['consecutive_count']==3
    p.feed(dict(packet(16),valid=False));end=p.feed(dict(type='end'))[0]
    assert end['prediction']=='a' and end['completion']['label']=='a' and len(seen)==3
    p.feed(dict(type='reset'));assert p.feed(dict(type='end'))[0]['prediction'] is None


def test_no_lock_before_all_gates_and_candidate_reset():
    seq=iter(([.98,.02],[.02,.98],[.02,.98],[.02,.98]))
    cfg=OnlineConfig(window=4,stride=2,stable_seconds=.012,consecutive=3,lock_on_confirm=True)
    p=OnlinePredictor(prefix_bundle(),labels=['a','b'],predict_proba=lambda x:np.array(next(seq)),config=cfg)
    events=[e for i in range(12) for e in p.feed(packet(i))]
    assert [e['consecutive_count'] for e in events if e['type']=='prediction']==[1,1,2,3]
    assert p.feed(dict(type='end'))[0]['prediction']=='b'


def test_old_policy_missing_stop_fields_validates_exact_hash(tmp_path):
    import hashlib,json
    path=tmp_path/'model.pt';path.write_bytes(b'model')
    settings=asdict(OnlineConfig());settings.pop('consecutive');settings.pop('lock_on_confirm')
    old=dict(format='cloth-online-policy-v1',model_sha256=hashlib.sha256(b'model').hexdigest(),settings=settings)
    old['fingerprint']=hashlib.sha256(json.dumps(old,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    cfg=validate_policy(old,path);assert cfg.consecutive==1 and not cfg.lock_on_confirm
    assert validate_policy(freeze_policy(path,replace(cfg,consecutive=2,lock_on_confirm=True)),path).lock_on_confirm
