import copy
import pytest
from cloth_trainer.calibration import calibrate,stop_trial,group_risk_summary

def rows(n=5):
    return [dict(id=f'{c}{i}:{d}',group=f'{c}{i}',true=c,label=['A','B'][c],direction=d,
                 checkpoints=[dict(scores=[.99,.01] if c==0 else [.01,.99],risk=.001,checkpoint=q,predicted=c) for q in [.1,.2,.3,.5,.75,1.]])
            for c in [0,1] for i in range(n) for d in ['out','back']]

def test_small_calibration_never_claims_five_percent():
    state=calibrate(rows(),rows(),['A','B'],family_size=63)
    assert state['status']=='insufficient_evidence'
    assert all(q is None for q in state['quantiles'])
    assert not stop_trial(rows()[0],state)['accepted']

def test_two_checkpoint_stability_and_full_fallback():
    state=dict(status='certified',quantiles=[.1]*6,theta=.05,variant='full')
    a=stop_trial(rows()[0],state)
    assert a['accepted'] and a['stop_fraction']==.2
    state['theta']=0
    a=stop_trial(rows()[0],state)
    assert not a['accepted'] and a['stop_fraction']==1.

def test_unequal_group_sizes_do_not_inflate_independence():
    rr=rows(2);r=rr[0]
    many=rr+[copy.deepcopy(r) for _ in range(50)]
    a=calibrate(rr,rr,['A','B'],family_size=21)
    b=calibrate(rr,many,['A','B'],family_size=21)
    assert a['audit'][0]['class_bounds']==b['audit'][0]['class_bounds']
    assert a['audit'][0]['class_groups']==b['audit'][0]['class_groups']

def test_stopping_summary_uses_group_and_class_weights_not_session_sizes():
    from cloth_trainer.calibration import stopping_summary
    rr=[dict(label='A',group='big') for _ in range(100)]+[dict(label='A',group='small')]+[dict(label='B',group='third')]
    oo=[dict(accepted=True,predicted=0,stop_fraction=.2) for _ in range(100)]+[dict(accepted=True,predicted=0,stop_fraction=1.)]+[dict(accepted=False,predicted=1,stop_fraction=1.)]
    for r in rr:r['true']=0 if r['label']=='A' else 1
    result=stopping_summary(rr,oo,['A','B'])['aggregate']
    assert result['acceptance']==.5
    assert result['nonacceptance']==.5
    assert result['mean_accepted_stop']==pytest.approx(.6)
    assert result['mean_stop']==pytest.approx(.8)
    assert result['median_stop']==1.

def test_large_independent_calibration_can_certify_first_stable_checkpoint():
    state=calibrate(rows(10),rows(1500),['A','B'],family_size=21)
    assert state['status']=='certified'
    outcome=stop_trial(rows()[0],state)
    assert outcome['accepted'] and outcome['stop_fraction']==.2
    assert all(u<=0 for a in state['audit'] if a['passed'] for u in a['class_bounds'].values())
