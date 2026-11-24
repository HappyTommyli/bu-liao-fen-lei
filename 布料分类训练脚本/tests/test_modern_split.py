from types import SimpleNamespace
import pytest


def records(n=10):return [SimpleNamespace(group=f'{c}{i}',label=c) for c in ('A','B') for i in range(n)]


def test_modern_defaults_and_roundtrip():
    from cloth_trainer.modern_config import ModernConfig
    c=ModernConfig();assert c.device=='cuda' and c.lr==.0003 and c.early_stopping
    assert ModernConfig.from_dict(c.to_dict())==c
    assert c.with_preset('standard').early_stopping
    assert ModernConfig(architectures=('conv_gru','conv_cera')).validate()


def test_grouped_dev_cv_has_no_test_leakage_and_is_deterministic():
    from cloth_trainer.modern_split import make_plan,validate_plan
    rr=records();p=make_plan(rr,42);assert p==make_plan(rr,42)
    assert len(p['test'])==4 and len(p['dev'])==16 and len(p['cv'])==5
    validate_plan(rr,p)
    assert sorted(i for fold in p['cv'] for i in fold['val'])==sorted(p['dev'])
    assert all(not set(f['train']+f['val'])&set(p['test']) for f in p['cv'])
    bad=dict(p,test=p['test']+[p['dev'][0]])
    with pytest.raises(ValueError):validate_plan(rr,bad)


def test_paired_records_and_multilabel_sessions_are_indivisible():
    from cloth_trainer.modern_split import make_plan
    rr=[SimpleNamespace(group=f'session{i}',label=c) for i in range(12) for c in ('A','B')]
    p=make_plan(rr,7)
    for split in [dict(train=p['dev'],val=p['test'])]+p['cv']:
        assert not {rr[i].group for i in split['train']}&{rr[i].group for i in split['val']}


def test_too_few_independent_groups_fail_explicitly():
    from cloth_trainer.modern_split import make_plan
    with pytest.raises(ValueError,match='独立|组'):make_plan(records(5),42)
