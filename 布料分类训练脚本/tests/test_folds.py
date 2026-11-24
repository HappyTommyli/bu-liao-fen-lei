from types import SimpleNamespace
import pytest
from cloth_trainer.folds import make_folds,fold_partition


def test_fivefold_whole_groups_coverage_and_once_only_oof():
    records=[SimpleNamespace(group=f'{c}:{i}',label=c) for c in 'AB' for i in range(10) for direction in range(2)]
    folds=make_folds(records,42)
    assert make_folds(records,42)==folds and len(folds)==5
    tested=[]
    for i in range(5):
        p=fold_partition(folds,i);assert [len(p[k]) for k in ('train','val','test')]==[24,8,8]
        sets=[{records[j].group for j in v} for v in p.values()]
        assert all(not a&b for k,a in enumerate(sets) for b in sets[k+1:])
        assert all({records[j].label for j in v}=={'A','B'} for v in p.values())
        tested+=p['test']
    assert sorted(tested)==list(range(len(records)))


def test_five_groups_per_class_work_one_does_not():
    records=[SimpleNamespace(group=f'{c}:{i}',label=c) for c in 'AB' for i in range(5)]
    assert len(make_folds(records))==5
    with pytest.raises(ValueError,match='覆盖|独立组'):make_folds(records+[SimpleNamespace(group='C',label='铜氨丝')])


def test_mixed_sessions_remain_whole():
    records=[SimpleNamespace(group=str(i),label=c) for i in range(10) for c in 'ABC']
    folds=make_folds(records)
    assert all(len(v)==6 for v in folds)
    assert all(len({records[j].group for j in v})==2 for v in folds)
