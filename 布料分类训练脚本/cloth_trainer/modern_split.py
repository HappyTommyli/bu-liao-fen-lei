"""Fixed group holdout and development-only fivefold model selection."""
import numpy as np
from .folds import make_folds


def _select_holdout(records,seed,fraction,reserve):
    groups={}
    for r in records:groups.setdefault(r.group,set()).add(r.label)
    labels=sorted({r.label for r in records});counts={c:sum(c in v for v in groups.values()) for c in labels}
    if len(labels)<2 or '' in labels or any(n<reserve+1 for n in counts.values()):raise ValueError('内部测试加开发集：每类独立采集组不足，并需满足会话结构约束')
    if any(not getattr(r,'label_confirmed',True) for r in records):raise ValueError('文件名候选标签待确认，请明确指定标签')
    rng=np.random.default_rng(seed);test_groups=set()
    if all(len(v)==1 for v in groups.values()):
        for c in labels:
            gs=sorted(g for g in groups if c in groups[g]);rng.shuffle(gs)
            quota=min(len(gs)-reserve,max(1,round(fraction*len(gs))))
            test_groups.update(gs[:quota])
    else:
        from scipy.optimize import milp,Bounds,LinearConstraint
        gs=sorted(groups);ng=len(gs);nc=len(labels);rows=[];lo=[];hi=[]
        for ci,c in enumerate(labels):
            row=np.zeros(ng+2*nc);row[:ng]=[c in groups[g] for g in gs]
            rows.append(row.copy());lo.append(1);hi.append(counts[c]-reserve)
            row[ng+2*ci]=-1;row[ng+2*ci+1]=1
            rows.append(row);lo.append(fraction*counts[c]);hi.append(fraction*counts[c])
        result=milp(np.r_[rng.uniform(0,1e-6,ng),np.ones(2*nc)],integrality=np.r_[np.ones(ng),np.zeros(2*nc)],
                    bounds=Bounds(np.zeros(ng+2*nc),np.r_[np.ones(ng),np.full(2*nc,np.inf)]),
                    constraints=LinearConstraint(np.array(rows),lo,hi),options={'time_limit':10})
        if result.x is None:raise ValueError('独立会话组不足以同时覆盖内部测试和开发集五折')
        test_groups={g for i,g in enumerate(gs) if result.x[i]>.5}
    dev=[i for i,r in enumerate(records) if r.group not in test_groups];test=[i for i,r in enumerate(records) if r.group in test_groups]
    return dev,test


def make_plan(records,seed=42,cv_folds=5):
    if type(cv_folds) is not int or not 1<=cv_folds<=10:raise ValueError('开发集划分次数需为1到10')
    reserve=max(2,cv_folds)
    groups={}
    for r in records:groups.setdefault(r.group,set()).add(r.label)
    labels={r.label for r in records}
    # Reuse the historical test allocation whenever it leaves enough development groups.
    reference=5 if all(sum(c in v for v in groups.values())>=6 for c in labels) else reserve
    try:dev,test=_select_holdout(records,seed,.2,reference)
    except ValueError:
        if reference==reserve:raise
        dev,test=_select_holdout(records,seed,.2,reserve)
    if any(len({records[i].group for i in dev if records[i].label==c})<reserve for c in labels):
        dev,test=_select_holdout(records,seed,.2,reserve)
    if cv_folds==1:
        train,val=_select_holdout([records[i] for i in dev],seed+1,.25,1)
        cv=[dict(train=[dev[j] for j in train],val=[dev[j] for j in val])]
    else:
        folds=make_folds([records[i] for i in dev],seed+1,cv_folds)
        cv=[dict(train=[dev[j] for k,fold in enumerate(folds) if k!=f for j in fold],val=[dev[j] for j in folds[f]]) for f in range(cv_folds)]
    plan=dict(version='grouped-dev-test-v1' if cv_folds==5 else 'grouped-dev-test-v2',seed=seed,dev=dev,test=test,cv=cv)
    if cv_folds!=5:plan['cv_folds']=cv_folds
    validate_plan(records,plan);return plan


def validate_plan(records,plan):
    if plan.get('version') not in ('grouped-dev-test-v1','grouped-dev-test-v2'):raise ValueError('分区版本无效')
    count=plan.get('cv_folds',5)
    if type(count) is not int or not 1<=count<=10 or len(plan.get('cv',[]))!=count:raise ValueError('开发集划分次数不一致')
    labels={r.label for r in records}
    def indices(values):
        if not isinstance(values,list) or not values or any(type(i) is not int or not 0<=i<len(records) for i in values) or len(values)!=len(set(values)):
            raise ValueError('分区索引无效、重复或越界')
    dev=plan['dev'];test=plan['test'];indices(dev);indices(test)
    if sorted(dev+test)!=list(range(len(records))):raise ValueError('开发/测试索引有重叠或缺失')
    def check(a,b):
        if set(a)&set(b) or {records[i].group for i in a}&{records[i].group for i in b}:raise ValueError('采集组跨区泄漏')
        if {records[i].label for i in a}!=labels or {records[i].label for i in b}!=labels:raise ValueError('类别覆盖不足')
    check(dev,test)
    if count>1 and sorted(i for f in plan['cv'] for i in f['val'])!=sorted(dev):raise ValueError('开发集交叉验证覆盖错误')
    for f in plan['cv']:
        indices(f['train']);indices(f['val'])
        if sorted(f['train']+f['val'])!=sorted(dev):raise ValueError('开发折包含内部测试或缺失记录')
        check(f['train'],f['val'])
