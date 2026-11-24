"""Reproducible fivefold allocation; capture groups are indivisible."""
import numpy as np


def make_folds(records,seed=42,count=5):
    if type(count) is not int or not 2<=count<=10:raise ValueError("折数需为2到10")
    labels=sorted({r.label for r in records})
    if len(labels)<2 or '' in labels:raise ValueError('至少2类且标签需确认，才能建立五折')
    if any(not getattr(r,'label_confirmed',True) for r in records):raise ValueError('文件名候选标签待确认，请明确指定标签')
    groups={}
    for r in records:groups.setdefault(r.group,set()).add(r.label)
    counts={c:sum(c in v for v in groups.values()) for c in labels}
    short={c:count-n for c,n in counts.items() if n<count}
    if short:raise ValueError('五折类别覆盖不足；各类还缺独立组：'+str(short))
    rng=np.random.default_rng(seed);alloc=[set() for _ in range(count)]
    if all(len(v)==1 for v in groups.values()):
        for ci,c in enumerate(labels):
            gs=sorted(g for g,v in groups.items() if c in v);rng.shuffle(gs)
            for j,g in enumerate(gs):alloc[(j+ci)%count].add(g)
    else:
        from scipy.optimize import milp,Bounds,LinearConstraint
        gs=sorted(groups);ng=len(gs);nc=len(labels);nv=count*ng+2*count*nc
        rows=[];lo=[];hi=[]
        for gi in range(ng):
            a=np.zeros(nv);a[count*gi:count*gi+count]=1;rows.append(a);lo.append(1);hi.append(1)
        for ci,c in enumerate(labels):
            for j in range(count):
                a=np.zeros(nv)
                for gi,g in enumerate(gs):a[count*gi+j]=int(c in groups[g])
                rows.append(a.copy());lo.append(1);hi.append(np.inf)
                a[count*ng+2*count*ci+2*j]=-1;a[count*ng+2*count*ci+2*j+1]=1
                rows.append(a);lo.append(counts[c]/count);hi.append(counts[c]/count)
        result=milp(np.r_[rng.uniform(0,1e-6,count*ng),np.ones(2*count*nc)],
                    integrality=np.r_[np.ones(count*ng),np.zeros(2*count*nc)],bounds=Bounds(np.zeros(nv),np.r_[np.ones(count*ng),np.full(2*count*nc,np.inf)]),
                    constraints=LinearConstraint(np.array(rows),lo,hi),options={'time_limit':10})
        if result.x is None:raise ValueError('会话组结构无法覆盖五折全部类别，需补独立采集；不能拆组或复制样本')
        choices=result.x[:count*ng].reshape(ng,count).round()
        if not np.all(choices.sum(1)==1):raise ValueError('无法取得可行五折组分配')
        alloc=[{g for gi,g in enumerate(gs) if choices[gi,j]==1} for j in range(count)]
    folds=[[i for i,r in enumerate(records) if r.group in group] for group in alloc]
    validate_folds(records,folds,count)
    return folds


def validate_folds(records,folds,count=5):
    if type(count) is not int or not 2<=count<=10:raise ValueError("折数需为2到10")
    if len(folds)!=count or sorted(i for fold in folds for i in fold)!=list(range(len(records))):raise ValueError('五折必须恰好覆盖每条记录一次')
    seen=set();labels={r.label for r in records}
    for fold in folds:
        groups={records[i].group for i in fold}
        if seen & groups:raise ValueError('同采集组跨折泄漏')
        seen.update(groups)
        if {records[i].label for i in fold}!=labels:raise ValueError('五折类别覆盖不足')


def fold_partition(folds,index):
    return dict(train=[i for j,fold in enumerate(folds) if j not in (index,(index+1)%len(folds)) for i in fold],val=list(folds[(index+1)%len(folds)]),test=list(folds[index]))
