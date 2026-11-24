"""Independent, group-aware finite-policy calibration; failure to certify is valid."""
from collections import defaultdict
import math
import numpy as np
from .config import CHECKPOINTS

def prediction_set(scores,q):
    return list(range(len(scores))) if q is None else [i for i,p in enumerate(scores) if 1-p<=q+1e-12]

def _simulate(row,quantiles,theta,variant):
    previous=None
    for i,p in enumerate(row['checkpoints']):
        predicted=p['predicted'];members=prediction_set(p['scores'],quantiles[i])
        stable=variant=='no_stability' or previous==predicted
        if len(members)==1 and stable and p['risk']<=theta:
            return dict(accepted=True,predicted=predicted,stop_fraction=p['checkpoint'],prediction_set=members,
                        raw_risk=p['risk'],stop_reason='accepted',checkpoint_index=i)
        previous=predicted
    last=row['checkpoints'][-1]
    return dict(accepted=False,predicted=last['predicted'],stop_fraction=1.,prediction_set=prediction_set(last['scores'],quantiles[-1]),
                raw_risk=last['risk'],stop_reason='full_length_no_early_stop',checkpoint_index=5)

def stop_trial(row,state):
    variant=state.get('variant','full')
    if state['status']=='certified':result=_simulate(row,state['quantiles'],state['theta'],variant)
    elif state['status']=='exploratory':
        result=_simulate(row,[None]*6,state.get('theta',.05),variant)
        # Uncalibrated ablation deliberately skips the set filter, never certified.
        previous=None
        for i,p in enumerate(row['checkpoints']):
            if previous==p['predicted'] and p['risk']<=state.get('theta',.05):
                result.update(accepted=True,predicted=p['predicted'],stop_fraction=p['checkpoint'],raw_risk=p['risk'],
                              stop_reason='exploratory_uncalibrated',checkpoint_index=i,prediction_set=[]);break
            previous=p['predicted']
    else:
        result=_simulate(row,state.get('quantiles',[None]*6),-1,variant)
        result['stop_reason']='fixed_full_length' if state['status']=='fixed_length' else 'calibration_insufficient_evidence'
    result['calibration_status']=state['status']
    return result

def group_risk_summary(rows,outcomes):
    groups=defaultdict(list)
    for r,o in zip(rows,outcomes):
        groups[(r['label'],r['group'])].append((float(o['accepted']),float(o['accepted'] and o['predicted']!=r['true']),o['stop_fraction']))
    by_class=defaultdict(list)
    for (label,g),values in groups.items():by_class[label].append(np.mean(values,axis=0))
    result={}
    for label,values in by_class.items():
        a,e,s=np.mean(values,axis=0)
        result[label]=dict(acceptance=float(a),accepted_error=float(e/a) if a else None,mean_stop=float(s),groups=len(values))
    return result

def stopping_summary(rows,outcomes,labels):
    """Uniform class -> uniform group -> uniform trial weights, as registered."""
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[(r['label'],r['group'])].append(i)
    counts={c:sum(label==c for label,g in groups) for c in labels}
    weights=np.zeros(len(rows))
    for (c,g),indices in groups.items():
        if c in counts and counts[c]:weights[indices]=1/(len(labels)*counts[c]*len(indices))
    def summary(indices,w):
        mass=float(w.sum())
        if not mass:return dict(acceptance=None,nonacceptance=None,accepted_error=None,mean_stop=None,median_stop=None,mean_accepted_stop=None,median_accepted_stop=None)
        w=w/mass;oo=[outcomes[i] for i in indices];rr=[rows[i] for i in indices]
        accept=np.array([o['accepted'] for o in oo],dtype=float);stop=np.array([o['stop_fraction'] for o in oo])
        error=np.array([o['predicted']!=r['true'] for r,o in zip(rr,oo)],dtype=float);accepted=float(w@accept)
        def median(value,ww):
            total=ww.sum()
            if not total:return None
            order=np.argsort(value,kind='stable');idx=min(len(order)-1,int(np.searchsorted(np.cumsum(ww[order]),total/2,side='left')))
            return float(value[order[idx]])
        return dict(acceptance=accepted,nonacceptance=1-accepted,accepted_error=float(w@(accept*error)/accepted) if accepted else None,
                    mean_stop=float(w@stop),median_stop=median(stop,w),mean_accepted_stop=float(w@(accept*stop)/accepted) if accepted else None,
                    median_accepted_stop=median(stop,w*accept))
    return dict(class_weights={c:1/len(labels) for c in labels},aggregate=summary(list(range(len(rows))),weights),
                per_class={c:summary([i for i,r in enumerate(rows) if r['label']==c],weights[[i for i,r in enumerate(rows) if r['label']==c]]) for c in labels})

def calibrate(set_rows,risk_rows,labels,family_size=63,variant='full',epsilon=.05,delta=.05,alpha=.05):
    base=dict(status='insufficient_evidence',variant=variant,epsilon=epsilon,delta=delta,alpha_set=alpha,
              family_size=family_size,theta=None,quantiles=[None]*6,audit=[])
    if variant in ('local','context'):
        base['status']='fixed_length';return base
    if variant=='no_calibration':
        base.update(status='exploratory',theta=.05);return base
    group_scores=defaultdict(lambda:np.full(6,-np.inf))
    for r in set_rows:
        scores=np.array([1-p['scores'][r['true']] for p in r['checkpoints']])
        group_scores[r['group']]=np.maximum(group_scores[r['group']],scores)
    n=len(group_scores);rank=math.ceil((n+1)*(1-alpha))
    if n and rank<=n:base['quantiles']=np.sort(np.stack(list(group_scores.values())),axis=0)[rank-1].tolist()
    base['set_groups']=n
    candidates=[]
    for theta in np.linspace(0,1,21):
        outcomes=[_simulate(r,base['quantiles'],float(theta),variant) for r in risk_rows]
        grouped=defaultdict(list)
        for r,o in zip(risk_rows,outcomes):
            grouped[(r['label'],r['group'])].append(float(o['accepted'])*((o['predicted']!=r['true'])-epsilon))
        bounds={};counts={}
        for label in labels:
            z=[float(np.mean(v)) for (c,g),v in grouped.items() if c==label];counts[label]=len(z)
            bounds[label]=float(np.mean(z)+math.sqrt(math.log(family_size*len(labels)/delta)/(2*len(z)))) if z else None
        passed=all(u is not None and u<=0 for u in bounds.values()) and any(o['accepted'] for o in outcomes)
        summary=stopping_summary(risk_rows,outcomes,labels)['aggregate']
        acceptance=summary['acceptance'] or 0.
        audit=dict(theta=float(theta),passed=passed,class_bounds=bounds,class_groups=counts,
                   acceptance=acceptance,mean_accepted_stop=summary['mean_accepted_stop'])
        base['audit'].append(audit)
        if passed:candidates.append(audit)
    if candidates:
        chosen=min(candidates,key=lambda c:(c['mean_accepted_stop'],-c['acceptance'],c['theta']))
        base.update(status='certified',theta=chosen['theta'])
    return base
