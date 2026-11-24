"""Versioned single-capture review. Does not alter signals, baselines or weights."""
from pathlib import Path
import hashlib
import json
import numpy as np

ANNOTATIONS=json.loads((Path(__file__).parent/'resources/quality_annotations.json').read_text(encoding='utf-8'))['records']
ADMISSION_ANNOTATIONS=json.loads((Path(__file__).parent/'resources/admission_annotations.json').read_text(encoding='utf-8'))['records']
QUALITY={'normal':'可用·审查通过','suggest_review':'可用·建议复核','excluded':'本实验排除·建议复采',
         'undetermined':'待复核·无法判断','data_error':'不可用·数据错误','suggest_recollect':'可用·建议复采（旧规则）'}
RULE_VERSION='admission-v1'
RULE=dict(bin_mm=2.,minimum_bins=10,minimum_points_per_bin=3,rate=.10,same_sign_fraction=.80,
          minimum_difference_uT=5.,noise_multiplier=3.,denominator_epsilon_uT=1e-6,class_rate=.80,minimum_class_groups=10,
          common_center='median_of_valid_samples',group_policy='all_members_assessable_and_all_marked',by_provenance='all_used_contact_deltas_verifiable_or_computed')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def current_rule():
    definition=dict(version=RULE_VERSION,parameters=RULE,annotations=ADMISSION_ANNOTATIONS)
    return dict(definition,sha256=digest(definition))


def direction_metrics(record):
    d=dict(rule=RULE.copy(),status='not_assessable',marked=False,reason='缺少可审查的By通道')
    trials=record.views.get('by',record.trials)
    if any('y' not in (t.axes or record.axes) for t in trials):return d
    curves=[]
    for t in trials:
        col=(t.axes or record.axes).index('y')
        good=(t.raw_valid if t.raw_valid is not None else True)&np.isfinite(t.raw_x)&np.isfinite(t.raw_values[:,col])&(t.raw_x>=min(t.bounds))&(t.raw_x<=max(t.bounds))
        curves.append((t.raw_x[good],np.abs(t.raw_values[good,col])))
    if len(curves)!=2 or any(not len(x) for x,_ in curves):return dict(d,reason='缺少两方向有效采样')
    a,b=curves;lo=max(a[0].min(),b[0].min());hi=min(a[0].max(),b[0].max())
    if hi<=lo:return dict(d,reason='共同X范围不足')
    common=[v[(x>=lo)&(x<=hi)] for x,v in curves]
    full=np.array([np.median(v) for _,v in curves]);center=np.array([np.median(v) for v in common])
    df,dc=float(full[1]-full[0]),float(center[1]-center[0]);den_full=float(full.mean());den_common=float(center.mean())
    d.update(full_medians=full.tolist(),common_medians=center.tolist(),common_x_mm=[float(lo),float(hi)],common_points=[len(v) for v in common],
             full_difference=df,common_difference=dc,full_denominator=den_full,common_denominator=den_common)
    pairs=[]
    for x in np.arange(np.ceil(lo/RULE['bin_mm'])*RULE['bin_mm'],hi,RULE['bin_mm']):
        end=min(x+RULE['bin_mm'],hi)
        pieces=[v[(xx>=x)&(xx<end)] for xx,v in curves]
        if all(len(v)>=RULE['minimum_points_per_bin'] for v in pieces):pairs.append([float(np.median(v)) for v in pieces])
    d.update(bins=len(pairs),bin_medians=pairs)
    reasons=[]
    if den_full<=RULE['denominator_epsilon_uT'] or den_common<=RULE['denominator_epsilon_uT']:reasons.append('幅值分母接近零')
    else:d.update(full_rate=abs(df)/den_full,common_rate=abs(dc)/den_common)
    if len(pairs)<RULE['minimum_bins']:reasons.append('共同有效位置箱不足')
    sigma=record.diagnostics.get('baseline_sigma')
    try:noise=float(sigma[1])
    except (TypeError,ValueError,IndexError,KeyError):noise=float('nan')
    if not np.isfinite(noise) or noise<0:reasons.append('缺少有效By基线噪声标准差')
    else:d.update(baseline_noise_uT=noise,amplitude_threshold_uT=max(RULE['minimum_difference_uT'],RULE['noise_multiplier']*noise))
    if pairs:d['same_sign_fraction']=float(np.mean(np.sign(np.diff(pairs,axis=1)[:,0])==np.sign(dc))) if dc else 0.
    if reasons:return dict(d,reason='；'.join(reasons))
    conditions=dict(rates=d['full_rate']>=RULE['rate'] and d['common_rate']>=RULE['rate'],same_direction=df*dc>0,
                    amplitude=abs(dc)>=d['amplitude_threshold_uT'],persistent=d['same_sign_fraction']>=RULE['same_sign_fraction'])
    return dict(d,status='assessed',reason='',conditions={k:bool(v) for k,v in conditions.items()},marked=bool(all(conditions.values())))


def annotate(record,annotations=None):
    d=direction_metrics(record);record.diagnostics['direction_quality']=d
    notes=[];history=None
    for item in ADMISSION_ANNOTATIONS if annotations is None else annotations:
        if record.sha256==item['sha256'] and record.metadata_sha256==item.get('metadata_sha256'):
            history=item.get('decision',item.get('quality_status'));notes.append(item.get('reason','历史记录'))
    if history=='excluded':status='excluded'
    elif record.diagnostics.get('by_verification',{}).get('status','verified')!='verified':status='undetermined';notes.append('By差分缺少完整可验证依据，见by_verification计数')
    elif d['status']!='assessed':status='undetermined';notes.append(d['reason'])
    elif d['marked']:status='excluded';notes.append('四项大幅去回不一致条件全部满足')
    elif history in ('suggest_review','suggest_recollect') or max(d['full_rate'],d['common_rate'])>=RULE['rate'] or record.diagnostics.get('review_clues'):
        status='suggest_review';notes.append('存在待核查线索，未完整满足四项大幅异常条件')
    else:status='normal'
    notes.extend(record.diagnostics.get('review_clues',[]))
    record.quality=status;record.quality_notes=list(dict.fromkeys(notes))
    record.audit=dict(rule_version=RULE_VERSION,rule_sha256=current_rule()['sha256'],status=status,eligible=status in ('normal','suggest_review'),
                      single_status=status,single_eligible=status in ('normal','suggest_review'),history=history,reasons=record.quality_notes.copy())
    return record
