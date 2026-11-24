"""Frozen TCN candidates and the single source of window/aggregation rules."""
import math
import torch
from torch.nn.utils.rnn import pad_sequence
from .tcn_models import default_spec
from .cera_models import normalize_model_spec as normalize_spec,CERA_REVISIONS,PREFIX_REVISION

STUDY_REVISION='tcn-workbench-v1'
TRAINING_REVISION='parent-window-v1'
AGGREGATION='length-probability-v1'
MODE_TITLES={'single':'单配置调参','four_way':'通道与切窗四组对照','window_pool':'窗口与汇总四组对照','overlap_pair':'重叠与不重叠两组对照'}
MODE_TITLES['prefix']='累积前缀CERA'
POOLING_TITLES={'mean':'均值 mean','mean_std':'均值＋标准差 mean_std','max':'最大值 max'}


def make_candidate(identity,spec,window,stride=0):
    result=dict(id=identity,spec=normalize_spec(spec),window_size=window,
                training_revision=TRAINING_REVISION,aggregation=AGGREGATION)
    if result['spec']['revision']==PREFIX_REVISION:
        if type(window) is not int or not 128<=window<=8192 or type(stride) is not int or not 1<=stride<=8192:raise ValueError('前缀观察量和间隔无效')
        return dict(result,prefix_policy=dict(min_points=window,step=stride,samples_per_parent=2),training_revision='parent-prefix-v1',aggregation='final-prefix-probability-v1')
    if stride and stride!=window:
        if not window or type(stride) is not int or not 1<=stride<window:raise ValueError('训练步长需在1到窗口长度之间')
        result.update(window_stride=stride,training_revision='parent-window-v2',aggregation='normalized-window-probability-v2')
    return result


def validate_candidate(candidate):
    if isinstance(candidate,dict) and 'prefix_policy' in candidate:
        if not isinstance(candidate.get('id'),str) or not candidate['id'] or not candidate['id'].replace('_','').isalnum():raise ValueError('前缀候选ID无效')
        policy=candidate['prefix_policy']
        if (not isinstance(policy,dict) or set(policy)!={'min_points','step','samples_per_parent'} or policy['samples_per_parent']!=2
                or candidate.get('window_size')!=policy['min_points']):raise ValueError('前缀策略无效')
        expected=make_candidate(candidate['id'],candidate['spec'],candidate['window_size'],policy['step'])
        if expected!=candidate or expected['spec']['revision']!=PREFIX_REVISION:raise ValueError('前缀候选字段或版本不支持')
        return expected
    if not isinstance(candidate,dict) or set(candidate)-{'id','spec','window_size','training_revision','aggregation','window_stride'} or not {'id','spec','window_size','training_revision','aggregation'}<=set(candidate):
        raise ValueError('TCN候选字段无效')
    overlap='window_stride' in candidate
    if (candidate['training_revision'],candidate['aggregation'])!=(('parent-window-v2','normalized-window-probability-v2') if overlap else (TRAINING_REVISION,AGGREGATION)):
        raise ValueError('TCN训练或聚合版本不支持')
    window=candidate['window_size']
    if not isinstance(window,int) or isinstance(window,bool) or (window!=0 and not 128<=window<=8192):
        raise ValueError('窗口长度需为0（完整单程）或128–8192整数')
    if not isinstance(candidate['id'],str) or not candidate['id'] or not candidate['id'].replace('_','').isalnum():
        raise ValueError('TCN候选ID无效')
    if overlap and (type(candidate['window_stride']) is not int or not 0<candidate['window_stride']<window):raise ValueError('重叠训练步长无效')
    return make_candidate(candidate['id'],candidate['spec'],window,candidate.get('window_stride',0))


def candidates(cfg):
    if cfg.tcn_spec is None:return []
    if cfg.tcn_mode=='prefix':return [make_candidate('prefix',cfg.tcn_spec,cfg.tcn_window,cfg.tcn_stride)]
    if cfg.tcn_mode=='overlap_pair':
        return [make_candidate('overlap_a_nonoverlap',cfg.tcn_spec,cfg.tcn_window),
                make_candidate('overlap_b_overlap',cfg.tcn_spec,cfg.tcn_window,cfg.tcn_stride)]
    if cfg.tcn_mode=='four_way':
        return [make_candidate(f'{mode}_h{width}',default_spec(width),window)
                for mode,window in [('full',0),('w900',900)] for width in (64,128)]
    if cfg.tcn_mode=='window_pool':
        return [make_candidate(identity,dict(default_spec(128),pooling=pooling),window)
                for identity,window,pooling in [('wp_a_w900_mean',900,'mean'),('wp_b_w900_meanstd',900,'mean_std'),
                                               ('wp_c_w450_mean',450,'mean'),('wp_d_w1800_mean',1800,'mean')]]
    return [make_candidate('custom',cfg.tcn_spec,cfg.tcn_window,getattr(cfg,'tcn_stride',0))]


def window_slices(length,size,stride=0):
    if length<1 or size<0:raise ValueError('单程不能为空，窗口长度不能为负')
    size=size or length
    stride=stride or size
    if type(stride) is not int or not 1<=stride<=size:raise ValueError('训练步长需在1到窗口长度之间')
    result=[]
    for start in range(0,length,stride):
        end=min(start+size,length);result.append((start,end))
        if end==length:break
    return result


def split_batch(x,lengths,size,stride=0):
    windows=[];lens=[];parents=[];weights=[];starts=[]
    for parent,length in enumerate(lengths.tolist()):
        slices=window_slices(length,size,stride);denominator=sum(end-start for start,end in slices)
        for start,end in slices:
            windows.append(x[parent,start:end]);lens.append(end-start)
            parents.append(parent);weights.append((end-start)/denominator);starts.append(start)
    return (pad_sequence(windows,batch_first=True),torch.tensor(lens,dtype=torch.long),
            torch.tensor(parents,dtype=torch.long),torch.tensor(weights,dtype=torch.float32),
            torch.tensor(starts,dtype=torch.long))


def split_candidate(x,lengths,candidate,generator=None):
    policy=candidate.get('prefix_policy')
    if not policy:return split_batch(x,lengths,candidate['window_size'],candidate.get('window_stride',0))
    windows=[];lens=[];parents=[];weights=[]
    for parent,length in enumerate(lengths.tolist()):
        if length<policy['min_points']:raise ValueError('有效单程短于最少前缀观察量')
        ends=list(range(policy['min_points'],length+1,policy['step']))
        if ends[-1]!=length:ends.append(length)
        if generator is not None and len(ends)>policy['samples_per_parent']:
            selected=sorted(torch.randperm(len(ends),generator=generator)[:policy['samples_per_parent']].tolist());ends=[ends[i] for i in selected]
        for end in ends:
            windows.append(x[parent,:end]);lens.append(end);parents.append(parent);weights.append(1/len(ends))
    return pad_sequence(windows,batch_first=True),torch.tensor(lens),torch.tensor(parents),torch.tensor(weights),torch.zeros(len(lens),dtype=torch.long)


def aggregate_log_probs(logits,parents,weights,count):
    logp=logits.float().log_softmax(-1)
    parents=parents.to(logits.device);weights=weights.to(logits.device)
    return torch.stack([torch.logsumexp(logp[parents==i]+weights[parents==i,None].log(),dim=0) for i in range(count)])


def candidate_label(candidate):
    spec=candidate['spec'];widths=spec['channels']
    width=str(widths[0]) if len(set(widths))==1 else '/'.join(map(str,widths))
    stride=f" · 步长{candidate['window_stride']}（重叠{1-candidate['window_stride']/candidate['window_size']:.0%}）" if 'window_stride' in candidate else ''
    prefix=f"CERA · GRU{spec['gru_hidden']}×{spec['gru_layers']}层 · " if spec['revision'] in CERA_REVISIONS else ''
    if candidate.get('prefix_policy'):return prefix+f"累积历史 · 最少{candidate['window_size']}点 · 间隔{candidate['prefix_policy']['step']}点 · 通道{width}"
    return prefix+f"{'完整单程' if not candidate['window_size'] else str(candidate['window_size'])+'点'} · 通道{width} · {len(widths)}块 · {POOLING_TITLES[spec['pooling']]}"+stride


def study_description(study):
    """Human explanations refer to the actual frozen candidates, not editor widgets."""
    mode=study['mode']
    if mode=='prefix':
        purpose='只训练累积前缀CERA；每次使用从有效扫描起点到当前的全部历史，每条原始单程总训练权重为1。'
        changes=[('累积历史','不同长度前缀监督','积累足够证据后提前输出并锁定')]
    elif mode=='overlap_pair':
        purpose='比较训练切窗是否重叠；两组共用窗口长度、TCN结构、采集组划分和训练参数，仅改变训练步长。在线效果另用同一冻结验证子集和相同在线门槛比较。'
        changes=[('不重叠基准','步长等于窗口长度','不重叠切窗的基准效果'),
                 ('重叠切窗','仅缩短训练步长','增加相邻窗口覆盖能否改善验证结果与在线识别')]
    elif mode=='window_pool':
        purpose='比较窗口长度与特征汇总方式；四组共用128通道、6块、卷积核5、膨胀1/2/4/8/16/32、Dropout 0.1。'
        changes=[('基准','基准配置','保留当前最好方案'),('波动信息','仅改变特征汇总','显式提供特征波动强度是否有帮助'),
                 ('短窗口','仅缩短窗口','更短的局部片段是否仍有辨识力'),('长窗口','仅加长窗口','更多上下文能否减少局部混淆')]
    elif mode=='four_way':
        purpose='比较网络通道宽度和是否切窗；完整单程/900点窗口与64/128通道交叉对照。'
        changes=[('基准','完整单程、64通道','完整单程基准'),('增加通道','相对A仅增至128通道','增加模型容量是否有效'),
                 ('切窗','相对A仅改900点窗口','切窗是否有效'),('切窗＋增加通道','相对B切窗、相对C增加通道','两种变化的组合效果')]
    else:
        purpose='自定义单组模型；用于检查一个具体配置。';changes=[('自定义','按当前配置','评估所选配置')]
    groups=[]
    for i,(candidate,(role,change,question)) in enumerate(zip(study['candidates'],changes)):
        groups.append(dict(group=chr(65+i),candidate_id=candidate['id'],role=role,change=change,question=question,
                           window=candidate['window_size'],pooling=candidate['spec']['pooling'],label=candidate_label(candidate)))
    return dict(version='study-description-v1',mode=mode,title=MODE_TITLES[mode],purpose=purpose,groups=groups)


def description_text(description):
    lines=[description['title'],description['purpose']]
    for row in description['groups']:
        lines.append(f"{row['group']}：{row['role']} · {row['label']}\n  改变：{row['change']}\n  问题：{row['question']}")
    return '\n\n'.join(lines)


def study_definition(cfg):
    if cfg.tcn_spec is None:return None
    return dict(revision=STUDY_REVISION,mode=cfg.tcn_mode,candidates=candidates(cfg))


def validate_study(study,cfg):
    if study!=study_definition(cfg):
        raise ValueError('TCN研究版本或冻结候选不一致')


def configuration_id(architecture,signal,readout,candidate=None):
    base=f'{architecture}_{signal}_{readout}'
    return base+'_'+candidate['id'] if candidate is not None else base


def window_metrics(rows,labels):
    from .evaluation import classification_metrics
    return classification_metrics([dict(true=row['true'],predicted=w['predicted'])
                                   for row in rows for w in row.get('windows',[])],labels)


def contrasts(meta):
    result={}
    lookup={(r['candidate']['id'],r['seed'],r['fold']):r['val_metrics']['macro_f1'] for r in meta['models'].values()}
    if meta.get('study',{}).get('mode')=='overlap_pair':
        ids=[c['id'] for c in meta['study']['candidates']]
        pairs=[('B-A',ids[1],ids[0])]
    elif meta.get('study',{}).get('mode')=='window_pool':
        ids=[c['id'] for c in meta['study']['candidates']]
        pairs=[(f'{chr(65+i)}-A',ids[i],ids[0]) for i in (1,2,3)]
    else:pairs=[('B-A','full_h128','full_h64'),('C-A','w900_h64','full_h64'),('D-B','w900_h128','full_h128'),('D-C','w900_h128','w900_h64')]
    for name,left,right in pairs:
        differences=[dict(seed=seed,fold=fold,difference=value-lookup[(right,seed,fold)])
                     for (identity,seed,fold),value in lookup.items() if identity==left and (right,seed,fold) in lookup]
        result[name]=dict(folds=differences,mean=sum(r['difference'] for r in differences)/len(differences) if differences else None)
    return result
