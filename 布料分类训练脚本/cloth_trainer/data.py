"""Immutable CSV sources, causal sample holding, fixed-horizon trials and grouped partitions."""
from dataclasses import dataclass,replace,asdict,field
from pathlib import Path
import hashlib
import csv
import json
import re
import numpy as np
from scipy.signal import firwin,lfilter
from .config import DataConfig

@dataclass
class Trial:
    id:str
    path:str
    sha256:str
    group:str
    label:str
    direction:str
    values:np.ndarray
    times:np.ndarray
    raw_t:np.ndarray
    raw_x:np.ndarray
    raw_values:np.ndarray
    bounds:tuple
    raw_valid:np.ndarray | None=None
    axes:tuple | None=None

@dataclass
class Record:
    path:str
    sha256:str
    metadata_sha256:str | None
    group:str
    label:str
    trials:list
    warnings:list
    ppmm:float
    speed:float
    axes:tuple=('x','y')
    quality:str='normal'
    quality_notes:list=field(default_factory=list)
    diagnostics:dict=field(default_factory=dict)
    label_confirmed:bool=True
    views:dict=field(default_factory=dict)
    audit:dict=field(default_factory=dict)

    def manifest(self):
        def info(t):return dict(id=t.id,direction=t.direction,points=len(t.values),bounds=t.bounds,axes=t.axes or self.axes)
        return dict(path=self.path,sha256=self.sha256,metadata_sha256=self.metadata_sha256,group=self.group,label=self.label,warnings=self.warnings,
                    ppmm=self.ppmm,speed=self.speed,axes=self.axes,quality=self.quality,quality_notes=self.quality_notes,diagnostics=self.diagnostics,audit=self.audit,label_confirmed=self.label_confirmed,sample_weight=1.,trials=[info(t) for t in self.trials],views={key:[info(t) for t in trials] for key,trials in self.views.items()})

class Unassessable(ValueError):pass

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def metadata_hash(csv_path):
    path=Path(csv_path).parent/'metadata.json'
    return file_hash(path) if path.exists() else None

IGNORED={'.workbench-trash','outputs','.venv','.git','__pycache__'}

def discover_report(paths):
    found=set();ignored=[];errors=[]
    for p in map(Path,paths):
        if p.is_dir():
            for f in p.rglob('*.csv'):
                target=f.resolve()
                if IGNORED & (set(f.parts)|set(target.parts)):ignored.append(dict(path=str(f),reason='回收站/输出/环境目录'))
                else:found.add(target)
        elif IGNORED & (set(p.parts)|set(p.resolve().parts)):
            errors.append(dict(path=str(p.resolve()),message='请将回收站/输出/环境中的CSV恢复到正常数据目录后导入'))
        else:found.add(p.resolve())
    return sorted(found),ignored,errors

def discover(paths):return discover_report(paths)[0]

def num(v):
    try:return float(v)
    except (TypeError,ValueError):return float('nan')

def required_axes(signals):
    axes={'by':{'y'},'bx':{'x'},'bxby':{'x','y'},'xy':{'x','y'}}
    if not signals or set(signals)-axes.keys():raise ValueError('请选择有效输入通道')
    needed=set().union(*(axes[s] for s in signals))
    return tuple(axis for axis in ('x','y') if axis in needed)


def contact_windows(path,meta,rows,cfg):
    bounds={'step_003':(cfg.out_low,cfg.out_high),'step_004':(cfg.back_low,cfg.back_high)}
    detail=dict(version=cfg.contact_rule,mode='fixed-x',sequence=None)
    if cfg.contact_rule=='fixed-x-v1':return bounds,detail
    match=re.search(r'[_\-](\d+)$',path.stem)
    named=int(match[1]) if match else None
    # A renamed, unlabeled recognition file is not evidence of a capture number.
    sequence=meta.get('sequence',named if meta.get('label') else None)
    if sequence is not None:
        if isinstance(sequence,bool) or not isinstance(sequence,int) or sequence<0:raise ValueError('采集序号无效')
        if named is not None and sequence!=named:raise ValueError('metadata采集序号与CSV文件名序号冲突')
    detail['sequence']=sequence
    if sequence is None or sequence<35:return bounds,detail
    outgoing,back=rows['step_003'],rows['step_004']
    if len(outgoing)<2 or len(back)<2:raise ValueError('缺少完整155mm往返，无法确定起点')
    start,end,turn,finish=outgoing[0][1],outgoing[-1][1],back[0][1],back[-1][1]
    tolerance=cfg.boundary_tolerance
    if not np.isfinite([start,end,turn,finish]).all() or any(abs(delta)>tolerance for delta in (end-start-155.,turn-start-155.,finish-start)):
        raise ValueError('未覆盖完整155mm往返，不能可靠确定每次去程起点')
    # Keep the measured motor coordinates intact. Both legs cross the same X window.
    bounds={stage:(start+44.,start+144.) for stage in bounds}
    detail.update(mode='start-relative',origin_x_mm=float(start),turnaround_x_mm=float(start+155.),out_travel_mm=[44.,144.],back_travel_mm=[11.,111.],absolute_x_mm=list(bounds['step_003']))
    return bounds,detail

def load_record(path,config=None,label=None,signals=None,_review=True):
    cfg=(config or DataConfig()).validate();axes=('x','y') if cfg.preprocess_version=='legacy-v2' or signals is None else required_axes(signals);p=Path(path).resolve()
    mp=p.parent/'metadata.json';meta_bytes=mp.read_bytes() if mp.exists() else None
    meta_digest=hashlib.sha256(meta_bytes).hexdigest() if meta_bytes is not None else None
    meta=json.loads(meta_bytes.decode('utf-8-sig')) if meta_bytes is not None else {}
    if not isinstance(meta,dict):raise ValueError('metadata必须是JSON对象')
    for key in ('context','baseline_B0_uT'):
        if meta.get(key) is not None and not isinstance(meta[key],dict):raise ValueError(f'metadata.{key}必须是JSON对象')
    ctx=meta.get('context') or {}
    for key in ('config','baseline'):
        if ctx.get(key) is not None and not isinstance(ctx[key],dict):raise ValueError(f'metadata.context.{key}必须是JSON对象')
    settings=ctx.get('config') or {};warnings=[]
    ppmm=num(settings.get('x_ppmm',cfg.fallback_ppmm));speed=num(settings.get('x_speed',cfg.fallback_speed))
    if not np.isfinite(ppmm) or ppmm<=0 or not np.isfinite(speed) or speed<=0:raise ValueError('位置换算或速度无效')
    if 'x_ppmm' not in settings:warnings.append(f'使用默认{ppmm:g} pulses/mm')
    if 'x_speed' not in settings:warnings.append(f'无速度元数据，使用明确配置的{speed:g} mm/s')
    explicit_label=label is not None
    inferred=meta.get('label','')
    candidate=re.sub(r'[_\-]\d+$','',p.stem)
    if cfg.preprocess_version=='abs-v3' and not explicit_label and inferred and candidate!=p.stem and inferred!=candidate:raise ValueError('标签与规范文件名冲突，请明确指定标签')
    if not isinstance(inferred,str) or not inferred.strip():
        inferred=re.sub(r'[_\-]\d+$','',p.stem)
        if inferred==p.stem:inferred=''
    label=(inferred if label is None else label).strip()
    digest=file_hash(p);group='capture:'+str(ctx['capture_session']) if ctx.get('capture_session') else 'csv:'+digest
    sample_quality=dict(version=cfg.sample_quality,axes=list(axes))
    with p.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=set(reader.fieldnames or [])
        if not {'stage','motor_x_pulses'}<=fields:raise ValueError('缺少stage或motor_x_pulses，不是原始磁采样CSV')
        tk=next((k for k in ['sensor_t_s','device_time_unwrapped_us'] if k in fields),None)
        if not tk:raise ValueError('缺少传感器时间戳')
        cols=[];by_verification=dict(source='not_loaded',checked_samples=0,unverifiable_samples=0,status='not_assessable')
        baseline=meta.get('baseline_B0_uT') or {};base_list=(ctx.get('baseline') or {}).get('baseline',[])
        checks=[];max_errors={}
        for axis in axes:
            i='xyz'.index(axis)
            delta,raw=f'dB{axis}_uT',f'B{axis}_uT'
            if axis=='y':by_verification['source']='delta_and_raw' if delta in fields and raw in fields else 'delta_only' if delta in fields else 'computed_from_raw'
            zero=num(baseline.get(raw,base_list[i] if len(base_list)>=3 else None))
            other_zero=num(base_list[i] if len(base_list)>=3 else None)
            if cfg.preprocess_version=='abs-v3' and np.isfinite(zero) and np.isfinite(other_zero) and abs(zero-other_zero)>cfg.baseline_tolerance:raise ValueError('两处悬空基线冲突：'+raw)
            if cfg.preprocess_version=='abs-v3' and not np.isfinite(zero):raise Unassessable(f'{delta}差分来源待确认：缺少可追溯悬空基线')
            if delta in fields:
                cols.append((delta,0.))
                if raw in fields and cfg.preprocess_version=='abs-v3':checks.append((delta,raw,zero))
            elif raw in fields:
                if not np.isfinite(zero):raise ValueError(f'{raw}缺少明确悬空基线')
                cols.append((raw,zero))
            else:raise Unassessable(f'缺少{delta}或{raw}')
        rows={'step_003':[],'step_004':[]};verification_rows=[]
        if cfg.sample_quality=='selected-axis-v1':
            from .sample_quality import selected_axis_rows
            reader=selected_axis_rows(reader,axes,sample_quality)
        for row in reader:
            if row.get('stage') in rows:
                t=num(row.get(tk))/(1e6 if tk.endswith('_us') else 1)
                valid=str(row.get('sample_valid','1')).strip().lower() not in ('0','false','no')
                if valid and 'y' in axes:verification_rows.append((row['stage'],num(row.get('motor_x_pulses'))/ppmm,num(row.get('By_uT')),num(row.get('dBy_uT'))))
                if valid:
                    for delta,raw,zero in checks:
                        dv,rv=num(row.get(delta)),num(row.get(raw))
                        if np.isfinite(dv) and np.isfinite(rv):max_errors[delta]=max(max_errors.get(delta,0.),abs(dv-(rv-zero)))
                rows[row['stage']].append((t,num(row.get('motor_x_pulses'))/ppmm,
                    *[num(row.get(k))-zero for k,zero in cols],str(row.get('device_clock_segment','0')),valid,str(row.get('motor_fault','NONE')),str(row.get('invalid_reason',''))))
    if max(max_errors.values(),default=0)>cfg.baseline_tolerance:raise ValueError('差分与悬空基线不一致：'+str(max_errors))
    windows,contact_detail=contact_windows(p,meta,rows,cfg)
    if contact_detail['sequence'] is None and cfg.contact_rule=='relative-035-v1':warnings.append('缺少采集序号，使用固定X接触窗；请确认记录所属协议')
    for stage,x,raw_by,delta_by in verification_rows:
        lo,hi=windows[stage]
        if not lo<=x<=hi:continue
        if by_verification['source']=='computed_from_raw':by_verification['checked_samples']+=int(np.isfinite(raw_by))
        elif np.isfinite(delta_by):
            by_verification['checked_samples']+=int(np.isfinite(raw_by));by_verification['unverifiable_samples']+=int(not np.isfinite(raw_by))
    master_axes=axes;master_rows=rows;stage_checks={}
    def resample_view(view_axes):
        axes=tuple(view_axes);width=2+len(master_axes);warnings=[]
        rows={stage:[(r[0],r[1],*[r[2+master_axes.index(axis)] for axis in axes],*r[width:]) for r in rr] for stage,rr in master_rows.items()}
        trials=[];invalid_counts={}
        for stage,direction in [('step_003','out'),('step_004','back')]:
            lo,hi=windows[stage]
            rr=rows[stage]
            if len(rr)<2:raise ValueError(f'{stage}缺少完整单程')
            width=2+len(axes);a=np.array([r[:width] for r in rr]);clocks=np.array([r[width] for r in rr]);valid=np.array([r[width+1] for r in rr])&np.isfinite(a).all(axis=1)
            inside=(a[:,1]>=lo)&(a[:,1]<=hi);ix=np.flatnonzero(inside)
            if len(ix)<2:raise ValueError(f'{stage}没有{lo:g}–{hi:g}mm接触数据')
            if not inside[ix[0]:ix[-1]+1].all():raise ValueError(f'{stage}轨迹离开后重新进入有效范围')
            if cfg.preprocess_version=='abs-v3':
                faults=[rr[j][width+2] for j in ix if rr[j][width+2].strip().lower() not in ('','none','0','false','no')]
                if faults:raise ValueError(f'{stage}接触窗内有明确电机故障：{faults[0]}')
            tt=a[ix,0]
            if not np.isfinite(tt).all() or (np.diff(tt)<=0).any() or len(set(clocks[ix]))!=1:raise ValueError(f'{stage}时间倒退或时钟重置')
            from collections import Counter
            invalid_counts[stage]=dict(Counter(rr[j][width+3] or ('采集器标记无效' if not rr[j][width+1] else '所选通道或时间/位置非有限') for j in ix if not valid[j]))
            good=ix[valid[ix]]
            if len(good)<2:raise ValueError(f'{stage}没有有效磁数据')
            t,x,y=a[good,0],a[good,1],a[good,2:]
            if x.min()>lo+cfg.boundary_tolerance or x.max()<hi-cfg.boundary_tolerance:raise ValueError(f'{stage}未覆盖完整裁剪范围')
            actual=(x[-1]-x[0])/(t[-1]-t[0])
            if axes==master_axes:stage_checks[stage]=dict(contact_mm=[lo,hi],observed_mm=[float(x.min()),float(x.max())],valid_samples=len(good),invalid_samples=int((~valid[ix]).sum()),max_gap_s=float(np.diff(t).max()),observed_speed_mm_s=float(actual),observed_direction='out' if actual>0 else 'back',clock_segments=len(set(clocks[ix])),motor_fault='none_recorded',invalid_reasons=invalid_counts[stage])
            if (actual>0)!=(direction=='out') or abs(abs(actual)/speed-1)>cfg.speed_tolerance:raise ValueError(f'{stage}方向或扫描速度不符合固定协议')
            if np.diff(t).max()>cfg.max_gap+1e-9:raise ValueError(f'{stage}缺口超过{cfg.max_gap*1000:g}ms')
            duration=(hi-lo)/speed-cfg.tail_margin
            n=int(np.floor(duration*cfg.fs))
            if n<10:raise ValueError('有效扫描时长过短，请检查固定速度和裁剪范围')
            grid=t[0]+np.arange(n)/cfg.fs
            if grid[-1]>(t[-1] if cfg.preprocess_version=='abs-v3' else a[ix[-1],0])+1e-9:raise ValueError(f'{stage}不足预先固定的完整扫描时长，网格超过最后有效观测')
            measured_native=1/np.median(np.diff(tt if cfg.preprocess_version=='abs-v3' else t));native=cfg.source_fs
            if axes==master_axes:stage_checks[stage]['observed_fs']=float(measured_native)
            if abs(measured_native/native-1)>.10:raise ValueError(f'{stage}原始采样率不符合固定协议，请明确配置source_fs')
            delay=0.
            if native>cfg.fs*1.05:
                # A causal FIR; every output depends only on already received samples.
                taps=firwin(63,.45*cfg.fs,fs=native)
                if cfg.preprocess_version=='abs-v3':
                    native_grid=t[0]+np.arange(int(np.floor((t[-1]-t[0])*native))+1)/native
                    previous=np.searchsorted(t,native_grid,side='right')-1
                    if np.max(native_grid-t[previous])>cfg.max_gap+1e-9:raise ValueError(f'{stage}原始时基缺口超过允许保持时间')
                    y=y[previous];t=native_grid
                y=lfilter(taps,[1.],y,axis=0);delay=31/native
                warnings.append(f'{stage}因果抗混叠FIR延迟{delay:.6f}s')
            positions=np.searchsorted(t,grid,side='right')-1
            if positions.min()<0 or np.max(grid-t[positions])>cfg.max_gap+1e-9:raise ValueError(f'{stage}时基缺口超过允许保持时间')
            values=y[positions].astype(np.float32)
            if cfg.preprocess_version=='abs-v3':values=np.abs(values)
            if (~valid[ix]).any():warnings.append(f'{stage}剔除{int((~valid[ix]).sum())}个无效采样；只以过去有效值保持')
            trials.append(Trial(digest+':'+direction,str(p),digest,group,label,direction,values,grid,
                                a[:,0],a[:,1],a[:,2:],(lo,hi),raw_valid=valid,axes=axes))
        return trials,warnings,invalid_counts
    trials,view_warnings,invalid_counts=resample_view(axes);warnings.extend(view_warnings);views={}
    if cfg.preprocess_version=='abs-v3' and len(axes)==2:
        for signal,view_axes in [('by',('y',)),('bx',('x',))]:
            if cfg.sample_quality=='selected-axis-v1':
                # Each view must have the same quality mask as importing that
                # channel alone; another selected input cannot remove its points.
                view=load_record(p,cfg,label,signals=(signal,),_review=False)
                views[signal]=view.trials;warnings.extend(view.warnings)
                if signal=='by':by_verification=view.diagnostics['by_verification']
            else:
                vv,ww,ii=resample_view(view_axes);views[signal]=vv;warnings.extend(ww)
    warnings=list(dict.fromkeys(warnings))
    if not label:warnings.append('未指定标签，训练前需指定；识别允许无标签')
    if file_hash(p)!=digest or metadata_hash(p)!=meta_digest:raise ValueError('CSV或元数据在导入期间发生改变，请重新导入')
    sigma=(ctx.get('baseline') or {}).get('sigma')
    sigma=[num(v) if np.isfinite(num(v)) else None for v in sigma] if isinstance(sigma,(list,tuple)) else None
    record=Record(str(p),digest,meta_digest,group,label,trials,warnings,ppmm,speed,axes=axes,views=views,label_confirmed=explicit_label or bool(meta.get('label')),diagnostics=dict(baseline_max_error=max_errors,invalid_reasons=invalid_counts,baseline_sigma=sigma,preprocess_version=cfg.preprocess_version))
    if by_verification['checked_samples'] and not by_verification['unverifiable_samples']:by_verification['status']='verified'
    record.diagnostics['by_verification']=by_verification
    if cfg.sample_quality=='selected-axis-v1':record.diagnostics['sample_quality']=sample_quality
    if cfg.contact_rule!='fixed-x-v1':record.diagnostics['contact_rule']=contact_detail
    record.diagnostics['integrity']=dict(limits=dict(max_gap_s=cfg.max_gap,baseline_tolerance_uT=cfg.baseline_tolerance,boundary_tolerance_mm=cfg.boundary_tolerance,speed_tolerance=cfg.speed_tolerance,fs=cfg.fs,source_fs=cfg.source_fs,tail_margin_s=cfg.tail_margin),stages=stage_checks,baseline_max_error_uT=max_errors,sources_unchanged=True,unverifiable=['仅凭采样与软件位置无法证明未发生撞击、离开接触面或机械滑动'])
    record.diagnostics['review_clues']=[f'采集状态需复核：{meta.get("outcome")} {meta.get("error") or ""}'] if meta.get('error') or meta.get('outcome','completed')!='completed' else []
    if cfg.preprocess_version=='abs-v3' and _review:
        if 'y' not in axes:
            try:
                by_record=load_record(p,cfg,label,signals=('by',),_review=False);record.views['by']=by_record.trials;record.diagnostics['by_verification']=by_record.diagnostics['by_verification']
            except (ValueError,OSError) as exc:record.diagnostics['review_clues'].append('By审查依据不足：'+str(exc))
        from .quality import annotate
        annotate(record)
    return record

def protocol_for(records,config):
    if not records:raise ValueError('没有可用记录')
    speed=records[0].speed;points={t.direction:len(t.values) for t in records[0].trials}
    variable=config.contact_rule=='relative-035-v1'
    profiles=[]
    for record in records:
        lengths={t.direction:len(t.values) for t in record.trials}
        if record.speed!=speed or (not variable and lengths!=points):
            raise ValueError('正式协议要求固定扫描速度和各方向时长，请分别训练不同扫描协议')
        if variable:
            expected={t.direction:int(np.floor(((t.bounds[1]-t.bounds[0])/speed-config.tail_margin)*config.fs)) for t in record.trials}
            if lengths!=expected or record.diagnostics.get('contact_rule',{}).get('version')!=config.contact_rule:raise ValueError('片段接触段规则或时长不匹配，请重新导入CSV')
            if lengths not in profiles:profiles.append(lengths)
    labels=sorted({r.label for r in records})
    data=asdict(config)
    if config.sample_quality=='recorded-v1':data.pop('sample_quality')
    if not variable:data.pop('contact_rule')  # Preserve serialized protocols of old models/snapshots.
    result=dict(speed=speed,trial_points=points,data=data,class_weights={c:1/len(labels) for c in labels})
    if variable:result['accepted_trial_points']=sorted(profiles,key=lambda p:(p['out'],p['back']))
    return result

def matches_protocol(record,protocol):
    return record.speed==protocol['speed'] and {t.direction:len(t.values) for t in record.trials} in protocol.get('accepted_trial_points',[protocol['trial_points']])

def import_dataset(paths,config=None,signals=None,overrides=None,progress=None):
    records=[];duplicates=[];files,ignored,errors=discover_report(paths);seen={};conflicts=set()
    candidate_paths=list(files)+[Path(e['path']) for e in errors]
    for i,p in enumerate(files):
        try:
            r=load_record(p,config,(overrides or {}).get(str(p)),signals=signals)
            if r.sha256 in seen:
                old=seen[r.sha256]
                if old.label!=r.label or old.metadata_sha256!=r.metadata_sha256:
                    conflicts.add(r.sha256);errors.extend(dict(path=x.path,message='重复内容的标签或元数据冲突，请明确正确来源') for x in (old,r))
                else:duplicates.append(dict(path=str(p),canonical=old.path,sha256=r.sha256))
            else:seen[r.sha256]=r;records.append(r)
        except (ValueError,OSError,KeyError,TypeError) as exc:
            from .quality import RULE_VERSION,current_rule
            try:sha,md=file_hash(p),metadata_hash(p)
            except OSError:sha,md=None,None
            errors.append(dict(path=str(p),message=str(exc),sha256=sha,metadata_sha256=md,audit=dict(rule_version=RULE_VERSION,rule_sha256=current_rule()['sha256'],status='undetermined' if isinstance(exc,Unassessable) else 'data_error',eligible=False,reasons=[str(exc)])))
        if progress:progress(i+1,len(files))
    records=[r for r in records if r.sha256 not in conflicts]
    report=dict(candidate_count=len(candidate_paths),usable_count=len(records),ignored_count=len(ignored),ignored=ignored,duplicates=duplicates,errors=errors,
                suggest_recollect=sum(r.quality=='suggest_recollect' for r in records),suggest_review=sum(r.quality=='suggest_review' for r in records))
    if (config or DataConfig()).preprocess_version=='abs-v3':
        from .admission import review_batch
        batch=review_batch(records,errors,candidate_paths,config or DataConfig(),signals or ('by','bxby'),overrides=overrides)
        report.update(review_batches=[batch],eligible_count=sum(r.audit['eligible'] for r in records),excluded_count=sum(r.quality=='excluded' for r in records),
                      undetermined_count=sum(r.quality=='undetermined' for r in records)+sum(e['audit']['status']=='undetermined' for e in errors),suggest_review=sum(r.quality=='suggest_review' for r in records))
    return records,report

def import_records(paths,config=None,overrides=None,progress=None,signals=None):
    records,report=import_dataset(paths,config,signals,overrides,progress)
    return records,report['errors']+[dict(path=d['path'],message='重复CSV内容，未重复加入') for d in report['duplicates']]

def prepare_trials(records,signal):
    result=[]
    needed=set(required_axes((signal,)))
    for r in records:
        if not needed.issubset(r.axes):raise ValueError('所选输入缺少必要通道：'+str(sorted(needed-set(r.axes))))
        for t in r.views.get(signal,r.trials):
            axes=t.axes or r.axes
            values=t.values
            if signal=='by':values=values[:,axes.index('y'):axes.index('y')+1]
            elif signal=='bx':values=values[:,axes.index('x'):axes.index('x')+1]
            elif signal=='xy':values=np.linalg.norm(values,axis=1,keepdims=True)
            elif signal!='bxby':raise ValueError('未知输入信号')
            if t.direction=='back':values=np.abs(values)
            prepared=replace(t,label=r.label,values=np.ascontiguousarray(values,dtype=np.float32));prepared.quality=r.quality;result.append(prepared)
    return result

def split_records(records,seed=42):
    labels=sorted({r.label for r in records})
    if '' in labels or len(labels)<2:raise ValueError('至少选择2类布料，所有文件都需标签')
    groups={}
    for r in records:groups.setdefault(r.group,set()).add(r.label)
    counts={c:sum(c in v for v in groups.values()) for c in labels}
    if min(counts.values())<10:raise ValueError('每类需至少10个独立CSV／采集组；不足：'+ '，'.join(f'{c} {n}组' for c,n in counts.items() if n<10))
    rng=np.random.default_rng(seed);keys=['train','val','test'];ratios=np.array([.7,.1,.2])
    allocations={k:set() for k in keys}
    def quota(n):
        q=np.array([7,1,2]);extra=n-10
        while extra:
            target=ratios*n-q;q[np.argmax(target)]+=1;extra-=1
        return q
    desired=np.array([quota(counts[c]) for c in labels])
    if all(len(v)==1 for v in groups.values()):
        for ci,c in enumerate(labels):
            gs=sorted(g for g,v in groups.items() if c in v);rng.shuffle(gs);start=0
            for k,n in zip(keys,desired[ci]):allocations[k].update(gs[start:start+n]);start+=n
    else:
        # Minimise deviation from per-class group quotas while preserving whole
        # capture sessions and at least one independent group/class/partition.
        from scipy.optimize import milp,Bounds,LinearConstraint
        gs=sorted(groups);n=len(gs);nc=len(labels);nv=3*n+6*nc
        matrix=[];lower=[];upper=[]
        for gi in range(n):
            row=np.zeros(nv);row[3*gi:3*gi+3]=1;matrix.append(row);lower.append(1);upper.append(1)
        for ci,c in enumerate(labels):
            for j in range(3):
                row=np.zeros(nv)
                for gi,g in enumerate(gs):row[3*gi+j]=int(c in groups[g])
                matrix.append(row.copy());lower.append(1);upper.append(np.inf)
                row[3*n+6*ci+2*j]=-1;row[3*n+6*ci+2*j+1]=1
                matrix.append(row);lower.append(desired[ci,j]);upper.append(desired[ci,j])
        cost=np.r_[rng.uniform(0,1e-6,3*n),np.ones(6*nc)]
        upper_bounds=np.r_[np.ones(3*n),np.full(6*nc,np.inf)]
        result=milp(cost,integrality=np.r_[np.ones(3*n),np.zeros(6*nc)],
                    bounds=Bounds(np.zeros(nv),upper_bounds),constraints=LinearConstraint(np.array(matrix),lower,upper),options={'time_limit':5})
        if result.x is None:raise ValueError('采集组结构无法形成覆盖全部类别的训练/验证/测试分区；不能拆分同组')
        choices=result.x[:3*n].reshape(n,3).round()
        if not np.all(choices.sum(1)==1):raise ValueError('无法取得可用的完整采集组分配')
        allocations={k:{g for gi,g in enumerate(gs) if choices[gi,j]==1} for j,k in enumerate(keys)}

    return {k:[i for i,r in enumerate(records) if r.group in allocations[k]] for k in keys}
