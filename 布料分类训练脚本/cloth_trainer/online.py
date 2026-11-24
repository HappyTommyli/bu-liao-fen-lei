"""Prefix-only, frozen-model online inference shared by CSV replay and TCP input.

No raw recording is rewritten. Labels are used by result evaluation only.
"""
from collections import deque, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import math
import socket
import hashlib
import os
import numpy as np
from .config import DataConfig


@dataclass(frozen=True)
class OnlineConfig:
    window:int=900
    stride:int=900
    probability:float=.95
    margin:float=.20
    stable_seconds:float=2.7
    smoothing:bool=False
    smoothing_alpha:float=.3
    include_return:bool=False
    consecutive:int=1
    lock_on_confirm:bool=False

    def __post_init__(self):
        if type(self.window) is not int or type(self.stride) is not int or self.window<1 or not 1<=self.stride<=self.window:raise ValueError('窗口/预测步长无效')
        if not all(math.isfinite(v) for v in (self.probability,self.margin,self.stable_seconds,self.smoothing_alpha)):raise ValueError('门槛必须为有限数')
        if not 0<=self.probability<=1 or not 0<=self.margin<=1 or self.stable_seconds<0:raise ValueError('置信门槛无效')
        if not 0<self.smoothing_alpha<=1:raise ValueError('平滑参数无效')
        if type(self.consecutive) is not int or not 1<=self.consecutive<=20 or type(self.lock_on_confirm) is not bool:raise ValueError('连续确认次数或锁定设置无效')


def decode_packet(line):
    try:p=json.loads(line)
    except (ValueError,UnicodeDecodeError) as e:raise ValueError('无效NDJSON数据') from e
    if not isinstance(p,dict) or p.get('type') not in ('sample','reset','end'):raise ValueError('不支持的数据包类型')
    if p['type']=='sample' and not {'capture_id','stage','t_s','x_mm','by_uT','baseline_by_uT'}<=p.keys():raise ValueError('采样包字段不完整')
    return p


class OnlinePredictor:
    def __init__(self,bundle=None,*,labels=None,predict_proba=None,config=None,data=None):
        self.bundle=bundle or {};self.labels=list(labels or self.bundle.get('labels',[]))
        self.config=config or OnlineConfig()
        self.prefix=bool((self.bundle.get('candidate') or {}).get('prefix_policy'))
        if self.prefix and not self.config.lock_on_confirm:raise ValueError('累积前缀模型须启用达标锁定')
        self.data=data or DataConfig.from_dict(self.bundle.get('config',{}).get('data',asdict(DataConfig())))
        if bundle and (bundle.get('signal','by')!='by' or bundle.get('architecture','tcn') not in ('tcn','cera') or self.data.preprocess_version!='abs-v3'):raise ValueError('在线识别要求TCN/CERA / |dBy|冻结模型')
        if self.data.contact_rule!='fixed-x-v1':raise ValueError('在线识别当前仅支持冻结固定坐标协议')
        if bundle:
            candidate=bundle.get('candidate') or {}
            if bundle.get('architecture')=='cera' and candidate.get('spec',{}).get('revision') not in ('configurable-cera-v1','configurable-cera-prefix-v1'):raise ValueError('旧CERA不含窗口训练定义，不能用于当前在线预测')
            trained_window=candidate.get('window_size',bundle.get('config',{}).get('tcn_window'))
            if trained_window is None and 'architecture' in bundle:raise ValueError('模型没有冻结窗口定义，不能作为在线窗口模型')
            if trained_window is not None and trained_window!=self.config.window:raise ValueError('在线窗口必须匹配模型训练窗口，整段模型不支持任意在线窗口')
        self.predict_proba=predict_proba or self._torch_probability
        self.buffer=deque(maxlen=None if self.prefix else self.config.window);self.identity=None;self.last_t=None;self.baseline=None
        self.history=[];self.reset()

    def reset(self):
        self.buffer.clear();self.pending=None;self.left=None;self.previous=None;self.next_grid=None
        self.sample_count=0;self.candidate=None;self.candidate_since=None;self.confirmed_label=None
        self.first_confirmation=None;self.flip_count=0;self.smoothed=None;self.quality_resets=0;self.last_prediction=None;self.in_crop=False
        self.consecutive_count=0;self.locked=False;self.completion=None;self.effective_start_t=None;self._waiting_state=None

    def _clear_segment(self):
        self.buffer.clear();self.pending=None;self.left=None;self.previous=None;self.next_grid=None
        self.sample_count=0;self.candidate=None;self.candidate_since=None;self.confirmed_label=None;self.smoothed=None;self.last_prediction=None
        self.consecutive_count=0;self.effective_start_t=None

    def _error(self,reason,t=None):
        self._clear_segment();self.quality_resets+=1;self._waiting_state=None
        return dict(type='quality',status='error',reason=reason,t_s=t,capture_id=(self.identity or ('',''))[0])

    def _torch_probability(self,values):
        import torch
        model=self.bundle['_model'];model.eval()
        mean=np.asarray(self.bundle['scaler']['mean']);std=np.asarray(self.bundle['scaler']['std'])
        if not np.isfinite(std).all() or np.any(std<=0):raise ValueError('冻结标准化参数无效')
        x=torch.from_numpy(((values-mean)/std).astype(np.float32))[None]
        device=next(model.parameters()).device
        with torch.inference_mode():return model(x.to(device),torch.tensor([len(values)],dtype=torch.long)).float().softmax(-1)[0].cpu().numpy()

    def _grid_sample(self,value,t,x,available):
        if self.locked:return []
        if self.prefix and self.sample_count>=120000:raise ValueError('累积历史超过安全上限，请检查扫描时长或位置数据')
        self.buffer.append(value);self.sample_count+=1
        cfg=self.config
        if self.sample_count<cfg.window or (self.sample_count-cfg.window)%cfg.stride:return []
        p=np.asarray(self.predict_proba(np.asarray(self.buffer,dtype=np.float32).reshape(-1,1)),dtype=float)
        if p.shape!=(len(self.labels),) or not np.isfinite(p).all() or np.any(p<0) or p.sum()<=0:raise ValueError('模型概率输出无效')
        p=p/p.sum();raw=p.copy()
        if cfg.smoothing:
            self.smoothed=p if self.smoothed is None else cfg.smoothing_alpha*p+(1-cfg.smoothing_alpha)*self.smoothed;p=self.smoothed
        order=np.argsort(p);index=int(order[-1]);best=float(p[index]);margin=best-(float(p[order[-2]]) if len(p)>1 else 0.)
        label=self.labels[index];eligible=best>=cfg.probability and margin>=cfg.margin
        if eligible:
            if self.candidate!=label:self.candidate=label;self.candidate_since=available;self.consecutive_count=1
            else:self.consecutive_count+=1
        else:self.candidate=None;self.candidate_since=None;self.consecutive_count=0
        stable=eligible and self.consecutive_count>=cfg.consecutive and available-self.candidate_since+1e-10>=cfg.stable_seconds
        if stable:
            if self.confirmed_label is not None and self.confirmed_label!=label:self.flip_count+=1
            self.confirmed_label=label
            if self.first_confirmation is None:self.first_confirmation=dict(t_s=available,x_mm=x,label=label)
            if cfg.lock_on_confirm:
                self.locked=True;self.completion=dict(self.first_confirmation,probability=best,points=self.sample_count,effective_elapsed_s=available-self.effective_start_t if self.effective_start_t is not None else None)
        reason='probability_below_threshold' if best<cfg.probability else 'margin_below_threshold' if margin<cfg.margin else 'awaiting_consecutive' if self.consecutive_count<cfg.consecutive else 'awaiting_stability'
        event=dict(type='prediction',status='locked' if self.locked else 'confirmed' if stable else 'judging',t_s=available,sample_end_t_s=t,
                   decision_delay_s=max(0.,available-t),x_mm=x,prediction=label,probability=best,margin=margin,
                   probabilities=p.tolist(),raw_probabilities=raw.tolist(),confirmed_label=self.confirmed_label,
                   stable_elapsed_s=0. if self.candidate_since is None else available-self.candidate_since,
                   sample_count=self.sample_count,flip_count=self.flip_count,capture_id=self.identity[0],stage=self.identity[1],consecutive_count=self.consecutive_count,reason='completed' if stable else reason,effective_elapsed_s=available-self.effective_start_t if self.effective_start_t is not None else None,completion=self.completion)
        self.last_prediction=event;self.history.append(event);return [event]

    def _accept(self,sample,available):
        t,x,value=sample
        if self.effective_start_t is None:self.effective_start_t=t
        if self.next_grid is None:self.next_grid=t
        result=[]
        # Previous-sample holding; never interpolate from a frame that has not arrived.
        if self.previous is not None:
            while self.next_grid<t-1e-9:
                result+=self._grid_sample(self.previous[2],self.next_grid,self.previous[1],available);self.next_grid+=1/self.data.fs
        if self.next_grid<=t+1e-9:
            result+=self._grid_sample(value,self.next_grid,x,available);self.next_grid+=1/self.data.fs
        self.previous=sample;return result

    def feed(self,p):
        kind=p.get('type')
        if kind=='reset':
            self.reset();self.identity=None;self.last_t=None;self.baseline=None
            return [dict(type='reset',status='judging',reason=p.get('reason','reset'))]
        if kind=='end':
            # The final pending sample is deliberately not extrapolated/quality-certified.
            stable=self.locked or self.last_prediction is not None and self.last_prediction['status']=='confirmed'
            return [dict(type='end',status='confirmed' if stable else 'end-unconfirmed',
                         prediction=self.confirmed_label if stable else None,first_confirmation=self.first_confirmation,
                         flip_count=self.flip_count,quality_resets=self.quality_resets,sample_count=self.sample_count,
                         warmup_points=max(0,self.config.window-self.sample_count),capture_id=(self.identity or ('',''))[0],completion=self.completion,locked=self.locked)]
        if kind!='sample':raise ValueError('不支持的数据包类型')
        events=[];identity=(str(p.get('capture_id','')),p.get('stage'))
        if identity!=self.identity:
            self.reset();self.last_t=None;self.baseline=None;self.identity=identity
        if self.locked:return []
        try:t=float(p['t_s']);x=float(p['x_mm']);by=float(p['by_uT']);base=float(p['baseline_by_uT'])
        except (ValueError,TypeError,KeyError):return [self._error('missing_baseline_or_value')]
        fault=p.get('motor_fault',False)
        reason=p.get('invalid_reason','')
        # Forwarding also carries idle/preparation frames. These are expected
        # waiting states, never usable model input and not a signal-quality fault.
        if (reason in ('not_recording','stage_not_classifiable')
                and (reason=='not_recording' or identity[1] not in ('step_003','step_004'))
                and fault in (False,None,'','NONE','0',0)
                and all(math.isfinite(v) for v in (t,x,by,base))):
            waiting=(identity,reason);previous_waiting=self._waiting_state
            self._clear_segment();self.in_crop=False;self.last_t=t;self.baseline=None;self._waiting_state=waiting
            return [] if previous_waiting==waiting else [dict(type='waiting',status='waiting',reason=reason,stage=identity[1],capture_id=identity[0],t_s=t)]
        self._waiting_state=None
        if not all(math.isfinite(v) for v in (t,x,by,base)) or (not p.get('valid',True) and p.get('invalid_reason')!='isolated_spike') or fault not in (False,None,'','NONE','0',0):
            self.last_t=t;return [self._error(p.get('invalid_reason') or 'invalid_sample_or_motor_fault',t)]
        if self.last_t is not None and (t<=self.last_t or t-self.last_t>self.data.max_gap+1e-12):
            self.last_t=t;return [self._error('time_reversal_or_gap',t)]
        self.last_t=t
        if self.baseline is not None and abs(base-self.baseline)>self.data.baseline_tolerance:
            self.baseline=base;return [self._error('baseline_changed',t)]
        self.baseline=base
        bounds={'step_003':(self.data.out_low,self.data.out_high),'step_004':(self.data.back_low,self.data.back_high)}
        if identity[1] not in bounds or not bounds[identity[1]][0]<=x<=bounds[identity[1]][1]:
            if self.in_crop:
                self.buffer.clear();self.pending=None;self.left=None;self.previous=None;self.next_grid=None
            self.in_crop=False;return []
        if not self.in_crop:
            self._clear_segment();self.in_crop=True
        sample=(t,x,by-base)
        if self.pending is not None:
            if self.left is not None and abs(self.pending[2]-self.left[2])>=1000 and abs(self.pending[2]-sample[2])>=1000 and abs(sample[2]-self.left[2])<=250:
                return [self._error('selected_axis_isolated_spike',t)]
            events+=self._accept((self.pending[0],self.pending[1],abs(self.pending[2])),t)
            self.left=self.pending
        self.pending=sample
        return events


def csv_metadata(path):
    path=Path(path);mp=path.parent/'metadata.json'
    meta=json.loads(mp.read_text(encoding='utf-8-sig')) if mp.exists() else {}
    baseline=meta.get('baseline_B0_uT') or {};ctx=meta.get('context') or {};values=(ctx.get('baseline') or {}).get('baseline',[])
    base=baseline.get('By_uT',values[1] if len(values)>1 else None)
    other=values[1] if len(values)>1 else None
    if base is None or not math.isfinite(float(base)):raise ValueError('缺少固定可追溯By悬空基线')
    if other is not None and abs(float(base)-float(other))>.01:raise ValueError('两处By悬空基线冲突')
    return meta,float(base),float((ctx.get('config') or {}).get('x_ppmm',250.))


def csv_packets(path,cancelled=lambda:False):
    """Read metadata once, then rows in file order; never load_record or full-record quality."""
    path=Path(path);meta,base,ppmm=csv_metadata(path)
    if not math.isfinite(ppmm) or ppmm<=0:raise ValueError('无效位置换算')
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=set(reader.fieldnames or [])
        tk='sensor_t_s' if 'sensor_t_s' in fields else 'device_time_unwrapped_us'
        if not {'stage','motor_x_pulses',tk}<=fields or not {'By_uT','dBy_uT'}&fields:raise ValueError('缺少原始CSV字段')
        for row in reader:
            if cancelled():return
            try:
                by=float(row['By_uT']) if 'By_uT' in fields else float(row['dBy_uT'])+base
                valid=str(row.get('sample_valid','1')).lower() not in ('0','false','no') or row.get('invalid_reason')=='isolated_spike'
                reason=row.get('invalid_reason','')
                if 'By_uT' in fields and 'dBy_uT' in fields:
                    delta=float(row['dBy_uT'])
                    if not math.isfinite(delta) or not math.isfinite(by) or abs(delta-(by-base))>.01:
                        valid=False;reason='by_baseline_mismatch'
                yield dict(type='sample',capture_id=str(path.resolve()),stage=row['stage'],
                           t_s=float(row[tk])/(1e6 if tk.endswith('_us') else 1),x_mm=float(row['motor_x_pulses'])/ppmm,
                           by_uT=by,baseline_by_uT=base,valid=valid,invalid_reason=reason,
                           motor_fault=row.get('motor_fault','NONE') not in ('NONE','','0'))
            except (ValueError,TypeError,KeyError):
                yield dict(type='reset',capture_id=str(path),reason='malformed_csv_row')
    yield dict(type='end',capture_id=str(path))


def tcp_packets(host='127.0.0.1',port=8766,cancelled=lambda:False):
    if host!='127.0.0.1':raise ValueError('仅允许本机127.0.0.1')
    with socket.create_connection((host,int(port)),timeout=2.) as connection:
        connection.settimeout(.2);buffer=b''
        while not cancelled():
            try:chunk=connection.recv(65536)
            except socket.timeout:continue
            if not chunk:break
            buffer+=chunk
            if len(buffer)>1048576:raise ValueError('数据包过长')
            while b'\n' in buffer:
                line,buffer=buffer.split(b'\n',1)
                if line.strip():yield decode_packet(line)
        if buffer.strip():raise ValueError('连接结束时存在不完整数据包')
    yield dict(type='end',reason='disconnect')


def summarize_records(records):
    counts=Counter(r.get('status','abstain') for r in records);classes={}
    for label in sorted({r.get('true_label') for r in records if r.get('true_label')}):
        actual=[r for r in records if r.get('true_label')==label];tp=sum(r.get('prediction')==label for r in actual)
        predicted=sum(r.get('prediction')==label for r in records);precision=tp/predicted if predicted else 0.;recall=tp/len(actual)
        classes[label]=dict(count=len(actual),precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0.)
    first=Counter()
    for r in records:
        decision=(r.get('first_confirmation') or {}).get('label')
        first['abstain' if decision is None else 'unlabelled' if not r.get('true_label') else 'correct' if decision==r['true_label'] else 'error']+=1
    return dict(counts={k:counts[k] for k in ('correct','error','abstain')},unlabelled=counts['unlabelled'],first_confirmation_counts={k:first[k] for k in ('correct','error','abstain','unlabelled')},classes=classes,records=len(records))


def _model_sha256(path,cancelled=None):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            if cancelled and cancelled():raise ValueError('策略操作已取消')
            block=stream.read(1024*1024)
            if not block:break
            digest.update(block)
    return digest.hexdigest()


def freeze_policy(model_path,config,cancelled=None):
    policy=dict(format='cloth-online-policy-v1',model_sha256=_model_sha256(model_path,cancelled),settings=asdict(config))
    policy['fingerprint']=hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return policy


def validate_policy(policy,model_path,cancelled=None):
    if policy.get('format')!='cloth-online-policy-v1':raise ValueError('未知在线策略格式')
    config=OnlineConfig(**policy['settings'])
    base={k:v for k,v in policy.items() if k!='fingerprint'}
    fingerprint=hashlib.sha256(json.dumps(base,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if set(policy)!={'format','settings','model_sha256','fingerprint'} or policy['model_sha256']!=_model_sha256(model_path,cancelled) or policy['fingerprint']!=fingerprint:raise ValueError('冻结在线策略或模型权重已改变')
    return config


def claim_holdout_policy(model_path,policy,bundle):
    """Bind the first heldout policy before opening any raw heldout input.

    Exclusive creation never replaces an existing lock. Interrupted or malformed
    locks fail closed; a retry must match weights, policy and frozen cohort.
    """
    validate_policy(policy,model_path)
    try:
        cohort=[dict(index=i,sha256=bundle['sources'][i]['sha256'],
                     metadata_sha256=bundle['sources'][i].get('metadata_sha256'))
                for i in bundle['split_audit']['test']]
        if not cohort:raise ValueError('固定留出集为空')
        encoded=json.dumps(cohort,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    except (KeyError,TypeError,IndexError) as e:raise ValueError('冻结留出来源清单无效') from e
    expected=dict(format='cloth-online-holdout-lock-v1',model_sha256=policy['model_sha256'],
                  policy_fingerprint=policy['fingerprint'],cohort_sha256=hashlib.sha256(encoded).hexdigest(),sources=cohort)
    lock=Path(model_path).parent/'online_holdout_lock.json'
    try:fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:
        try:existing=json.loads(lock.read_text(encoding='utf-8'))
        except (OSError,ValueError,UnicodeError) as e:raise ValueError('在线留出锁损坏或不完整，拒绝读取留出数据') from e
        if existing!=expected:raise ValueError('首次留出评价已锁定策略、模型及测试来源；不能换门槛重复确认性评价')
    else:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(expected,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
            stream.flush();os.fsync(stream.fileno())
    return expected
