"""Versioned defaults for GPU-first classification; legacy config stays intact."""
from dataclasses import dataclass,fields,replace
from .config import TrainConfig

PIPELINE='gpu-standard-v4'
ARCHITECTURES={'gru':'GRU','tcn':'TCN','cera':'CERA（研究）','conv_gru':'Conv-GRU（压缩对照）','conv_cera':'Conv-CERA（压缩对照）'}

@dataclass
class ModernConfig(TrainConfig):
    pipeline:str=PIPELINE
    device:str='cuda'
    precision:str='fp32'
    workers:int=2
    micro_batch:int=0
    lr:float=.0003
    early_stopping:bool=True
    tcn_spec:dict | None=None
    tcn_mode:str='single'
    tcn_window:int=900
    tcn_stride:int=0
    cv_folds:int=5
    grad_clip_norm:float=1.
    scheduler_factor:float=.5

    def to_dict(self):
        value=super().to_dict()
        if self.tcn_spec is None:
            for key in ('tcn_spec','tcn_mode','tcn_window','grad_clip_norm','scheduler_factor'):
                value.pop(key,None)
        return value

    def validate(self):
        if not self.architectures or set(self.architectures)-ARCHITECTURES.keys():raise ValueError('请选择有效模型')
        base={f.name:getattr(self,f.name) for f in fields(TrainConfig)}
        base['architectures']=tuple(dict.fromkeys(a.removeprefix('conv_') for a in self.architectures))
        TrainConfig(**base).validate()
        if type(self.cv_folds) is not int or not 1<=self.cv_folds<=10:raise ValueError('开发集划分次数需为1到10')
        if type(self.tcn_stride) is not int or not 0<=self.tcn_stride<=(8192 if self.tcn_mode=='prefix' else self.tcn_window):raise ValueError('TCN步长需为0或1到窗口长度')
        if self.pipeline!=PIPELINE or self.device not in ('cuda','cpu'):raise ValueError('训练流程或设备无效')
        if self.precision not in ('fp32','amp_fp16','amp_bf16'):raise ValueError('精度策略无效')
        if self.device=='cpu' and self.precision!='fp32':raise ValueError('CPU仅支持显式FP32验证')
        if not 0<=self.workers<=8 or not 0<=self.micro_batch<=(128 if self.tcn_spec is not None else self.batch_size):raise ValueError('数据加载进程或微批大小无效')
        if self.tcn_spec is not None:
            import math
            from .tcn_models import default_spec
            from .cera_models import normalize_model_spec as normalize_spec,CERA_REVISIONS,PREFIX_REVISION
            from .tcn_study import validate_candidate,make_candidate
            spec=normalize_spec(self.tcn_spec)
            expected='cera' if spec['revision'] in CERA_REVISIONS else 'tcn'
            if self.architectures!=(expected,) or self.signals!=('by',) or self.readouts!=('mean',) or self.data.preprocess_version!='abs-v3':
                raise ValueError('窗口模型结构须匹配TCN/CERA架构，输入须为 |dBy| / abs-v3')
            if expected=='cera' and self.tcn_mode not in ('single','overlap_pair','prefix'):raise ValueError('CERA支持单配置或重叠两组对照')
            if self.tcn_mode not in ('single','four_way','window_pool','overlap_pair','prefix'):
                raise ValueError('TCN实验模式无效')
            if self.tcn_mode=='overlap_pair' and not 0<self.tcn_stride<self.tcn_window:
                raise ValueError('重叠对照需使用切窗，B组步长须大于0且小于窗口长度')
            if (self.tcn_mode=='prefix')!=(spec['revision']==PREFIX_REVISION):raise ValueError('累积前缀需使用独立CERA前缀模型版本')
            if self.tcn_mode=='prefix' and (not self.tcn_window or self.tcn_stride<1):raise ValueError('前缀最少观察量及预测间隔须为正数')
            validate_candidate(make_candidate('custom',spec,self.tcn_window,self.tcn_stride if self.tcn_mode=='prefix' else 0))
            if self.tcn_mode=='four_way' and (spec!=default_spec() or self.tcn_window!=900):
                raise ValueError('四组对照的结构和900点窗口已锁定')
            if self.tcn_mode=='window_pool' and (spec!=default_spec(128) or self.tcn_window!=900):
                raise ValueError('窗口与汇总对照锁定128通道基础结构和900点基准')
            if not math.isfinite(self.grad_clip_norm) or self.grad_clip_norm<=0:
                raise ValueError('梯度裁剪上限需为有限正数')
            if not math.isfinite(self.scheduler_factor) or not 0<self.scheduler_factor<1:
                raise ValueError('学习率衰减倍率需在0到1之间')
        return self

    def with_preset(self,name):
        if name not in ('quick','standard','repeat'):raise ValueError('未知预设')
        return replace(self,preset=name,epochs=30 if name=='quick' else 200,patience=8 if name=='quick' else 30,
                       repetitions=3 if name=='repeat' else 1,batch_size=16,micro_batch=0,lr=.0003,
                       weight_decay=.0001,dropout=.1,scheduler=True,early_stopping=True).validate()
