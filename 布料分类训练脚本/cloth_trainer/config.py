from dataclasses import dataclass, field, asdict, replace
import math

CHECKPOINTS=(.10,.20,.30,.50,.75,1.0)
SIGNALS={'by':'ΔBy','bxby':'ΔBx＋ΔBy','xy':'XY模长','bx':'ΔBx'}
ABS_SIGNALS={'by':'|ΔBy|','bxby':'[|ΔBx|, |ΔBy|]','xy':'XY模长','bx':'|ΔBx|'}

def input_label(signal,version='abs-v3'):
    return (ABS_SIGNALS if version in ('abs-v3','cloth-classification-v3','cloth-classification-v4') else SIGNALS)[signal]

ARCHITECTURES={'gru':'GRU','tcn':'TCN','cera':'CERA'}
READOUTS={'mean':'整段特征汇总','last':'最后时刻读出（对照）'}
PRESETS={'standard':'标准训练','quick':'快速检查（不作为正式成果）','repeat':'重复验证（每项3次）','custom':'自定义'}
VARIANTS={'full':'完整CERA-Net','local':'仅TCN固定检查点','context':'仅GRU固定检查点',
          'plain':'无门控','no_evidence':'无证据状态','no_risk':'无风险头',
          'no_calibration':'无校准（探索）','no_stability':'无两点稳定',
          'softmax':'max-softmax停止','entropy':'entropy停止'}

@dataclass(frozen=True)
class DataConfig:
    out_low:float=51.
    out_high:float=140.
    back_low:float=30.
    back_high:float=120.
    fallback_ppmm:float=250.
    fallback_speed:float=6.
    fs:float=333.
    source_fs:float=333.
    max_gap:float=.020
    tail_margin:float=.200
    speed_tolerance:float=.10
    preprocess_version:str='abs-v3'
    baseline_tolerance:float=.01
    boundary_tolerance:float=2.
    contact_rule:str='fixed-x-v1'
    sample_quality:str='selected-axis-v1'

    @classmethod
    def from_dict(cls,value):
        value=dict(value)
        value.setdefault('contact_rule','fixed-x-v1')
        value.setdefault('sample_quality','recorded-v1')
        for key,default in dict(out_low=45.,out_high=142.,back_low=16.,back_high=114.).items():
            value.setdefault(key,default)
        return cls(**value).validate()

    def validate(self):
        if not all(math.isfinite(v) for v in asdict(self).values() if not isinstance(v,str)):raise ValueError('参数必须为有限数')
        if not (0<=self.out_low<self.out_high and 0<=self.back_low<self.back_high):raise ValueError('裁剪范围需满足0≤下界<上界')
        if min(self.fs,self.source_fs,self.fallback_ppmm,self.fallback_speed,self.max_gap)<=0 or self.tail_margin<0:raise ValueError('时基、速度、脉冲换算和缺口参数无效')
        if self.preprocess_version not in ('legacy-v2','abs-v3'):raise ValueError('未知预处理版本')
        if self.contact_rule not in ('fixed-x-v1','relative-035-v1'):raise ValueError('未知接触段规则')
        if self.sample_quality not in ('recorded-v1','selected-axis-v1'):raise ValueError('未知采样质量规则')
        if self.baseline_tolerance<0 or self.boundary_tolerance<0:raise ValueError('核对容差无效')
        if self.fs!=333.:raise ValueError('当前协议固定333Hz')
        if not 0<self.speed_tolerance<=.5:raise ValueError('速度容差需在0到0.5之间')
        return self

@dataclass
class TrainConfig:
    data:DataConfig=field(default_factory=DataConfig)
    signals:tuple=('by',)
    architectures:tuple=('gru','tcn','cera')
    readouts:tuple=('mean',)
    preset:str='standard'
    repetitions:int=1
    epochs:int=200
    patience:int=30
    early_stopping:bool=False
    batch_size:int=16
    lr:float=.001
    weight_decay:float=.0001
    seed:int=42
    split_seed:int=42
    threads:int=2
    hidden_size:int=64
    dropout:float=.1
    scheduler:bool=True
    scheduler_patience:int=8
    min_lr:float=.00001
    name:str=''

    def validate(self):
        self.data.validate()
        if not self.signals or set(self.signals)-SIGNALS.keys() or len(set(self.signals))!=len(self.signals):raise ValueError('请选择有效输入')
        for value,allowed,title in [(self.architectures,ARCHITECTURES,'模型'),(self.readouts,READOUTS,'读出方式')]:
            if not value or set(value)-allowed.keys() or len(set(value))!=len(value):raise ValueError('请选择有效'+title)
        if not 1<=self.epochs<=1000 or not 1<=self.patience<=1000 or not 1<=self.batch_size<=128:raise ValueError('训练轮数或批大小无效')
        if not math.isfinite(self.lr) or self.lr<=0 or not math.isfinite(self.weight_decay) or self.weight_decay<0:raise ValueError('优化器参数无效')
        if not 1<=self.threads<=32:raise ValueError('CPU线程数需为1到32')
        if self.preset not in PRESETS or not 1<=self.repetitions<=10:raise ValueError('训练预设或复验次数无效')
        if self.hidden_size not in (32,64,128) or not math.isfinite(self.dropout) or not 0<=self.dropout<1:raise ValueError('模型宽度或随机忽略比例无效')
        if not 0<=self.seed<=999999 or not 0<=self.split_seed<=999999:raise ValueError('随机种子需为0到999999')
        if not 1<=self.scheduler_patience<=1000 or not 0<self.min_lr<=self.lr:raise ValueError('学习率调度设置无效')
        return self

    def to_dict(self):return asdict(self)

    @classmethod
    def from_dict(cls,v):
        v=dict(v);data=dict(v.get('data',{}));data.setdefault('preprocess_version','legacy-v2');v['data']=DataConfig.from_dict(data)
        # Old projects may recover data/settings, but never resume joint training.
        v.pop('variants',None)
        for k in ['signals','architectures','readouts']:
            if k in v:v[k]=tuple(v[k])
        return cls(**v).validate()

    def with_preset(self,name):
        if name not in ('quick','standard','repeat'):raise ValueError('未知训练预设')
        return replace(self,preset=name,epochs=30 if name=='quick' else 200,
                       patience=8 if name=='quick' else 30,repetitions=3 if name=='repeat' else 1,
                       batch_size=16,lr=.001,weight_decay=.0001,dropout=.1,scheduler=True,early_stopping=name=='quick').validate()
