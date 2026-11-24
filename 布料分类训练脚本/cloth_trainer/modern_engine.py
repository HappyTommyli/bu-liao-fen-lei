"""V4: development-only CV, frozen final refit, one untouched holdout."""
from pathlib import Path
from datetime import datetime
import copy
import json
import shutil
import uuid
from dataclasses import replace
import numpy as np
import torch
from .modern_config import ModernConfig,ARCHITECTURES
from .modern_split import make_plan,validate_plan
from .modern_training import fit,evaluate,make_model
from .tcn_study import study_definition,validate_study,configuration_id,candidate_label,window_metrics,contrasts,study_description,description_text
from .modern_models import ModernClassifier
from .devices import resolve_device
from .data import file_hash,import_dataset,prepare_trials,protocol_for,required_axes,load_record
from .dataset import save_dataset,load_dataset
from .admission import package,require_admission,eligible_records,verify_raw_batches
from .storage import read_json,write_json,new_task
from .evaluation import classification_metrics,recognition_summary

FORMAT='cloth-classification-v4'


def frozen(task):
    task=Path(task);m=read_json(task/'metadata.json');e=read_json(task/'execution.json')
    if m['format']!=FORMAT or file_hash(task/'execution.json')!=m['execution_sha256']:raise ValueError('冻结配置校验失败')
    records,snapshot=load_dataset(task/'dataset')
    if file_hash(task/'dataset/dataset.json')!=e['dataset_sha256'] or snapshot['dataset_id']!=m['dataset_id']:raise ValueError('冻结数据校验失败')
    if e['config']!=m['config'] or e['plan']!=m['split_plan'] or snapshot['sources']!=m['sources']:raise ValueError('配置/分区/来源校验失败')
    cfg=ModernConfig.from_dict(e['config']);validate_plan(records,e['plan'])
    if len(e['plan']['cv'])!=cfg.cv_folds:raise ValueError('冻结划分次数与训练配置不同')
    require_admission(records,snapshot['admission'],cfg.data)
    validate_study(e.get('study'),cfg)
    if m.get('study')!=e.get('study'):raise ValueError('TCN研究候选关联不一致')
    if 'study_description' in e and m.get('study_description')!=e['study_description']:raise ValueError('冻结对照说明不一致')
    if m.get('study'):
        allowed=m['study']['candidates']
        for row in list(m['models'].values())+([m['final_model']] if m.get('final_model') else []):
            candidate=row.get('candidate')
            if row['architecture'] not in cfg.architectures or candidate not in allowed or row['configuration_id']!=configuration_id(row['architecture'],'by','mean',candidate):
                raise ValueError('模型候选与冻结研究关联不一致')
    return m,cfg,records


def initialize(request):
    from .engine import _verify_sources
    cfg=ModernConfig.from_dict(request['config']);resolve_device(cfg.device)
    if cfg.data.preprocess_version!='abs-v3':raise ValueError('新版训练必须使用abs-v3')
    if request.get('snapshot'):
        records,snapshot=load_dataset(request['snapshot']);screening=snapshot['screening'];admission=snapshot['admission']
        from .config import DataConfig
        if DataConfig.from_dict(snapshot['data'])!=cfg.data or snapshot['split_seed']!=cfg.split_seed:raise ValueError('快照协议或划分种子不同')
        require_admission(records,admission,cfg.data)
        if any(set(r.axes)!=set(required_axes(cfg.signals)) for r in records):raise ValueError('快照通道已冻结，请重新导入CSV')
    else:
        _verify_sources(request.get('sources',[]))
        if request.get('review_batches'):
            all_records=verify_raw_batches(request['review_batches'],cfg.data);selected={str(Path(p).resolve()) for p in request['paths']}
            records=[r for r in all_records if r.path in selected]
            if {r.path for r in records}!=selected:raise ValueError('记录不在完整审查批次中')
            admission=package(request['review_batches']);screening=dict(request.get('screening',{}),review_batches=request['review_batches'])
        else:
            records,screening=import_dataset(request['paths'],cfg.data,cfg.signals,request.get('overrides'));admission=package(screening['review_batches'])
        loaded={r.path:r for r in records}
        for s in request.get('sources',[]):
            r=loaded.get(str(Path(s['path']).resolve()))
            if r is None or r.sha256!=s['sha256'] or r.metadata_sha256!=s.get('metadata_sha256'):raise ValueError('来源已改变，请重新导入')
        records=eligible_records(records);require_admission(records,admission,cfg.data)
        prepared=[]
        for r in records:
            if set(r.axes)!=set(required_axes(cfg.signals)):
                p=load_record(r.path,cfg.data,r.label,signals=cfg.signals)
                if p.diagnostics['direction_quality']!=r.diagnostics['direction_quality']:raise ValueError('通道改变影响审查指标')
                p.audit=copy.deepcopy(r.audit);p.quality=r.quality;p.quality_notes=r.quality_notes.copy();r=p
            prepared.append(r)
        records=prepared
    for signal in cfg.signals:prepare_trials(records,signal)
    plan=copy.deepcopy(snapshot['split_plan']) if request.get('snapshot') and snapshot.get('split_plan') else make_plan(records,cfg.split_seed,cfg.cv_folds)
    validate_plan(records,plan)
    if len(plan['cv'])!=cfg.cv_folds:raise ValueError('冻结划分次数与训练配置不同')
    if plan['seed']!=cfg.split_seed:raise ValueError('固定划分种子与训练配置不一致')
    protocol=protocol_for(records,cfg.data)
    if request.get('protocol') is not None and request['protocol']!=protocol:raise ValueError('扫描协议不匹配')
    if request.get('snapshot') and Path(request['output']).resolve().is_relative_to(Path(request['snapshot']).resolve()):raise ValueError('成果目录不能位于快照内')
    task=new_task(request['output'],prefix='CLOTH4')
    if request.get('snapshot'):shutil.copytree(request['snapshot'],task/'dataset')
    else:
        from .folds import make_folds
        save_dataset(task/'dataset',records,cfg.data,cfg.split_seed,dict(screening,review_batches=admission['batches']),make_folds(records,cfg.split_seed,max(2,cfg.cv_folds)),signals=cfg.signals,admission=admission,split_plan=plan)
    records,snapshot=load_dataset(task/'dataset')
    m=dict(format=FORMAT,id=task.name,experiment_id=uuid.uuid4().hex,dataset_id=snapshot['dataset_id'],name=cfg.name or task.name,
        created_at=datetime.now().isoformat(),status='running',stage='classification',purpose='quick' if cfg.preset=='quick' else 'formal',
        labels=sorted({r.label for r in records}),records=len(records),trials=2*len(records),config=cfg.to_dict(),sources=snapshot['sources'],
        split_plan=plan,partitions=plan['cv'],protocol=protocol,models={},summaries={},stage2_status='not_available',screening=screening,
        admission=snapshot['admission'],final_model=None,holdout=None)
    execution=dict(config=cfg.to_dict(),plan=plan,dataset_sha256=file_hash(task/'dataset/dataset.json'))
    if cfg.tcn_spec is not None:
        m['study']=study_definition(cfg);execution['study']=m['study']
        m['study_description']=study_description(m['study']);execution['study_description']=m['study_description']
    write_json(task/'execution.json',execution)
    m['execution_sha256']=file_hash(task/'execution.json');write_json(task/'metadata.json',m)
    return task,m,cfg,records


def checkpoint_bundle(task,m,cfg,key,result,architecture,signal,readout,seed,phase,fold=None):
    folder=task/key;folder.mkdir(exist_ok=True)
    model=result['model'];model_id=uuid.uuid4().hex
    bundle=dict(format=FORMAT,model_id=model_id,dataset_id=m['dataset_id'],phase=phase,fold=fold,architecture=architecture,signal=signal,readout=readout,seed=seed,
        channels=len(result['scaler']['mean']),labels=m['labels'],config=m['config'],protocol=m['protocol'],scaler=result['scaler'],
        state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},sources=m['sources'],execution_sha256=m['execution_sha256'])
    candidate=getattr(model,'candidate',None)
    if candidate is not None:
        bundle.update(candidate=candidate,study_revision=m['study']['revision'])
    temp=folder/'model.pt.tmp';torch.save(bundle,temp);temp.replace(folder/'model.pt')
    row=dict(model_id=model_id,weight_sha256=file_hash(folder/'model.pt'),runtime=result['runtime'],best_epoch=result['best_epoch'],history=result['history'],parameters=sum(p.numel() for p in model.parameters()),
        architecture=architecture,signal=signal,readout=readout,seed=seed,configuration_id=configuration_id(architecture,signal,readout,candidate),fold=fold,status='completed',stage='classification',phase=phase,
        training_seconds=sum(row['seconds'] for row in result['history']),stage2_status='not_available')
    row['stop_reason']=result.get('stop_reason')
    if candidate is not None:
        row.update(candidate=candidate,candidate_label=candidate_label(candidate),receptive_field=model.receptive_field)
    return row


def summarize(m):
    grouped={}
    for r in m['models'].values():grouped.setdefault(r['configuration_id'],[]).append(r)
    return {k:dict(val_f1_mean=float(np.mean([r['val_metrics']['macro_f1'] for r in rr])),val_f1_std=float(np.std([r['val_metrics']['macro_f1'] for r in rr])),
        val_loss_mean=float(np.mean([r['val_loss'] for r in rr])),folds=len(rr),note='仅开发集；复验不是新增独立样本') for k,rr in grouped.items()}


def report(task,m):
    lines=['# 整段布料分类 · GPU标准流程','',f'实验：{m["name"]}',f'{m["records"]} CSV；开发集 {len(m["split_plan"]["dev"])}，固定内部测试 {len(m["split_plan"]["test"])}。',
        f'开发集{len(m["split_plan"]["cv"])}次训练用于选择配置；训练完成后可直接进行在线调参。最终重训需明确触发，测试集不参与标准化、选轮或调参。',
        '单程作为分类输入，去回程与同一会话始终同组。第二阶段未开放。','', '| 配置 | 验证Macro-F1均值 | 折间/复验标准差 |','| --- | --- | --- |']
    for k,s in m.get('summaries',{}).items():lines.append(f'| {k} | {s["val_f1_mean"]:.4f} | {s["val_f1_std"]:.4f} |')
    if m.get('study'):
        lines[0]='# TCN / |dBy| 训练实验'
        lines.insert(2,description_text(m.get('study_description') or study_description(m['study'])))
        lines=[line.replace('第二阶段未开放。','') for line in lines]
        lines+=['','主指标为原单程Macro-F1；窗口成绩为相关片段辅助统计。有效批按原单程计数，每父批全部窗口梯度累积后仅更新一次。',
                '','| 候选 | 参数量 | 感受野/点 | 窗口验证F1（辅助） |','| --- | --- | --- | --- |']
        seen=set()
        for row in m['models'].values():
            if row['configuration_id'] in seen:continue
            seen.add(row['configuration_id'])
            lines.append(f"| {row['candidate_label']} | {row['parameters']} | {row['receptive_field']} | {row['val_window_metrics']['macro_f1']:.4f}（当前折） |")
        if m.get('contrasts'):
            lines+=['','配对比较（同折/种子，父单程F1差值）：']
            lines += [f"- {name}: {value['mean']:+.4f}" for name,value in m['contrasts'].items()]

    if m.get('final_model'):lines+=['',f'最终模型：{m["final_model"]["configuration_id"]}；{m["final_model"]["epochs"]}轮；权重 {m["final_model"]["weight_sha256"]}。']
    if m.get('holdout'):
        h=m['holdout'];lines+=['',f'固定留出单程准确率 {h["metrics"]["accuracy"]:.2%}；Macro-F1 {h["metrics"]["macro_f1"]:.4f}。',f'往返合并准确率 {h["roundtrip_metrics"]["accuracy"]:.2%}；按CSV概率平均，每次往返只计1次。']
    (task/'报告.md').write_text('\n'.join(lines),encoding='utf-8');write_json(task/'汇总.json',dict(cv=m.get('summaries',{}),contrasts=m.get('contrasts',{}),final=m.get('final_model'),holdout=m.get('holdout')))


def train_request(request,emit=lambda e:None,cancelled=lambda:False):
    from .engine import _jobs,check_cancel,Cancelled
    if request.get('resume'):
        task=Path(request['resume']).resolve();m,cfg,records=frozen(task);resolve_device(cfg.device)
        for key,r in m['models'].items():
            if file_hash(task/key/'model.pt')!=r['weight_sha256']:raise ValueError('已完成折权重校验失败')
        if m['status']=='completed':return task
    else:task,m,cfg,records=initialize(request)
    def event(e):
        e=dict(e,task=str(task))
        with (task/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(e,ensure_ascii=False)+'\n')
        emit(e)
    try:
        m['status']='running';write_json(task/'metadata.json',m);jobs=[(*job,c) for job in _jobs(cfg) for c in ((m.get('study') or {}).get('candidates') or [None])]
        for ji,(a,s,r,repetition,fold,candidate) in enumerate(jobs,1):
            check_cancel(cancelled);seed=cfg.seed+repetition;identity=configuration_id(a,s,r,candidate);key=f'{identity}_s{seed}_f{fold+1}'
            if key in m['models']:continue
            def progress(e):event(dict(e,model=key,fold=fold,job=ji,jobs=len(jobs),epochs=cfg.epochs))
            progress(dict(type='phase',message=f'{candidate_label(candidate) if candidate else ARCHITECTURES[a]} · 开发集第{fold+1}/{cfg.cv_folds}次'))
            partition=m['split_plan']['cv'][fold];train=prepare_trials([records[i] for i in partition['train']],s);val=prepare_trials([records[i] for i in partition['val']],s)
            result=fit(train,val,m['labels'],cfg,a,r,seed,task/key,progress,cancelled,job_binding=dict(execution=m['execution_sha256'],key=key),frozen_config=m['config'],**(dict(candidate=candidate) if candidate is not None else {}))
            evalcfg=replace(cfg,micro_batch=result['runtime']['micro_batch'])
            vr,vloss=evaluate(result['model'],val,result['scaler'],m['labels'],evalcfg,cancelled);tr,_=evaluate(result['model'],train,result['scaler'],m['labels'],evalcfg,cancelled)
            row=checkpoint_bundle(task,m,cfg,key,result,a,s,r,seed,'cv',fold)
            row.update(repetition=repetition+1,train_metrics=classification_metrics(tr,m['labels']),val_metrics=classification_metrics(vr,m['labels']),val_predictions=vr,val_loss=vloss,
                val_direction_metrics={d:classification_metrics([x for x in vr if x['direction']==d],m['labels']) for d in ('out','back')},
                test_metrics=None,test_predictions=[],test_status='not_tested',validation_inference_seconds=0.)
            if candidate is not None:row['val_window_metrics']=window_metrics(vr,m['labels'])
            m['models'][key]=row;write_json(task/key/'result.json',row);write_json(task/'metadata.json',m);progress(dict(type='model_done',message='开发集验证已保存；测试集保持封存'))
        m.update(status='completed',completed_at=datetime.now().isoformat(),summaries=summarize(m))
        if cfg.tcn_spec is not None and cfg.tcn_mode in ('four_way','window_pool','overlap_pair'):m['contrasts']=contrasts(m)
        write_json(task/'metadata.json',m);report(task,m)
        event(dict(type='completed',message=f'开发集{cfg.cv_folds}次训练完成；可加载权重进行在线调参'));return task
    except Exception as exc:
        if isinstance(exc,Cancelled):event(dict(type='cancelled',message='用户取消训练；已完成轮次的断点保留，可恢复继续'))
        m.update(status='cancelled' if isinstance(exc,Cancelled) else 'failed',error=str(exc));write_json(task/'metadata.json',m);raise


def final_request(request,emit=lambda e:None,cancelled=lambda:False):
    task=Path(request['task']).resolve();m,cfg,records=frozen(task)
    if m['status']!='completed':raise ValueError('请先完成开发集五折')
    summaries=summarize(m);selected=request.get('configuration_id') or (m.get('final_selection') or {}).get('configuration_id') or max(summaries,key=lambda k:(summaries[k]['val_f1_mean'],-summaries[k]['val_loss_mean'],k))
    if m.get('final_selection') and selected!=m['final_selection']['configuration_id']:raise ValueError('最终配置已锁定，请建立新实验')
    if selected not in summaries:raise ValueError('候选配置不存在')
    candidates=[r for r in m['models'].values() if r['configuration_id']==selected]
    if len(candidates)!=cfg.cv_folds*cfg.repetitions:raise ValueError('开发集五折尚未完整')
    for key,r in m['models'].items():
        if file_hash(task/key/'model.pt')!=r['weight_sha256']:raise ValueError('开发集权重校验失败')
    epochs=max(1,int(np.median([r['best_epoch'] for r in candidates])))
    selection=dict(configuration_id=selected,epochs=epochs,cv_weights={k:r['weight_sha256'] for k,r in m['models'].items() if r['configuration_id']==selected})
    if m.get('final_selection') and selection!=m['final_selection']:raise ValueError('最终配置关联校验失败')
    if m.get('final_model'):
        load_bundle(task/'final/model.pt',allow_quick=True);return task
    m['final_selection']=selection;write_json(task/'metadata.json',m)
    reference=candidates[0];a,s,r=reference['architecture'],reference['signal'],reference['readout']
    train=prepare_trials([records[i] for i in m['split_plan']['dev']],s)
    def progress(e):emit(dict(e,task=str(task),model='final',scope='final',configuration_id=selected,epochs=epochs))
    progress(dict(type='phase',message='按已冻结候选进行最终重训；不使用验证或测试数据选轮'))
    result=fit(train,[],m['labels'],cfg,a,r,cfg.seed,task/'final',progress,cancelled,job_binding=dict(execution=m['execution_sha256'],selection=selection),fixed_epochs=epochs,frozen_config=m['config'],**(dict(candidate=reference['candidate']) if reference.get('candidate') else {}))
    row=checkpoint_bundle(task,m,cfg,'final',result,a,s,r,cfg.seed,'final');row.update(epochs=epochs,train_records=len(m['split_plan']['dev']))
    m['final_model']=row;write_json(task/'final/result.json',row);write_json(task/'metadata.json',m);report(task,m)
    emit(dict(type='final_done',task=str(task),message='最终模型训练完成；现在可运行固定留出测试'));return task


def load_bundle(path,allow_quick=False):
    path=Path(path).resolve();task=path.parent.parent;m,cfg,_=frozen(task);row=m.get('final_model')
    if m['status']!='completed' or path.parent.name!='final' or not row:raise ValueError('仅最终模型可用于正式识别；开发集折权重供研究检查')
    if m['purpose']=='quick' and not allow_quick:raise ValueError('快速检查不能用作正式成果')
    if file_hash(path)!=row['weight_sha256']:raise ValueError('最终权重校验失败')
    b=torch.load(path,map_location='cpu',weights_only=True)
    if b['format']!=FORMAT or b['model_id']!=row['model_id'] or b['dataset_id']!=m['dataset_id'] or b['execution_sha256']!=m['execution_sha256'] or b['phase']!='final' or json.loads(json.dumps(b['config']))!=m['config']:raise ValueError('最终模型关联校验失败')
    candidate=b.get('candidate')
    if m.get('study'):
        if candidate!=row.get('candidate') or candidate not in m['study']['candidates'] or b.get('study_revision')!=m['study']['revision']:
            raise ValueError('最终模型候选关联校验失败')
    elif candidate is not None:raise ValueError('旧模型不能包含未声明的TCN候选')
    model=make_model(cfg,b['channels'],len(b['labels']),b['architecture'],b['readout'],candidate);model.load_state_dict(b['state_dict']);model.eval();b['_model']=model
    return b


def test_request(request,emit=lambda e:None,cancelled=lambda:False):
    task=Path(request['task']).resolve();m,cfg,records=frozen(task)
    if not m.get('final_model'):raise ValueError('请先训练最终模型')
    if request.get('keys') and set(request['keys']) not in ({'final'},{m['final_model']['configuration_id']}):raise ValueError('固定测试仅允许已锁定最终模型')
    b=load_bundle(task/'final/model.pt');weight=file_hash(task/'final/model.pt')
    if m.get('test_weight_sha256') and m['test_weight_sha256']!=weight:raise ValueError('测试权重已锁定')
    if m.get('holdout'):return task
    m['test_weight_sha256']=weight;m['test_started_at']=datetime.now().isoformat();write_json(task/'metadata.json',m)
    trials=prepare_trials([records[i] for i in m['split_plan']['test']],b['signal']);evalcfg=replace(cfg,device='cpu',precision='fp32',workers=0)
    rows,_=evaluate(b['_model'],trials,b['scaler'],b['labels'],evalcfg,cancelled)
    groups={}
    for x in rows:groups.setdefault(x['sha256'],[]).append(x)
    paired=[]
    for values in groups.values():
        if len(values)!=2 or {v['direction'] for v in values}!={'out','back'}:raise ValueError('往返测试配对不完整')
        p=np.mean([v['scores'] for v in values],0);paired.append(dict(values[0],direction='roundtrip',scores=p.tolist(),predicted=int(p.argmax())))
    metrics=classification_metrics(rows,b['labels']);matrix=np.asarray(metrics['confusion']);den=matrix.sum(1,keepdims=True)
    m['holdout']=dict(weight_sha256=weight,metrics=metrics,confusion_normalized=np.divide(matrix,den,out=np.zeros_like(matrix,dtype=float),where=den>0).tolist(),predictions=rows,
        direction_metrics={d:classification_metrics([r for r in rows if r['direction']==d],b['labels']) for d in ('out','back')},roundtrip_metrics=classification_metrics(paired,b['labels']),roundtrip_predictions=paired)
    if m.get('study'):m['holdout']['window_metrics']=window_metrics(rows,b['labels'])
    write_json(task/'holdout.json',m['holdout']);write_json(task/'metadata.json',m);report(task,m)
    emit(dict(type='test_done',task=str(task),message='固定内部测试完成；重复查看不重新选择模型'));return task


def load_online_bundle(path,allow_quick=False):
    """Load a frozen completed CV/final model for online parameter tuning only."""
    path=Path(path).resolve();task=path.parent.parent;m,cfg,records=frozen(task)
    key=path.parent.name
    row=m.get('final_model') if key=='final' else m['models'].get(key)
    if path.name!='model.pt' or not row or row.get('status')!='completed':raise ValueError('请选择已完成训练的模型权重')
    if m['purpose']=='quick' and not allow_quick:raise ValueError('快速检查不能用作正式成果')
    phase='final' if key=='final' else 'cv'
    if row.get('phase')!=phase or file_hash(path)!=row['weight_sha256']:raise ValueError('在线模型权重校验失败')
    b=torch.load(path,map_location='cpu',weights_only=True)
    expected=dict(format=FORMAT,model_id=row['model_id'],dataset_id=m['dataset_id'],execution_sha256=m['execution_sha256'],phase=phase,
                  labels=m['labels'],protocol=m['protocol'])
    for field,value in expected.items():
        if b.get(field)!=value:raise ValueError('在线模型关联校验失败：'+field)
    if json.loads(json.dumps(b.get('config')))!=m['config']:raise ValueError('在线模型配置关联校验失败')
    for field in ('architecture','signal','readout','seed','fold'):
        if b.get(field)!=row.get(field):raise ValueError('在线模型关联校验失败：'+field)
    fold=b.get('fold')
    if phase=='cv':
        if type(fold) is not int or not 0<=fold<cfg.cv_folds:raise ValueError('在线模型开发折关联无效')
        partition=m['split_plan']['cv'][fold]
    else:partition=dict(train=m['split_plan']['dev'],val=[])
    candidate=b.get('candidate')
    if m.get('study'):
        if candidate!=row.get('candidate') or candidate not in m['study']['candidates'] or b.get('study_revision')!=m['study']['revision']:
            raise ValueError('在线模型候选关联校验失败')
    elif candidate is not None:raise ValueError('旧模型不能包含未声明的TCN候选')
    if b.get('sources')!=m['sources']:raise ValueError('在线模型来源关联校验失败')
    channels=1 if b['signal'] in ('by','bx') else 2
    if b.get('channels')!=channels:raise ValueError('在线模型通道关联校验失败')
    scaler=b.get('scaler',{})
    try:
        mean=np.asarray(scaler['mean'],dtype=float);std=np.asarray(scaler['std'],dtype=float)
        valid=mean.shape==(channels,) and std.shape==(channels,) and np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all()
    except (KeyError,TypeError,ValueError):valid=False
    if not valid:raise ValueError('在线模型标准化scaler无效')
    model=make_model(cfg,channels,len(b['labels']),b['architecture'],b['readout'],candidate)
    model.load_state_dict(b['state_dict']);model.eval();b['_model']=model
    b['model_source']=dict(task=str(task),path=str(path),key=key,phase=phase,weight_sha256=row['weight_sha256'],model_id=b['model_id'],dataset_id=b['dataset_id'])
    b['split_audit']=dict(seed=m['split_plan']['seed'],cv_folds=cfg.cv_folds,fold=fold,train=list(partition['train']),val=list(partition['val']),test=list(m['split_plan']['test']),
                          validation_sources=[records[i].manifest() for i in partition['val']])
    return b
