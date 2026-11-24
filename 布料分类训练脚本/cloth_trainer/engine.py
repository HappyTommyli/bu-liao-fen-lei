"""Stage-one classification. Fit, held-out test, and inference are separate jobs."""
from pathlib import Path
from datetime import datetime
import copy
import csv
import json
import hashlib
import platform
import time
import uuid
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from . import VERSION
from .config import TrainConfig,DataConfig,ARCHITECTURES,SIGNALS,READOUTS,input_label
from .data import load_record,import_dataset,import_records,prepare_trials,file_hash,metadata_hash,protocol_for,matches_protocol,required_axes
from .dataset import save_dataset,load_dataset
from .admission import package,require_admission,eligible_records,verify_raw_batches
from .folds import fold_partition
from .evaluation import classification_metrics,recognition_summary,summarize_configs
from .classifiers import SegmentClassifier
from .storage import read_json,write_json,new_task
from .recovery import save_recovery,load_recovery
from .numerical import VERSION as NUMERICAL_VERSION,NumericalFailure,require_finite,tensor_summary,trial_details,json_safe,clip_gradients

FORMAT='cloth-classification-v3'
class Cancelled(Exception):pass

def check_cancel(cancelled):
    if cancelled():raise Cancelled('已取消；未完成任务不作为正式成果')


def fit_scaler(trials):
    x=np.concatenate([t.values for t in trials]).astype(np.float64)
    return dict(mean=x.mean(0).tolist(),std=np.maximum(x.std(0),1e-6).tolist())

def scaled(trial,scaler):
    return ((trial.values-np.array(scaler['mean']))/np.array(scaler['std'])).astype(np.float32)

def batch(trials,scaler,labels):
    values=[torch.from_numpy(scaled(t,scaler)) for t in trials]
    x=pad_sequence(values,batch_first=True);lengths=torch.tensor([len(t.values) for t in trials])
    y=torch.tensor([labels.index(t.label) if t.label in labels else -1 for t in trials])
    return x,y,lengths

def evaluate(model,trials,scaler,labels,cancelled=lambda:False):
    model.eval();rows=[];loss=0.;count=0
    with torch.no_grad():
        for start in range(0,len(trials),8):
            check_cancel(cancelled);tt=trials[start:start+8];x,y,lengths=batch(tt,scaler,labels)
            context=dict(batch=start//8+1,trials=trial_details(tt),input=tensor_summary(x))
            require_finite(x,'validation_input',**context)
            logits=model(x,lengths);require_finite(logits,'validation_logits',**context)
            probabilities=logits.softmax(-1);require_finite(probabilities,'validation_probabilities',**context)
            p=probabilities.numpy();known=y>=0
            if known.any():
                ce=F.cross_entropy(logits[known],y[known],reduction='sum');require_finite(ce,'validation_loss',**context)
                loss+=float(ce);count+=int(known.sum())
            for i,t in enumerate(tt):
                rows.append(dict(id=t.id,path=t.path,sha256=t.sha256,group=t.group,label=t.label,true=int(y[i]),direction=t.direction,
                                 predicted=int(p[i].argmax()),scores=p[i].tolist(),elapsed=float((len(t.values)-1)/333),quality=getattr(t,'quality','normal')))
    return rows,loss/count if count else None

def _verify_sources(sources):
    for s in sources:
        if file_hash(s['path'])!=s['sha256']:raise ValueError('源文件已改变，请重新导入：'+s['path'])
        if metadata_hash(s['path'])!=s.get('metadata_sha256'):raise ValueError('元数据已改变，请重新导入：'+s['path'])

def _load_records(paths,cfg,sources,overrides=None):
    _verify_sources(sources)
    records,errors=import_records(paths,cfg,overrides)
    if errors:raise ValueError('数据检查失败：'+str(errors))
    loaded={r.path:r for r in records}
    for s in sources:
        r=loaded.get(str(Path(s['path']).resolve()))
        if r is None or r.sha256!=s['sha256'] or r.metadata_sha256!=s.get('metadata_sha256'):
            raise ValueError('源文件或元数据在导入期间发生改变，请重新导入')
    return records

def _record_numerical_failure(folder,exc,context,model,emit):
    details=dict(version=NUMERICAL_VERSION,**context);details.update(exc.details)
    details['parameters']={name:tensor_summary(p) for name,p in model.named_parameters()} if model is not None else {}
    details['gradients']={name:tensor_summary(p.grad) for name,p in model.named_parameters() if p.grad is not None} if model is not None else {}
    recovery_epoch=details.get('recovery_epoch')
    details.update(recovery_path=str(folder/'recovery.ckpt') if recovery_epoch is not None else None,
                   action='保留已完成折；检查异常报告。继续任务将恢复最近完整轮，参数调整需新建实验。')
    details.update(report_id=uuid.uuid4().hex,created_at=datetime.now().astimezone().isoformat())
    report_path=folder/'numerical_failures'/(details['report_id']+'.json')
    diagnostic=json_safe(details);write_json(report_path,diagnostic)
    write_json(folder/'numerical_failure.json',diagnostic)  # Convenient latest copy; events reference the immutable original.
    message=str(exc)+f'；诊断报告：{report_path}；'+(f'可恢复到第{recovery_epoch}轮' if recovery_epoch is not None else '尚无可恢复轮次')
    emit(dict(type='numerical_failure',epoch=details.get('epoch'),recovery_epoch=recovery_epoch,diagnostic=str(report_path),message=message))
    return message


def _fit(train,val,labels,cfg,architecture,readout,seed,emit,cancelled,folder,job_binding=None,allow_runtime_upgrade=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    # Hash actual model inputs as well as source identities: a checkpoint cannot
    # be reused for different processing, labels, ordering or partitions.
    def signature(trials):
        return [dict(id=t.id,path=t.path,sha256=t.sha256,group=t.group,label=t.label,direction=t.direction,
                     shape=list(t.values.shape),dtype=str(t.values.dtype),values_sha256=hashlib.sha256(t.values.tobytes()).hexdigest()) for t in trials]
    binding=dict(runtime=NUMERICAL_VERSION,torch=str(torch.__version__),numpy=np.__version__,threads=torch.get_num_threads(),deterministic=torch.are_deterministic_algorithms_enabled(),
                 config=json.loads(json.dumps(cfg.to_dict())),architecture=architecture,readout=readout,seed=seed,labels=labels,
                 train=signature(train),val=signature(val),job=job_binding)
    epoch=1;batch_index=None;tt=train;scaler=None;x=None;model=None;optim=None;gradient_norm=None;clip_stats=None;recovery_epoch=None
    try:
        # Validate before fitting the scaler or creating the initial checkpoint.
        for subset in (train,val):
            for trial in subset:
                tt=[trial];require_finite(torch.from_numpy(trial.values),'input',trials=trial_details(tt),input=tensor_summary(trial.values))
        scaler=fit_scaler(train)
        require_finite({k:torch.tensor(v,dtype=torch.float64) for k,v in scaler.items()},'scaler')
        torch.manual_seed(seed)
        model=SegmentClassifier(train[0].values.shape[1],len(labels),architecture,readout,cfg.hidden_size,cfg.dropout)
        optim=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode='min',factor=.5,patience=cfg.scheduler_patience,min_lr=cfg.min_lr) if cfg.scheduler else None
        rng=np.random.default_rng(seed);best=(-1.,-1e300);best_state=None;history=[];stale=0;best_epoch=0;completed_epoch=0;finished=False
        saved=load_recovery(folder,binding,allow_runtime_upgrade=allow_runtime_upgrade)
        if saved is not None:
            model.load_state_dict(saved['model']);optim.load_state_dict(saved['optimizer'])
            if scheduler:scheduler.load_state_dict(saved['scheduler'])
            scaler=saved['scaler'];rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'])
            best=tuple(saved['best']);best_state=saved['best_state'];history=saved['history'];stale=saved['stale'];best_epoch=saved['best_epoch']
            completed_epoch=saved['epoch'];finished=saved['finished'];recovery_epoch=completed_epoch
            write_json(folder/'history.json',history)
            emit(dict(type='restored',epoch=completed_epoch,history=copy.deepcopy(history),message=f'已恢复第{completed_epoch}轮状态；'+('训练轮次已完成，保存最佳模型' if finished else f'从第{completed_epoch+1}轮继续')))
        else:
            if (folder/'history.json').exists():
                import shutil
                backup=folder/'history-before-restart.json'
                if not backup.exists():shutil.copyfile(folder/'history.json',backup)
                emit(dict(type='restarted',message='旧任务没有按轮恢复点；该折从第1轮重跑，原历史已保留'))
        def checkpoint():
            state=dict(epoch=completed_epoch,finished=finished,model=model.state_dict(),optimizer=optim.state_dict(),scheduler=scheduler.state_dict() if scheduler else None,
                       scaler=scaler,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),best=best,best_state=best_state,
                       history=history,stale=stale,best_epoch=best_epoch)
            save_recovery(folder,state,binding)
        if saved is None:checkpoint();recovery_epoch=0
        elif saved.get('_runtime_upgrade'):
            checkpoint()
            emit(dict(type='runtime_upgraded',epoch=completed_epoch,upgrade=saved['_runtime_upgrade'],
                      message=f'修复验证副本已兼容旧第{completed_epoch}轮恢复点；使用64位梯度范数，原实验保留'))
        for epoch in range(completed_epoch+1,cfg.epochs+1) if not finished else ():
            check_cancel(cancelled);model.train();total=0.;correct=0;start_time=time.monotonic();clip_rows=[]
            order=rng.permutation(len(train))
            for start in range(0,len(order),cfg.batch_size):
                check_cancel(cancelled);tt=[train[i] for i in order[start:start+cfg.batch_size]];batch_index=start//cfg.batch_size+1;gradient_norm=None;clip_stats=None
                x,y,lengths=batch(tt,scaler,labels);optim.zero_grad(set_to_none=True)
                require_finite(x,'input')
                logits=model(x,lengths);require_finite(logits,'logits')
                loss=F.cross_entropy(logits,y);require_finite(loss,'loss')
                loss.backward()
                clip_stats=clip_gradients(model.named_parameters());gradient_norm=clip_stats['norm'];clip_rows.append(clip_stats)
                optim.step()
                require_finite(dict(model=model.state_dict(),optimizer=optim.state_dict()),'optimizer')
                total+=float(loss.detach())*len(tt);correct+=int((logits.detach().argmax(-1)==y).sum())
            vr,vloss=evaluate(model,val,scaler,labels,cancelled);vm=classification_metrics(vr,labels)
            score=(vm['macro_f1'],-vloss)
            require_finite(torch.tensor([total,vloss,*score],dtype=torch.float64),'epoch_metrics')
            improved=score[0]>best[0]+1e-4
            stale=0 if improved else stale+1
            if score>best:best=score;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
            row=dict(epoch=epoch,loss=total/len(train),train_accuracy=correct/len(train),val_loss=vloss,val_f1=vm['macro_f1'],val_accuracy=vm['accuracy'],lr=optim.param_groups[0]['lr'],seconds=time.monotonic()-start_time)
            row.update(gradient_norm_max=max(c['norm'] for c in clip_rows),gradient_abs_max=max(c['max_abs'] for c in clip_rows),
                       gradient_clip_fraction=sum(c['clipped'] for c in clip_rows)/len(clip_rows),gradient_clipped_norm_max=max(c['clipped_norm'] for c in clip_rows),
                       gradient_clip_coefficient_min=min(c['coefficient'] for c in clip_rows))
            history.append(row)
            if scheduler:scheduler.step(vloss)
            completed_epoch=epoch;finished=epoch>=cfg.epochs or (cfg.early_stopping and stale>=cfg.patience)
            checkpoint();recovery_epoch=epoch
            write_json(folder/'history.json',history)
            warning=' · 大梯度已安全裁剪（详见记录）' if row['gradient_norm_max']>float(torch.finfo(torch.float32).max)**.5 else ''
            emit(dict(type='epoch',**row,message=f'第{epoch}/{cfg.epochs}轮 · 分类损失 {row["loss"]:.4f} · 验证F1 {row["val_f1"]:.3f}'+warning))
            if finished:break
        model.load_state_dict(best_state);model.eval()
        return model,scaler,history,best_epoch
    except NumericalFailure as exc:
        context=dict(job=job_binding,architecture=architecture,readout=readout,seed=seed,epoch=epoch,batch=batch_index,
                     trials=trial_details(tt),input=tensor_summary(x) if x is not None else None,scaler=scaler,
                     gradient_norm=gradient_norm,max_abs_gradient=clip_stats['max_abs'] if clip_stats else None,
                     clip_coefficient=clip_stats['coefficient'] if clip_stats else None,clipped_gradient_norm=clip_stats['clipped_norm'] if clip_stats else None,
                     recovery_epoch=recovery_epoch,phase='fit')
        raise ValueError(_record_numerical_failure(folder,exc,context,model,emit)) from exc

def _jobs(cfg):
    return [(a,s,r,i,f) for a in cfg.architectures for s in cfg.signals for r in cfg.readouts for i in range(cfg.repetitions) for f in range(getattr(cfg,'cv_folds',5))]

def _frozen_task(task):
    meta=read_json(task/'metadata.json');execution=read_json(task/'execution.json')
    if file_hash(task/'execution.json')!=meta.get('execution_sha256'):raise ValueError('冻结训练配置校验失败')
    records,snapshot=load_dataset(task/'dataset')
    if file_hash(task/'dataset/dataset.json')!=execution['dataset_sha256'] or snapshot['dataset_id']!=meta['dataset_id']:raise ValueError('冻结数据集版本已改变')
    if meta['config']!=execution['config'] or meta['partitions']!=[fold_partition(snapshot['folds'],i) for i in range(5)] or meta['sources']!=snapshot['sources']:
        raise ValueError('冻结参数/分折/来源与实验记录不一致')
    return meta,TrainConfig.from_dict(execution['config']),records

def train_request(request,emit=lambda e:None,cancelled=lambda:False):
    if request.get("config",{}).get("pipeline")=="gpu-standard-v4" or (request.get("resume") and read_json(Path(request["resume"])/"metadata.json").get("format")=="cloth-classification-v4"):
        from .modern_engine import train_request as modern_train
        return modern_train(request,emit,cancelled)
    if request.get('resume'):
        task=Path(request['resume']).resolve();meta,cfg,records=_frozen_task(task)
        if meta.get('format')!=FORMAT:raise ValueError('只能继续新版五折任务')
        if meta['status']=='completed':emit(dict(type='completed',task=str(task),message='该实验已完成，无需重训'));return task
        require_admission(records,read_json(task/'dataset/dataset.json').get('admission'),cfg.data)
        for key,r in meta['models'].items():
            if file_hash(task/key/'model.pt')!=r['weight_sha256']:raise ValueError('已完成折的权重校验失败')
        if meta.get('training_runtime') not in (None,'numerical-safety-v1',NUMERICAL_VERSION):raise ValueError('未知训练数值处理版本，不能自动升级')
        if meta.get('training_runtime')!=NUMERICAL_VERSION:
            pending=[f'{a}_{s}_{r}_s{cfg.seed+i}_f{f+1}' for a,s,r,i,f in _jobs(cfg) if f'{a}_{s}_{r}_s{cfg.seed+i}_f{f+1}' not in meta['models']]
            if not pending:raise ValueError('各折已经保存，没有待验证折；请查看原结果或建立新版实验')
            from .runtime_validation import fork_runtime_validation
            task,meta=fork_runtime_validation(task,meta,pending[0],NUMERICAL_VERSION)
            emit(dict(type='validation_forked',task=str(task),parent_task=meta['parent_task'],message='已建立独立修复续训验证副本，仅验证当前未完成折；原实验保留'))
            meta,cfg,records=_frozen_task(task)
        meta.update(status='running',resumed_at=datetime.now().isoformat());write_json(task/'metadata.json',meta)
    else:
        cfg=TrainConfig.from_dict(request['config'])
        if cfg.data.preprocess_version!='abs-v3':raise ValueError('旧预处理配置仅供旧模型识别；新训练需使用abs-v3')
        if request.get('snapshot'):
            records,snapshot=load_dataset(request['snapshot']);screening=snapshot['screening']
            admission=snapshot.get('admission');require_admission(records,admission,cfg.data)
            if snapshot['data']!=cfg.to_dict()['data'] or snapshot['split_seed']!=cfg.split_seed:raise ValueError('快照的采集协议/分折编号与设置不同，请建立新版本')
        else:
            _verify_sources(request.get('sources',[]))
            if request.get('review_batches'):
                records=verify_raw_batches(request['review_batches'],cfg.data)
                selected={str(Path(p).resolve()) for p in request['paths']}
                records=[r for r in records if r.path in selected]
                if {r.path for r in records}!=selected:raise ValueError('选择记录未包含在完整审查批次中')
                admission=package(request['review_batches']);require_admission(records,admission,cfg.data)
                screening=dict(request.get('screening',{}),review_batches=request['review_batches'])
            else:
                records,screening=import_dataset(request['paths'],cfg.data,cfg.signals,request.get('overrides'))
                admission=package(screening['review_batches'])
            loaded={r.path:r for r in records}
            for source in request.get('sources',[]):
                r=loaded.get(str(Path(source['path']).resolve()))
                if r is None or r.sha256!=source['sha256'] or r.metadata_sha256!=source.get('metadata_sha256'):raise ValueError('来源快照已改变或导入记录不一致')
            records=eligible_records(records);require_admission(records,admission,cfg.data)
            # Audit the full original channel scope, then prepare the selected
            # model channels independently from the same verified raw sources.
            selected_records=[]
            for r in records:
                if set(r.axes)!=set(required_axes(cfg.signals)):
                    prepared=load_record(r.path,cfg.data,r.label,signals=cfg.signals)
                    if prepared.diagnostics['direction_quality']!=r.diagnostics['direction_quality']:raise ValueError('通道更改影响了审查指标，请重新审查完整批次')
                    prepared.audit=copy.deepcopy(r.audit);prepared.quality=r.quality;prepared.quality_notes=r.quality_notes.copy();r=prepared
                selected_records.append(r)
            records=selected_records
        if request.get('snapshot') and any(set(r.axes)!=set(required_axes(cfg.signals)) for r in records):raise ValueError('快照通道/有效点协议已冻结，更改输入范围需重新导入原始CSV建立新版本')
        # Fail before creating a task if any selected input is unavailable.
        for signal in cfg.signals:prepare_trials(records,signal)
        protocol=protocol_for(records,cfg.data)
        if request.get('protocol') is not None and request['protocol']!=protocol:raise ValueError('扫描协议与导入快照不一致')
        from .folds import make_folds
        if not request.get('snapshot'):snapshot=dict(folds=make_folds(records,cfg.split_seed))
        if request.get('snapshot') and Path(request['output']).resolve().is_relative_to(Path(request['snapshot']).resolve()):raise ValueError('成果目录不能放在输入快照目录内')
        task=new_task(request['output'],prefix='CLOTH5')
        if request.get('snapshot'):
            import shutil
            shutil.copytree(request['snapshot'],task/'dataset')
            records,snapshot=load_dataset(task/'dataset')
        else:
            save_dataset(task/'dataset',records,cfg.data,cfg.split_seed,dict(request.get('screening',{}),selected_import=screening,review_batches=admission['batches']),snapshot['folds'],signals=cfg.signals,admission=admission);records,snapshot=load_dataset(task/'dataset')
        screening=snapshot['screening']
        meta=dict(format=FORMAT,id=task.name,experiment_id=uuid.uuid4().hex,dataset_id=snapshot['dataset_id'],name=cfg.name or task.name,created_at=datetime.now().isoformat(),status='running',stage='classification',purpose='quick' if cfg.preset=='quick' else 'formal',
                  labels=sorted({r.label for r in records}),records=len(records),trials=len(records)*2,config=cfg.to_dict(),sources=snapshot['sources'],partitions=[fold_partition(snapshot['folds'],i) for i in range(5)],folds=snapshot['folds'],protocol=protocol,models={},summaries={},stage2_status='not_available',screening=screening,
                  admission=snapshot['admission'],versions=dict(app=VERSION,python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__))
        write_json(task/'execution.json',dict(config=cfg.to_dict(),dataset_sha256=file_hash(task/'dataset/dataset.json')))
        meta['execution_sha256']=file_hash(task/'execution.json');write_json(task/'metadata.json',meta);write_json(task/'sources.json',meta['sources']);write_json(task/'partitions.json',meta['partitions'])
    def report(e):
        e=dict(e,task=str(task))
        with (task/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(e,ensure_ascii=False)+'\n')
        emit(e)
    try:
        meta['training_runtime']=NUMERICAL_VERSION;write_json(task/'metadata.json',meta)
        labels=meta['labels'];torch.set_num_threads(cfg.threads);torch.use_deterministic_algorithms(True);jobs=_jobs(cfg)
        if meta.get('validation_kind')=='runtime_upgrade':
            jobs=[j for j in jobs if f'{j[0]}_{j[1]}_{j[2]}_s{cfg.seed+j[3]}_f{j[4]+1}'==meta['validation_target_key']]
        for ji,(architecture,signal,readout,repetition,fold) in enumerate(jobs,1):
            check_cancel(cancelled);seed=cfg.seed+repetition;configuration_id=f'{architecture}_{signal}_{readout}';key=f'{configuration_id}_s{seed}_f{fold+1}'
            if key in meta['models']:continue
            folder=task/key;folder.mkdir(exist_ok=True);model_id=uuid.uuid4().hex
            def event(e):report(dict(e,model=key,fold=fold,job=ji,jobs=len(jobs),epochs=cfg.epochs))
            event(dict(type='phase',message=f'{ARCHITECTURES[architecture]} · {input_label(signal,FORMAT)} · 第{fold+1}/5折 · 复验{repetition+1}'))
            partition=meta['partitions'][fold]
            train=prepare_trials([records[i] for i in partition['train']],signal);val=prepare_trials([records[i] for i in partition['val']],signal)
            job_binding=dict(dataset_id=meta['dataset_id'],execution_sha256=meta['execution_sha256'],key=key,fold=fold)
            model,scaler,history,best_epoch=_fit(train,val,labels,cfg,architecture,readout,seed,event,cancelled,folder,job_binding=job_binding,allow_runtime_upgrade=meta.get('validation_kind')=='runtime_upgrade')
            check_cancel(cancelled);inference_start=time.monotonic();phase='final_validation_evaluation'
            try:
                vr,_=evaluate(model,val,scaler,labels,cancelled);inference_seconds=time.monotonic()-inference_start
                phase='final_train_evaluation';tr,_=evaluate(model,train,scaler,labels,cancelled)
            except NumericalFailure as exc:
                context=dict(job=job_binding,architecture=architecture,readout=readout,seed=seed,epoch=len(history),scaler=scaler,
                             recovery_epoch=len(history),phase=phase,gradient_norm=None)
                raise ValueError(_record_numerical_failure(folder,exc,context,model,event)) from exc
            bundle=dict(format=FORMAT,version=VERSION,model_id=model_id,dataset_id=meta['dataset_id'],fold=fold,architecture=architecture,signal=signal,readout=readout,seed=seed,channels=train[0].values.shape[1],labels=labels,
                        config=cfg.to_dict(),protocol=meta['protocol'],scaler=scaler,state_dict=model.state_dict(),sources=meta['sources'])
            torch.save(bundle,folder/'model.pt')
            result=dict(status='completed',stage='classification',model_id=model_id,configuration_id=configuration_id,architecture=architecture,signal=signal,readout=readout,seed=seed,fold=fold,repetition=repetition+1,
                        best_epoch=best_epoch,history=history,train_metrics=classification_metrics(tr,labels),val_metrics=classification_metrics(vr,labels),val_predictions=vr,
                        val_direction_metrics={d:classification_metrics([x for x in vr if x['direction']==d],labels) for d in ('out','back')},validation_inference_seconds=inference_seconds,
                        test_metrics=None,test_predictions=[],test_status='not_tested',parameters=sum(p.numel() for p in model.parameters()),training_seconds=sum(row['seconds'] for row in history),training_runtime=NUMERICAL_VERSION,
                        weight_sha256=file_hash(folder/'model.pt'),stage2_status='not_available')
            write_json(folder/'result.json',result);meta['models'][key]=result;write_json(task/'metadata.json',meta)
            event(dict(type='model_done',message=f'{ARCHITECTURES[architecture]} · {input_label(signal,FORMAT)} · 第{fold+1}折已保存；折外测试未运行'))
        check_cancel(cancelled);meta.update(status='completed',completed_at=datetime.now().isoformat());write_json(task/'metadata.json',meta)
        write_report(task,meta);report(dict(type='completed',message='修复续训验证完成，仅当前折；正式比较请新建统一版本实验' if meta.get('validation_kind')=='runtime_upgrade' else '五折分类训练完成；请查看验证结果，再运行预定候选的折外测试'));return task
    except Exception as exc:
        meta.update(status='cancelled' if isinstance(exc,Cancelled) else 'failed',error=str(exc));write_json(task/'metadata.json',meta)
        report(dict(type=meta['status'],message=str(exc)));raise

def load_bundle(path,allow_quick=False):
    path=Path(path).resolve();meta=read_json(path.parent.parent/'metadata.json')
    if meta.get('format')=='cloth-classification-v4':
        from .modern_engine import load_bundle as modern_load
        return modern_load(path,allow_quick)
    if meta['status']!='completed':raise ValueError('训练尚未完成，不能加载为正式模型')
    if (meta.get('purpose')=='quick' or meta.get('validation_kind')) and not allow_quick:raise ValueError('快速检查为试跑，请使用正式训练成果识别')
    key=path.parent.name
    if meta.get('format') in (FORMAT,'cloth-classification-v2'):
        r=meta['models'].get(key)
        if not r or r.get('status')!='completed' or file_hash(path)!=r['weight_sha256']:raise ValueError('权重完整性校验失败')
        b=torch.load(path,map_location='cpu',weights_only=True)
        if b.get('format')!=meta['format'] or b['model_id']!=r['model_id']:raise ValueError('模型格式或关联标识不匹配')
        if b['format']==FORMAT:
            if b['fold']!=r['fold'] or b['dataset_id']!=meta['dataset_id'] or b['seed']!=r['seed'] or b['architecture']!=r['architecture'] or b['signal']!=r['signal'] or b['readout']!=r['readout']:raise ValueError('权重的折/种子/数据版本关联不匹配')
            if json.loads(json.dumps(b['config']))!=meta['config']:raise ValueError('权重与冻结配置关联不匹配')
        cfg=TrainConfig.from_dict(b['config']);model=SegmentClassifier(b['channels'],len(b['labels']),b['architecture'],b['readout'],cfg.hidden_size,cfg.dropout)
    elif meta.get('format')=='cera-net-v1':
        from .model import CERANet
        state=read_json(path.parent/'calibration.json')
        if file_hash(path)!=state['weight_sha256']:raise ValueError('旧版权重与校准状态不匹配')
        b=torch.load(path,map_location='cpu',weights_only=True)
        if b.get('format')!='cera-net-v1':raise ValueError('旧模型格式不匹配')
        model=CERANet(b['channels'],len(b['labels']),b['variant']);b['_calibration']=state
    else:raise ValueError('不支持的模型格式')
    model.load_state_dict(b['state_dict']);model.eval();b['_model']=model
    return b

def test_request(request,emit=lambda e:None,cancelled=lambda:False):
    task=Path(request['task']).resolve()
    if read_json(task/'metadata.json').get('format')=='cloth-classification-v4':
        from .modern_engine import test_request as modern_test
        return modern_test(request,emit,cancelled)
    meta,cfg,records=_frozen_task(task)
    if meta.get('format')!=FORMAT or meta['status']!='completed':raise ValueError('请选择已完成的新版五折分类实验')
    if meta.get('purpose')=='quick' or meta.get('validation_kind'):raise ValueError('快速检查或修复验证不能作为正式测试成果')
    keys=request.get('keys') or list(meta['models'])
    if not keys or set(keys)-meta['models'].keys():raise ValueError('测试候选模型不存在')
    # A configuration is pre-registered as a complete set of folds/repetitions.
    selected={meta['models'][k]['configuration_id'] for k in keys}
    keys=[k for k,r in meta['models'].items() if r['configuration_id'] in selected]
    if meta.get('test_selection') is not None and set(keys)!=set(meta['test_selection']):raise ValueError('测试候选已锁定，不能查看后追加其他配置')
    for key in keys:load_bundle(task/key/'model.pt')
    meta['test_selection']=keys;meta['test_started_at']=meta.get('test_started_at',datetime.now().isoformat());write_json(task/'metadata.json',meta)
    torch.set_num_threads(cfg.threads)
    for i,key in enumerate(keys,1):
        check_cancel(cancelled);r=meta['models'][key]
        if r['test_metrics'] is not None:continue
        b=load_bundle(task/key/'model.pt');trials=prepare_trials([records[j] for j in meta['partitions'][r['fold']]['test']],b['signal']);start=time.monotonic()
        rows,_=evaluate(b['_model'],trials,b['scaler'],b['labels'],cancelled)
        r.update(test_metrics=classification_metrics(rows,b['labels']),test_predictions=rows,test_status='completed',test_seconds=time.monotonic()-start,
                 direction_metrics={d:classification_metrics([x for x in rows if x['direction']==d],b['labels']) for d in ('out','back')})
        write_json(task/key/'result.json',r);write_json(task/'metadata.json',meta)
        emit(dict(type='test_progress',task=str(task),message=f'折外测试 {i}/{len(keys)} · {key}'))
    meta['summaries']=summarize_configs(meta);meta['tested_at']=datetime.now().isoformat();write_json(task/'metadata.json',meta);write_report(task,meta)
    emit(dict(type='test_done',task=str(task),message='折外测试完成；每组恰好一次折外预测，勿再据此调整参数'));return task

def predict_request(request,emit=lambda e:None,cancelled=lambda:False):
    results=[]
    for path in request['models']:
        check_cancel(cancelled);b=load_bundle(path);data=dict(b['config']['data']);data.setdefault('preprocess_version','legacy-v2');cfg=DataConfig.from_dict(data);torch.set_num_threads(b['config'].get('threads',2))
        records,errors=import_records(request['paths'],cfg,request.get('overrides'),signals=(b['signal'],));accepted=[]
        for r in records:
            if matches_protocol(r,b['protocol']):
                if not r.label_confirmed:r.label=''
                accepted.append(r)
            else:errors.append(dict(path=r.path,message=f'扫描协议不匹配：模型固定{b["protocol"]["speed"]:g}mm/s及各方向时长'))
        trials=prepare_trials(accepted,b['signal'])
        if b['format']=='cloth-classification-v4':
            from .modern_training import evaluate as modern_evaluate
            from .modern_config import ModernConfig
            from dataclasses import replace
            rows=modern_evaluate(b['_model'],trials,b['scaler'],b['labels'],replace(ModernConfig.from_dict(b['config']),device='cpu',precision='fp32',workers=0),cancelled)[0]
        elif b['format'] in (FORMAT,'cloth-classification-v2'):rows=evaluate(b['_model'],trials,b['scaler'],b['labels'],cancelled)[0]
        else:
            from .legacy import evaluate_legacy
            rows=evaluate_legacy(b,trials,cancelled)
        hashes={s['sha256'] for s in b['sources']};groups={s['group'] for s in b['sources']}
        for row in rows:row['training_overlap']=row['sha256'] in hashes or row['group'] in groups
        results.append(dict(model=str(Path(path).resolve()),model_id=b.get('model_id') or file_hash(path)[:16],weight_sha256=file_hash(path),experiment_name=read_json(Path(path).resolve().parent.parent/'metadata.json')['name'],fold=b.get('fold'),seed=b.get('seed'),format=b['format'],signal=b['signal'],architecture=b.get('architecture','旧版CERA'),readout=b.get('readout'),labels=b['labels'],predictions=rows,errors=errors,evaluation=recognition_summary(rows,b['labels'])))
        emit(dict(type='prediction_progress',message=input_label(b['signal'],b['format'])+'识别完成'))
    dest=Path(request['destination']);write_json(dest,dict(format='cloth-predictions-v3',created_at=datetime.now().isoformat(),
        evaluation_kind='independent' if request.get('independent') else 'recognition',input_issues=request.get('input_issues',[]),results=results))
    emit(dict(type='prediction_done',destination=str(dest),message='完整扫描识别完成'));return dest

def write_report(task,meta):
    lines=['# 布料五折整段分类报告','',f'实验：{meta["name"]}',f'数据集：{meta["dataset_id"]}；{len(meta["labels"])}类，{meta["records"]} CSV，{meta["trials"]}单程。',
           '每组一次折外测试；每轮约60%训练/20%验证/20%测试。每折scaler仅用训练部分。',
           '仅完整段交叉熵，无风险头/前缀损失。历史质量提示等权纳入，不输入网络。',
           '验证选权重，折外测试不选超参。若依据折外结果改进，最终结论需新独立数据确认。','']
    if meta.get('validation_kind')=='runtime_upgrade':lines+=['**修复续训验证：仅验证当前未完成折，非完整五折正式成果。**',f'来源实验：{meta["parent_task"]}',f'数值处理：{meta["runtime_upgrade"]["from"]} → {NUMERICAL_VERSION}','']
    elif meta.get('purpose')=='quick':lines+=['**快速检查：非正式成果。**','']
    lines+=['| 模型 | 输入 | 折 | 种子 | 最佳轮 | 验证F1 | 测试准确率 |','| --- | --- | --- | --- | --- | --- | --- |']
    for r in meta['models'].values():
        score='未测试' if r['test_metrics'] is None else f'{r["test_metrics"]["accuracy"]:.2%}'
        signal=input_label(r['signal'],FORMAT).replace('|',r'\|')
        lines.append(f'| {r["architecture"].upper()} | {signal} | {r["fold"]+1} | {r["seed"]} | {r["best_epoch"]} | {r["val_metrics"]["macro_f1"]:.4f} | {score} |')
    for key,summary in meta.get('summaries',{}).items():
        for seed,r in summary['by_seed'].items():
            m=r['oof_metrics'];f=r['fold_statistics'];lines+=['',f'## {key} · 种子{seed}',f'合并折外准确率 {m["accuracy"]:.2%}；Macro-F1 {m["macro_f1"]:.4f}；{m["correct"]}/{m["total"]}单程。',
                f'逐折准确率均值±标准差 {f["accuracy_mean"]:.2%} ±{f["accuracy_std"]:.2%}；逐折F1 {f["f1_mean"]:.4f} ±{f["f1_std"]:.4f}。',
                '完整混淆矩阵、逐类/方向/质量提示分项、混淆对和训练验证差距见汇总.json。']
    lines+=['','五折是评价流程；识别需选择具体折/种子权重，不能加载平均成绩。全数据部署重训需另建版本和新数据验证。','第二阶段尚未开放。']
    (task/'报告.md').write_text('\n'.join(lines),encoding='utf-8');write_json(task/'汇总.json',meta.get('summaries',{}));write_json(task/'导入筛查.json',meta.get('screening',{}))
    with (task/'分类预测.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['模型','输入','折','种子','分区','文件','方向','质量提示','真实布料','预测布料'])
        for key,r in meta['models'].items():
            for partition,rows in [('验证',r['val_predictions']),('折外测试',r['test_predictions'])]:
                for row in rows:w.writerow([r['architecture'],input_label(r['signal'],FORMAT),r['fold']+1,r['seed'],partition,row['path'],row['direction'],row.get('quality','normal'),row['label'],meta['labels'][row['predicted']]])


def final_request(request,emit=lambda e:None,cancelled=lambda:False):
    from .modern_engine import final_request as modern_final
    return modern_final(request,emit,cancelled)
