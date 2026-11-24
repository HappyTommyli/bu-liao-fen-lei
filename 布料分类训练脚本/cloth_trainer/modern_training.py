"""GPU-first whole-segment training. Effective batches, atomic exact recovery."""
from pathlib import Path
import copy
from dataclasses import replace
import hashlib
import json
import time
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader
from .devices import resolve_device,diagnostics,seed_runtime,rng_state,restore_rng
from .modern_models import ModernClassifier
from .recovery import save_recovery,load_recovery
from .numerical import NumericalFailure
from .evaluation import classification_metrics
from .storage import write_json


class CachedTrials(Dataset):
    def __init__(self,trials,scaler,labels):
        self.trials=trials
        self.values=[torch.from_numpy(((t.values-np.asarray(scaler['mean']))/np.asarray(scaler['std'])).astype(np.float32)) for t in trials]
        self.targets=[labels.index(t.label) if t.label in labels else -1 for t in trials]
        finite(self.values,'input')
    def __len__(self):return len(self.values)
    def __getitem__(self,i):return self.values[i],self.targets[i],i


def collate(items):
    xs,ys,ids=zip(*items)
    return pad_sequence(xs,batch_first=True),torch.tensor(ys),torch.tensor([len(x) for x in xs]),ids


def finite(value,stage):
    """One synchronization per device; detailed traversal only on failure."""
    groups={}
    def visit(v):
        if isinstance(v,torch.Tensor):groups.setdefault(v.device,[]).append(v)
        elif isinstance(v,dict):
            for a in v.values():visit(a)
        elif isinstance(v,(tuple,list)):
            for a in v:visit(a)
    visit(value)
    if any(not bool(torch.stack([torch.isfinite(t).all() for t in ts]).all()) for ts in groups.values()):
        from .recovery import nonfinite_tensors
        raise NumericalFailure(stage,nonfinite_tensors=nonfinite_tensors(value))


@torch.no_grad()
def clip(model,max_norm=1.):
    gs=[p.grad for p in model.parameters() if p.grad is not None];finite(gs,'gradients')
    norm=torch.stack([g.double().norm() for g in gs]).norm();maximum=torch.stack([g.abs().max().double() for g in gs]).max()
    coefficient=(max_norm/(norm+1e-6)).clamp(max=1.)
    for g in gs:g.copy_(g.double()*coefficient)
    clipped=torch.stack([g.double().norm() for g in gs]).norm()
    values=torch.stack([norm,maximum,coefficient,clipped]).cpu().tolist()
    if not np.isfinite(values).all() or values[3]>max_norm+1e-6:raise NumericalFailure('gradient_norm')
    return dict(norm=values[0],max_abs=values[1],coefficient=values[2],clipped_norm=values[3],clipped=values[0]>max_norm)


def context(cfg,device):
    return nullcontext() if cfg.precision=='fp32' else torch.autocast(device_type=device.type,dtype=torch.float16 if cfg.precision=='amp_fp16' else torch.bfloat16)


def loader(dataset,cfg,generator=None,shuffle=False):
    return DataLoader(dataset,batch_size=cfg.batch_size,shuffle=shuffle,num_workers=cfg.workers,pin_memory=cfg.device=='cuda',
                      collate_fn=collate,generator=generator,persistent_workers=False)


def evaluate(model,trials,scaler,labels,cfg,cancelled=lambda:False,candidate=None):
    candidate=candidate or getattr(model,'candidate',None)
    if candidate is not None:
        return evaluate_candidate(model,trials,scaler,labels,cfg,cancelled,candidate)
    from .engine import check_cancel
    model.eval();rows=[];total=0.;count=0;device=next(model.parameters()).device
    data=CachedTrials(trials,scaler,labels)
    with torch.inference_mode():
        for x,y,lengths,ids in loader(data,cfg):
            check_cancel(cancelled)
            micro=cfg.micro_batch or cfg.batch_size
            for start in range(0,len(y),micro):
                xx=x[start:start+micro].to(device,non_blocking=True);yy=y[start:start+micro].to(device,non_blocking=True)
                with context(cfg,device):logits=model(xx,lengths[start:start+micro])
                finite(logits,'validation_logits');p=logits.float().softmax(-1).cpu().numpy();known=yy>=0
                if bool(known.any()):total+=float(F.cross_entropy(logits[known].float(),yy[known],reduction='sum'));count+=int(known.sum())
                for j,i in enumerate(ids[start:start+micro]):
                    t=trials[i];rows.append(dict(id=t.id,path=t.path,sha256=t.sha256,group=t.group,label=t.label,true=data.targets[i],direction=t.direction,
                        predicted=int(p[j].argmax()),scores=p[j].tolist(),elapsed=(len(t.values)-1)/333,quality=getattr(t,'quality','normal')))
    return rows,total/count if count else None


def probe_microbatch(model,cfg,device,max_length,channels,generator,amp_factory=None):
    """Each OOM retry starts with clean weights, optimizer and scaler state."""
    initial=copy.deepcopy(model.state_dict());initial_rng=rng_state(generator)
    micro=cfg.micro_batch or cfg.batch_size
    amp_factory=amp_factory or (lambda:torch.amp.GradScaler('cuda',enabled=cfg.precision=='amp_fp16'))
    def attempt(size):
        # Locals release the failed graph and any partial optimizer allocation
        # when this frame unwinds, before empty_cache and the smaller attempt.
        model.load_state_dict(initial);restore_rng(initial_rng,generator)
        optim=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay);amp=amp_factory()
        optim.zero_grad(set_to_none=True)
        xx=torch.zeros(size,max_length,channels,device=device)
        with context(cfg,device):loss=model(xx,torch.full((size,),max_length,dtype=torch.long)).float().square().mean()
        amp.scale(loss).backward();amp.unscale_(optim);clip(model,max_norm=model.grad_clip_norm) if hasattr(model,'grad_clip_norm') else clip(model);amp.step(optim);amp.update()
        if device.type=='cuda':torch.cuda.synchronize()
    try:
        while True:
            try:attempt(micro);return micro
            except torch.OutOfMemoryError:
                if cfg.micro_batch or micro==1:raise ValueError('最长序列显存预检失败；请降低微批或选择压缩对照') from None
                micro=max(1,micro//2);model.zero_grad(set_to_none=True)
                if device.type=='cuda':torch.cuda.empty_cache()
    finally:
        model.load_state_dict(initial);model.zero_grad(set_to_none=True);restore_rng(initial_rng,generator)


def fit(train,val,labels,cfg,architecture,readout,seed,folder,emit=lambda e:None,cancelled=lambda:False,job_binding=None,fixed_epochs=None,candidate=None,frozen_config=None):
    from .engine import fit_scaler,check_cancel,_record_numerical_failure
    cfg.validate();device=resolve_device(cfg.device);seed_runtime(seed,cfg.threads)
    if frozen_config is not None:
        from .modern_config import ModernConfig
        if ModernConfig.from_dict(frozen_config).to_dict()!=cfg.to_dict():raise ValueError('恢复点冻结配置与实际训练设置不一致')
    if candidate is None and cfg.tcn_spec is not None:
        from .tcn_study import candidates
        options=candidates(cfg)
        if len(options)!=1:raise ValueError('四组实验必须明确选择当前TCN候选')
        candidate=options[0]
    if candidate is not None:
        from .tcn_study import validate_candidate
        candidate=validate_candidate(candidate)
    if cfg.precision=='amp_bf16' and not torch.cuda.is_bf16_supported():raise ValueError('当前GPU不支持BF16')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    runtime=dict(device=device.type,precision=cfg.precision,**diagnostics(),threads=cfg.threads,deterministic=True)
    def signature(tt):return [dict(id=t.id,label=t.label,group=t.group,hash=hashlib.sha256(t.values.tobytes()).hexdigest()) for t in tt]
    binding=dict(runtime=copy.deepcopy(runtime),config=json.loads(json.dumps(cfg.to_dict() if frozen_config is None else frozen_config)),architecture=architecture,readout=readout,seed=seed,
                 labels=labels,train=signature(train),val=signature(val),job=job_binding,fixed_epochs=fixed_epochs)
    if candidate is not None:binding['candidate']=candidate
    scaler=fit_scaler(train);finite([torch.from_numpy(t.values) for t in train+val],'input')
    data=CachedTrials(train,scaler,labels);generator=torch.Generator().manual_seed(seed)
    model=make_model(cfg,train[0].values.shape[1],len(labels),architecture,readout,candidate).to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,factor=cfg.scheduler_factor if candidate is not None else .5,patience=cfg.scheduler_patience,min_lr=cfg.min_lr) if cfg.scheduler and val else None
    amp=torch.amp.GradScaler('cuda',enabled=cfg.precision=='amp_fp16')
    epochs=fixed_epochs or cfg.epochs;history=[];best=(-1.,-1e300);best_state=None;best_epoch=0;completed=0;stale=0;min_loss=1e300;finished=False
    saved=load_recovery(folder,binding)
    micro=cfg.micro_batch or cfg.batch_size
    if saved:
        model.load_state_dict(saved['model']);optim.load_state_dict(saved['optimizer']);amp.load_state_dict(saved['amp'])
        if scheduler:scheduler.load_state_dict(saved['scheduler'])
        history=saved['history'];best=tuple(saved['best']);best_state=saved['best_state'];best_epoch=saved['best_epoch'];completed=saved['epoch']
        stale=saved['stale'];min_loss=saved['min_loss'];finished=saved['finished'];micro=saved['micro_batch'];restore_rng(saved['rng'],generator)
        emit(dict(type='restored',epoch=completed,history=copy.deepcopy(history),message=f'已恢复第{completed}轮（设备和精度保持冻结）'))
    elif device.type=='cuda':
        micro=probe_microbatch(model,cfg,device,min(max(len(x) for x in data.values),candidate['window_size']) if candidate and candidate['window_size'] and not candidate.get('prefix_policy') else max(len(x) for x in data.values),train[0].values.shape[1],generator)
    runtime['micro_batch']=micro
    if candidate is not None and device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    def checkpoint():
        save_recovery(folder,dict(model=model.state_dict(),optimizer=optim.state_dict(),scheduler=scheduler.state_dict() if scheduler else None,
            amp=amp.state_dict(),scaler=scaler,rng=rng_state(generator),history=history,best=best,best_state=best_state,best_epoch=best_epoch,
            epoch=completed,stale=stale,min_loss=min_loss,finished=finished,micro_batch=micro),binding)
    if saved is None:checkpoint()
    epoch=completed;batch_index=None
    try:
        for epoch in range(completed+1,epochs+1) if not finished else ():
            check_cancel(cancelled);model.train();total=0.;correct=0;stats=[];start_time=time.monotonic()
            for batch_index,(x,y,lengths,ids) in enumerate(loader(data,cfg,generator,True),1):
                check_cancel(cancelled);optim.zero_grad(set_to_none=True)
                if candidate is not None:
                    from .tcn_study import split_candidate,aggregate_log_probs
                    segments,lens,parents,weights,_=split_candidate(x,lengths,candidate,generator=generator)
                    detached=[]
                    for start in range(0,len(segments),micro):
                        check_cancel(cancelled)
                        stop=start+micro
                        xx=segments[start:stop].to(device,non_blocking=True)
                        yy=y[parents[start:stop]].to(device,non_blocking=True)
                        with context(cfg,device):
                            logits=model(xx,lens[start:stop])
                            per_window=F.cross_entropy(logits.float(),yy,reduction='none')
                            loss=(per_window*weights[start:stop].to(device)).sum()/len(y)
                        finite([logits,loss],'loss');amp.scale(loss).backward()
                        total+=float(loss.detach())*len(y)
                        detached.append(logits.detach().float().cpu())
                    parent_logp=aggregate_log_probs(torch.cat(detached),parents,weights,len(y))
                    correct+=int((parent_logp.argmax(-1)==y).sum())
                else:
                    for start in range(0,len(y),micro):
                        xx=x[start:start+micro].to(device,non_blocking=True);yy=y[start:start+micro].to(device,non_blocking=True)
                        with context(cfg,device):logits=model(xx,lengths[start:start+micro]);loss=F.cross_entropy(logits.float(),yy,reduction='sum')/len(y)
                        finite([logits,loss],'loss');amp.scale(loss).backward()
                        total+=float(loss.detach())*len(y);correct+=int((logits.detach().argmax(-1)==yy).sum())
                amp.unscale_(optim);stats.append(clip(model,max_norm=cfg.grad_clip_norm) if candidate is not None else clip(model));amp.step(optim);amp.update()
                finite(dict(model=model.state_dict(),optimizer=optim.state_dict()),'optimizer')
            vr,vloss=evaluate(model,val,scaler,labels,replace(cfg,micro_batch=micro),cancelled) if val else ([],None)
            vm=classification_metrics(vr,labels) if val else dict(macro_f1=None,accuracy=None)
            previous_lr=optim.param_groups[0]['lr']
            if val:
                score=(vm['macro_f1'],-vloss)
                if score>best:best=score;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
                stale=0 if vloss<min_loss-1e-4 else stale+1;min_loss=min(min_loss,vloss)
                if scheduler:scheduler.step(vloss)
            else:best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
            row=dict(epoch=epoch,loss=total/len(train),train_accuracy=correct/len(train),val_loss=vloss,val_f1=vm['macro_f1'],val_accuracy=vm['accuracy'],
                best_f1=best[0] if val else None,best_epoch=best_epoch,
                lr=optim.param_groups[0]['lr'],seconds=time.monotonic()-start_time,gradient_norm_max=max(s['norm'] for s in stats),
                gradient_abs_max=max(s['max_abs'] for s in stats),gradient_clip_fraction=sum(s['clipped'] for s in stats)/len(stats),
                gradient_clipped_norm_max=max(s['clipped_norm'] for s in stats),gradient_clip_coefficient_min=min(s['coefficient'] for s in stats))
            if candidate is not None:
                steps=(len(train)+cfg.batch_size-1)//cfg.batch_size
                row.update(optimizer_updates=epoch*steps,effective_points=epoch*sum(len(t.values) for t in train),
                           loss_kind='length-weighted window CE',validation_loss_kind='parent NLL',
                           gpu_peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0)
            history.append(row);completed=epoch;finished=epoch>=epochs or bool(val and cfg.early_stopping and stale>=cfg.patience)
            checkpoint();write_json(folder/'history.json',history)
            metrics=(f'训练损失 {row["loss"]:.4f}｜验证损失 {vloss:.4f}｜验证宏F1 {vm["macro_f1"]:.2%}｜验证准确率 {vm["accuracy"]:.2%}｜本折最佳F1 {best[0]:.2%}（第{best_epoch}轮）'
                     if val else f'训练损失 {row["loss"]:.4f}｜验证F1不适用（全部开发集重训）')
            emit(dict(type='epoch',**row,message=f'第{epoch}/{epochs}轮 · {metrics}｜学习率 {row["lr"]:.6g} · {device.type.upper()} · 微批{micro}'))
            if row['lr']!=previous_lr:emit(dict(type='lr_changed',epoch=epoch,previous_lr=previous_lr,lr=row['lr'],message=f'学习率下降：{previous_lr:.6g} → {row["lr"]:.6g}'))
            if finished:break
    except NumericalFailure as exc:
        raise ValueError(_record_numerical_failure(folder,exc,dict(epoch=epoch,batch=batch_index,recovery_epoch=completed,runtime=runtime),model,emit)) from exc
    model.load_state_dict(best_state);model.eval()
    reason='fixed_epochs' if fixed_epochs else 'max_epochs' if completed>=epochs else 'early_stopping'
    explanation={'fixed_epochs':'完成预定的最终重训轮数；验证F1不适用','max_epochs':'达到轮数上限','early_stopping':f'早停：验证损失连续{cfg.patience}轮未达到改善要求'}[reason]
    emit(dict(type='fit_done',epoch=completed,best_epoch=best_epoch,best_f1=best[0] if val else None,stop_reason=reason,
              message=explanation+(f'｜最佳验证宏F1 {best[0]:.2%}（第{best_epoch}轮）' if val else '')))
    return dict(model=model,scaler=scaler,history=history,best_epoch=best_epoch,runtime=runtime,stop_reason=reason)



def make_model(cfg,channels,classes,architecture,readout,candidate=None):
    if candidate is None:
        return ModernClassifier(channels,classes,architecture,readout,cfg.hidden_size,cfg.dropout)
    from .tcn_models import TCNClassifier
    from .tcn_study import validate_candidate
    candidate=validate_candidate(candidate)
    from .cera_models import CERAClassifier,CERA_REVISIONS
    expected='cera' if candidate['spec']['revision'] in CERA_REVISIONS else 'tcn'
    if architecture!=expected:raise ValueError('窗口候选与模型架构不匹配')
    model=(CERAClassifier if expected=='cera' else TCNClassifier)(channels,classes,candidate['spec'])
    model.candidate=candidate
    model.grad_clip_norm=cfg.grad_clip_norm
    return model


def evaluate_candidate(model,trials,scaler,labels,cfg,cancelled,candidate):
    from .engine import check_cancel
    from .tcn_study import split_candidate,aggregate_log_probs,validate_candidate
    candidate=validate_candidate(candidate)
    model.eval();rows=[];total=0.;count=0;device=next(model.parameters()).device
    data=CachedTrials(trials,scaler,labels)
    with torch.inference_mode():
        for x,y,lengths,ids in loader(data,cfg):
            check_cancel(cancelled)
            segments,lens,parents,weights,starts=split_candidate(x,lengths,candidate)
            outputs=[];micro=cfg.micro_batch or cfg.batch_size
            for start in range(0,len(segments),micro):
                check_cancel(cancelled)
                xx=segments[start:start+micro].to(device,non_blocking=True)
                with context(cfg,device):logits=model(xx,lens[start:start+micro])
                finite(logits,'validation_logits');outputs.append(logits.float().cpu())
            logits=torch.cat(outputs)
            if candidate.get('prefix_policy'):
                logp=torch.stack([logits[torch.where(parents==j)[0][-1]].log_softmax(-1) for j in range(len(y))])
            else:logp=aggregate_log_probs(logits,parents,weights,len(y))
            finite(logp,'validation_parent_log_probs');prob=logp.exp();wp=logits.softmax(-1)
            known=y>=0
            if bool(known.any()):
                total+=float(-logp[known,y[known]].sum());count+=int(known.sum())
            for j,index in enumerate(ids):
                t=trials[index];p=prob[j]
                windows=[dict(start=int(starts[k]),end=int(starts[k]+lens[k]),points=int(lens[k]),
                              scores=wp[k].tolist(),predicted=int(wp[k].argmax()))
                         for k in torch.where(parents==j)[0].tolist()]
                rows.append(dict(id=t.id,path=t.path,sha256=t.sha256,group=t.group,label=t.label,true=data.targets[index],
                    direction=t.direction,predicted=int(p.argmax()),scores=p.tolist(),elapsed=(len(t.values)-1)/333,
                    quality=getattr(t,'quality','normal'),windows=windows))
    return rows,total/count if count else None
