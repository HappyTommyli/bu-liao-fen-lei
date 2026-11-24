"""Fixed review batches and admission checks, independent of UI selection."""
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import json
import uuid
from .quality import RULE,RULE_VERSION,annotate,current_rule,digest


def error_entry(error,config):
    from .data import file_hash,metadata_hash
    e=deepcopy(error);p=Path(e['path']);meta={}
    try:meta=json.loads((p.parent/'metadata.json').read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):pass
    if not isinstance(meta,dict):meta={}
    try:e.setdefault('sha256',file_hash(p));e.setdefault('metadata_sha256',metadata_hash(p))
    except OSError:e.setdefault('sha256',None);e.setdefault('metadata_sha256',None)
    raw_label=meta.get('label','');label=raw_label.strip() if isinstance(raw_label,str) else ''
    e['label']=label;e.setdefault('label_confirmed',bool(label))
    context=meta.get('context');context=context if isinstance(context,dict) else {}
    session=context.get('capture_session');e.setdefault('group','capture:'+str(session) if session else 'csv:'+str(e.get('sha256') or p))
    e.setdefault('audit',dict(rule_version=RULE_VERSION,rule_sha256=current_rule()['sha256'],status='data_error',eligible=False,reasons=[e['message']]))
    e['audit'].setdefault('single_status',e['audit']['status']);e['audit'].setdefault('single_eligible',False)
    limits=dict(config.__dict__)
    if config.contact_rule=='fixed-x-v1':limits.pop('contact_rule')
    if config.sample_quality=='recorded-v1':limits.pop('sample_quality')
    e.setdefault('diagnostics',dict(integrity=dict(limits=limits,failure=e['message'],unverifiable=['无法仅凭记录证明撞击、离面等机械事件没有发生'])))
    if not isinstance(raw_label,str):e['diagnostics']['invalid_metadata_label']=repr(raw_label)
    return e


def class_summary(entries):
    classes=defaultdict(lambda:defaultdict(list))
    for e in entries:
        if e.get('label'):classes[e['label']][e['group']].append(e)
    result={}
    for label,groups in sorted(classes.items()):
        assessed=abnormal=unknown=mixed=0
        for members in groups.values():
            ds=[m.get('diagnostics',{}).get('direction_quality',{}) for m in members]
            if all(d.get('status')=='assessed' for d in ds) and all(m['audit']['single_status'] not in ('undetermined','data_error') and m.get('diagnostics',{}).get('by_verification',{}).get('status','verified')=='verified' for m in members):
                assessed+=1;flags=[bool(d.get('marked')) for d in ds]
                abnormal+=int(all(flags));mixed+=int(any(flags) and not all(flags))
            else:unknown+=1
        rate=abnormal/assessed if assessed else None
        result[label]=dict(independent_groups=len(groups),assessable_groups=assessed,abnormal_groups=abnormal,undetermined_groups=unknown,
                           mixed_groups=mixed,rate=rate,disabled=assessed>=RULE['minimum_class_groups'] and rate>=RULE['class_rate'],
                           review_small_batch=bool(assessed and rate>=RULE['class_rate'] and assessed<RULE['minimum_class_groups']))
    return result


def review_batch(records,errors,candidates,config,signals,overrides=None):
    batch_id=uuid.uuid4().hex
    for r in records:
        if not r.audit or r.audit.get('rule_version')!=RULE_VERSION:annotate(r)
        r.audit.update(status=r.audit['single_status'],eligible=r.audit['single_eligible'],batch_id=batch_id,class_disabled=False)
        r.quality=r.audit['status'];r.quality_notes=r.audit['reasons'].copy()
    normalized=[error_entry(e,config) for e in errors]
    for e in normalized:e['audit']['batch_id']=batch_id
    entries=[deepcopy(r.manifest()) for r in records]+normalized
    classes=class_summary(entries)
    for e in entries:
        c=classes.get(e.get('label'),{})
        if c.get('disabled'):
            e['audit'].update(eligible=False,class_disabled=True)
            if e['audit']['status'] in ('normal','suggest_review'):e['audit']['status']='excluded'
            e['audit']['reasons'].append(f'本批次该类暂不启用：大幅异常{c["abnormal_groups"]}/{c["assessable_groups"]}个可评估独立组')
    by_path={e['path']:e for e in entries}
    for r in records:
        r.audit=deepcopy(by_path[r.path]['audit']);r.quality=r.audit['status'];r.quality_notes=r.audit['reasons'].copy()
    for e in errors:
        if e['path'] in by_path:e.update(deepcopy(by_path[e['path']]))
    report=dict(format='cloth-review-batch-v1',batch_id=batch_id,created_at=datetime.now().isoformat(),rule=current_rule(),data=config.__dict__,
                signals=list(signals),candidates=[str(p) for p in candidates],overrides={str(p):overrides[str(p)] for p in candidates if str(p) in (overrides or {})},entries=entries,classes=classes)
    return dict(report,sha256=digest(report))


def package(batches):return dict(rule=current_rule(),batches=deepcopy(batches))


def validate_batches(batches,config=None):
    if not batches:raise ValueError('缺少当前审查批次，旧快照不能直接用于新训练；请从原CSV重新审查')
    seen=set()
    for b in batches:
        payload={k:v for k,v in b.items() if k!='sha256'}
        if b.get('sha256')!=digest(payload):raise ValueError('审查批次校验失败')
        if b.get('rule')!=current_rule():raise ValueError('审查规则版本不是当前版本，请重新审查原始CSV')
        from .config import DataConfig
        if config is not None and DataConfig.from_dict(b['data'])!=config:raise ValueError('审查采集协议与训练设置不一致')
        if b['batch_id'] in seen:raise ValueError('审查批次重复')
        seen.add(b['batch_id'])
        if b['classes']!=class_summary(b['entries']):raise ValueError('审查类别汇总不一致')


def require_admission(records,admission,config=None):
    if not admission or admission.get('rule')!=current_rule():raise ValueError('旧快照缺少当前审查规则，需重新导入原CSV后建立新版本')
    batches=admission.get('batches',[]);validate_batches(batches,config)
    by_id={b['batch_id']:{e['path']:e for e in b['entries']} for b in batches}
    for r in records:
        e=by_id.get(r.audit.get('batch_id'),{}).get(r.path)
        if not e or any(e.get(k)!=getattr(r,k) for k in ('sha256','metadata_sha256','label','group','label_confirmed')) or e['audit']!=r.audit:
            raise ValueError('记录与冻结审查结论不一致，请重新审查')
        if not r.audit.get('eligible') or r.audit['status'] not in ('normal','suggest_review'):raise ValueError('记录不符合训练准入：'+r.path)
    return records


def eligible_records(records):return [r for r in records if r.audit.get('eligible',False)]


def verify_batch_sources(batches,config,cancelled=lambda:False):
    from .data import file_hash,metadata_hash
    validate_batches(batches,config)
    for b in batches:
        for e in b['entries']:
            if cancelled():raise ValueError('已取消来源核验')
            try:sha=file_hash(e['path'])
            except OSError:sha=None
            try:md=metadata_hash(e['path'])
            except OSError:md=None
            if sha!=e.get('sha256') or md!=e.get('metadata_sha256'):raise ValueError('完整审查批次的来源在冻结前改变：'+e['path'])


def verify_raw_batches(batches,config):
    """Recheck the FULL fixed batches, never a user-selected denominator."""
    from .data import import_dataset
    validate_batches(batches,config);loaded={}
    for b in batches:
        overrides=b.get('overrides',{})
        records,report=import_dataset(b['candidates'],config,tuple(b['signals']),overrides)
        fresh=report['review_batches'][0]
        def comparable(entries):
            result=[]
            for e in entries:
                audit={k:v for k,v in e['audit'].items() if k!='batch_id'}
                result.append({**{k:e.get(k) for k in ('path','sha256','metadata_sha256','label','group','label_confirmed','diagnostics')},'audit':audit})
            return sorted(result,key=lambda e:e['path'])
        if comparable(fresh['entries'])!=comparable(b['entries']) or fresh['classes']!=b['classes']:
            raise ValueError('完整审查批次的来源或结论已改变，请重新导入审查')
        old={e['path']:e for e in b['entries']}
        for r in records:
            r.audit=deepcopy(old[r.path]['audit']);r.quality=r.audit['status'];r.quality_notes=r.audit['reasons'].copy();loaded[r.path]=r
    return list(loaded.values())
