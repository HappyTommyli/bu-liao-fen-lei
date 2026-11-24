"""Supplement fresh audit with measured gaps and raw direction statistics."""
import sys,json,csv,collections
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from cloth_trainer.data import file_hash
from cloth_trainer.quality import direction_metrics,QUALITY
DEST=Path(__file__).parent;bundle=json.loads((DEST/'当前完整审查批次.json').read_text());report=bundle['report'];batch=report['review_batches'][0]
entries={e['path']:e for e in batch['entries']};rows=[]
for k,path in enumerate(batch['candidates']):
    p=Path(path);e=entries.get(path)
    if not e:
        duplicate=next((x for x in report['duplicates'] if x['path']==path),None)
        if duplicate:
            rows.append(dict(path=path,record=p.stem,label='',status='duplicate',eligible=False,reasons=['重复CSV内容；原记录：'+duplicate['canonical']],action='去重·保留原记录'));continue
        raise ValueError('审查候选缺少结果：'+path)
    audit=e['audit'];dg=e.get('diagnostics',{});row=dict(path=path,record=p.stem,label=e.get('label',''),sha256=e.get('sha256'),metadata_sha256=e.get('metadata_sha256'),
        status=audit['status'],eligible=audit['eligible'],reasons=list(audit['reasons']),history=audit.get('history'),class_disabled=audit.get('class_disabled',False),
        primary_direction=dg.get('direction_quality'),stage_details={},supplement_alerts=[],sequence=None)
    rows.append(row)
    try:
        if file_hash(p)!=row['sha256'] or file_hash(p.parent/'metadata.json')!=row['metadata_sha256']:
            row.update(status='source_changed',eligible=False,reasons=['审查后CSV或metadata变化，本次暂不评估']);continue
        m=json.loads((p.parent/'metadata.json').read_text(encoding='utf-8-sig'));row.update(sequence=m.get('sequence'),outcome=m.get('outcome'),created_at=m.get('created_at'))
        with p.open(encoding='utf-8-sig',newline='')as stream:headers=next(csv.reader(stream))
        tk='sensor_t_s' if 'sensor_t_s'in headers else 'device_time_unwrapped_us'
        wanted=[tk,'stage','motor_x_pulses','sample_valid','Bx_uT','By_uT','Bz_uT','dBx_uT','dBy_uT','dBz_uT']
        names=[n for n in wanted if n in headers];arr=np.loadtxt(p,delimiter=',',skiprows=1,usecols=[headers.index(n)for n in names],dtype='U40',encoding='utf-8-sig',ndmin=2)
        col={name:arr[:,i]for i,name in enumerate(names)}
        settings=m.get('context',{}).get('config',{});ppmm=float(settings.get('x_ppmm',250));x=col['motor_x_pulses'].astype(float)/ppmm
        time=col[tk].astype(float)/(1e6 if tk.endswith('_us')else 1);stage=col['stage'];base=m.get('baseline_B0_uT',{});ctxbase=m.get('context',{}).get('baseline',{});basevec=ctxbase.get('baseline',[])
        sampling_valid=np.isin(np.char.lower(col.get('sample_valid',np.full(len(time),'1'))),['1','true','yes'])
        by=col.get('dBy_uT');by=by.astype(float) if by is not None else col['By_uT'].astype(float)-float(base.get('By_uT',basevec[1] if len(basevec)>1 else np.nan))
        valid=sampling_valid&np.isfinite(by)&np.isfinite(time)&np.isfinite(x)
        trials=[];all_contact=np.zeros(len(time),dtype=bool)
        for st,di,lo,hi in [('step_003','out',45.,142.),('step_004','back',16.,114.)]:
            within=(stage==st)&(x>=lo)&(x<=hi);all_contact|=within;use=within&valid;tt=time[use];xx=x[use];gaps=np.diff(tt)
            full=valid&(stage==st);fullx=x[full]
            detail=dict(full_stage_x_min=float(fullx.min())if len(fullx)else None,full_stage_x_max=float(fullx.max())if len(fullx)else None,
                full_stage_span=float(np.ptp(fullx))if len(fullx)else None,contact_samples=int(within.sum()),valid_samples=int(use.sum()),invalid_samples=int((within&~valid).sum()),gap_count_over20ms=int((gaps>.020000001).sum()),
                gap_count_over100ms=int((gaps>.100000001).sum()),time_not_increasing=int((gaps<=0).sum()),x_min=float(xx.min())if len(xx)else None,x_max=float(xx.max())if len(xx)else None,
                max_gap_ms=float(gaps.max()*1000)if len(gaps)else None,median_sample_hz=float(1/np.median(gaps))if len(gaps)and np.median(gaps)>0 else None)
            if len(gaps):
                pos=int(np.argmax(gaps));detail.update(max_gap_x_from=float(xx[pos]),max_gap_x_to=float(xx[pos+1]),max_gap_t_from=float(tt[pos]),max_gap_t_to=float(tt[pos+1]))
            row['stage_details'][di]=detail
            trials.append(SimpleNamespace(raw_values=by[:,None],raw_x=x,raw_valid=valid&(stage==st),bounds=(lo,hi),axes=('y',)))
        coverage=[]
        for di,lo,hi in [('out',45.,142.),('back',16.,114.)]:
            sd=row['stage_details'][di]
            if sd['x_min']is None or sd['x_min']>lo+2 or sd['x_max']<hi-2:coverage.append(di)
        row['coverage_issues']=coverage
        row['coordinate_reference']=m.get('context',{}).get('coordinate_reference')
        row['capture_start_mm']=m.get('context',{}).get('capture_start_mm')
        raw_record=SimpleNamespace(trials=trials,views={},axes=('y',),diagnostics={'baseline_sigma':ctxbase.get('sigma')})
        dm=direction_metrics(raw_record);row['direction']=dm
        if dm.get('bins',0)>=10:
            pairs=np.array(dm['bin_medians']);a,b=pairs.T
            row['relative_rms']=float(np.sqrt(np.mean((a-b)**2))/np.mean(pairs))
            row['centered_rms_uT']=float(np.sqrt(np.mean(((a-a.mean())-(b-b.mean()))**2)))
            row['shape_correlation']=float(np.corrcoef(a,b)[0,1])if min(a.std(),b.std())>1e-9 else None
        baseline_errors={}
        for axis in 'xyz':
            rawkey='B'+axis+'_uT';deltakey='dB'+axis+'_uT';ai='xyz'.index(axis);zero=base.get(rawkey,basevec[ai]if len(basevec)>ai else None)
            if rawkey not in col or deltakey not in col or zero is None:baseline_errors[axis]=None;continue
            raw=col[rawkey].astype(float);delta=col[deltakey].astype(float);good=all_contact&sampling_valid&np.isfinite(raw)&np.isfinite(delta)
            error=float(np.max(np.abs(delta[good]-(raw[good]-float(zero)))))if good.any()else None
            baseline_errors[axis]=error
            if error is not None and error>.010000001:row['supplement_alerts'].append(f'{rawkey}与差分/基线偏差{error:.4f}μT超过0.01μT')
            missing=int((all_contact&sampling_valid&(~np.isfinite(raw)|~np.isfinite(delta))).sum())
            if missing:row['supplement_alerts'].append(f'{axis.upper()}轴接触窗内{missing}条非有限原始/差分值')
        row['baseline_errors']=baseline_errors
        if coverage:
            aa,bb=row['stage_details']['out'],row['stage_details']['back'];bounds=[aa.get('full_stage_x_min'),aa.get('full_stage_x_max'),bb.get('full_stage_x_min'),bb.get('full_stage_x_max')]
            steps=m.get('context',{}).get('recipe',{}).get('steps',[]);moves=[float(s.get('value',0))for s in steps if s.get('kind')=='relative'and s.get('axis')=='X'and s.get('enabled',True)]
            intended=abs(moves[0])if moves else 155.
            if all(v is not None for v in bounds)and abs((bounds[1]-bounds[0])-intended)<=2 and abs((bounds[3]-bounds[2])-intended)<=2 and abs(bounds[0]-bounds[2])<=2:
                start=float((row.get('capture_start_mm')or{}).get('x',0));offset=(bounds[0]+bounds[2])/2-start
                row['coordinate_offset_candidate_mm']=float(offset)
                shifted=[SimpleNamespace(raw_values=tr.raw_values,raw_x=tr.raw_x-offset,raw_valid=tr.raw_valid,bounds=tr.bounds,axes=tr.axes)for tr in trials]
                candidate=SimpleNamespace(trials=shifted,views={},axes=('y',),diagnostics={'baseline_sigma':ctxbase.get('sigma')})
                row['conditional_aligned_direction']=direction_metrics(candidate)
                aligned_gaps={}
                for st,di,lo,hi in [('step_003','out',45.,142.),('step_004','back',16.,114.)]:
                    use=valid&(stage==st)&(x-offset>=lo)&(x-offset<=hi);dd=np.diff(time[use])
                    aligned_gaps[di]=float(dd.max()*1000)if len(dd)else None
                row['conditional_aligned_max_gap_ms']=aligned_gaps
        if file_hash(p)!=row['sha256'] or file_hash(p.parent/'metadata.json')!=row['metadata_sha256']:row.update(status='source_changed',eligible=False,reasons=['CSV或metadata在补查中发生变化，本次暂不评估'])
    except (ValueError,OSError,KeyError,IndexError,TypeError)as exc:row['supplement_failure']=str(exc)
    if (k+1)%50==0 or k+1==len(batch['candidates']):print('DETAIL',k+1,len(batch['candidates']),flush=True)
for r in rows:
    if r.get('sha256')and(file_hash(r['path'])!=r['sha256']or file_hash(Path(r['path']).parent/'metadata.json')!=r['metadata_sha256']):r.update(status='source_changed',eligible=False,reasons=['来源在本次报告完成前变化，需重新导入'])
    if r['status']=='source_changed':r['action']='暂缓·来源变化'
    elif r.get('outcome') not in (None,'completed'):r['action']='暂缓·采集未完成'
    elif r.get('coverage_issues'):r['action']='暂缓·接触范围不兼容'
    elif not r['eligible']:r['action']='不合格·建议复采'if r['status']=='excluded'else'不合格·数据错误'if r['status']=='data_error'else'暂缓·依据不足'
    elif r['status']=='suggest_review'or r.get('supplement_alerts')or r.get('supplement_failure'):r['action']='可用·建议复核'
    else:r['action']='可用·通过'
classes=[]
for label in sorted({r['label']for r in rows}):
    rr=[r for r in rows if r['label']==label];rates=[r['direction']['common_rate']for r in rr if r.get('direction',{}).get('status')=='assessed'];goodrates=[r['direction']['common_rate']for r in rr if r['action'].startswith('可用')and r.get('direction',{}).get('status')=='assessed']
    ac=collections.Counter(r['action']for r in rr)
    classes.append(dict(label=label,total=len(rr),passed=ac['可用·通过'],review=ac['可用·建议复核'],bad=sum(r['action'].startswith('不合格')for r in rr),hold=sum(r['action'].startswith('暂缓')for r in rr),
      assessed_directions=len(rates),rate_median=float(np.median(rates))if rates else None,rate_max=float(max(rates))if rates else None,
      usable_rate_median=float(np.median(goodrates))if goodrates else None,bad_records=[r['record']for r in rr if r['action'].startswith('不合格')],hold_records=[r['record']for r in rr if r['action'].startswith('暂缓')],
      batch_rule=batch['classes'].get(label)))
out=dict(scope=bundle['scope'],rule=batch['rule'],config=batch['data'],classes=classes,records=rows,counts=dict(collections.Counter(r['action']for r in rows)),
    direction_summary=dict(assessed=sum(r.get('direction',{}).get('status')=='assessed'for r in rows),marked=sum(bool(r.get('direction',{}).get('marked'))for r in rows)))
(DEST/'质量与去回一致性结果.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
summaryheaders=['布料','本次CSV','通过','可用建议复核','不合格','暂缓','可用记录去回差异率中位数','全部可计算记录最大去回差异率','不合格记录','暂缓记录']
summaryrows=[[s['label'],s['total'],s['passed'],s['review'],s['bad'],s['hold'],s['usable_rate_median'],s['rate_max'],'；'.join(s['bad_records']),'；'.join(s['hold_records'])]for s in classes]
headers=['布料','采集记录','处理建议','当前规则状态','去程最大缺口ms','回程最大缺口ms','去程超20ms次数','回程超20ms次数','完整窗去回差异率','共同X去回差异率','共同X幅值差μT','幅值差阈值μT','同向位置箱比例','四条件全部满足','去回相对RMS差','去回形状相关系数','原因','三轴补充问题','去程最大缺口X位置mm','回程最大缺口X位置mm','CSV指纹','metadata指纹','CSV路径','实际去程X范围mm','实际回程X范围mm','疑似整体坐标偏移mm','假设平移后共同X差异率','假设平移后去程最大缺口ms','假设平移后回程最大缺口ms']
def values(r):
    dm=r.get('direction',{});s=r.get('stage_details',{});a=s.get('out',{});b=s.get('back',{});loc=lambda d:f"{d['max_gap_x_from']:.3f} → {d['max_gap_x_to']:.3f}"if'max_gap_x_from'in d else''
    fullrange=lambda sd:f"{sd['full_stage_x_min']:.3f} ～ {sd['full_stage_x_max']:.3f}"if sd.get('full_stage_x_min')is not None else''
    return[r['label'],r['record'],r['action'],QUALITY.get(r['status'],r['status']),a.get('max_gap_ms'),b.get('max_gap_ms'),a.get('gap_count_over20ms'),b.get('gap_count_over20ms'),dm.get('full_rate'),dm.get('common_rate'),abs(dm['common_difference'])if'common_difference'in dm else None,dm.get('amplitude_threshold_uT'),dm.get('same_sign_fraction'),'是'if dm.get('marked')else'否'if dm.get('status')=='assessed'else'无法判断',r.get('relative_rms'),r.get('shape_correlation'),'；'.join(r['reasons']+[r['supplement_failure']]if r.get('supplement_failure')else r['reasons']),'；'.join(r.get('supplement_alerts',[])),loc(a),loc(b),r.get('sha256'),r.get('metadata_sha256'),r['path'],fullrange(a),fullrange(b),r.get('coordinate_offset_candidate_mm'),r.get('conditional_aligned_direction',{}).get('common_rate'),r.get('conditional_aligned_max_gap_ms',{}).get('out'),r.get('conditional_aligned_max_gap_ms',{}).get('back')]
rank={'不合格·建议复采':0,'不合格·数据错误':1,'暂缓·采集未完成':2,'暂缓·来源变化':2,'暂缓·依据不足':2,'暂缓·接触范围不兼容':2,'可用·建议复核':3,'可用·通过':4}
rows.sort(key=lambda r:(rank.get(r['action'],5),r['label'],r.get('sequence')or 0))
bad=[r for r in rows if r['action'].startswith('不合格')];selected=[r for r in rows if r['action']!='可用·通过']
for fn,hh,rr in [('按布料质量汇总.csv',summaryheaders,summaryrows),('不合格采集清单.csv',headers,[values(r)for r in bad]),('待处理与复核清单.csv',headers,[values(r)for r in selected]),('全部采集质量明细.csv',headers,[values(r)for r in rows])]:
    with(DEST/fn).open('w',encoding='utf-8-sig',newline='')as f:w=csv.writer(f);w.writerow(hh);w.writerows(rr)
(DEST/'表格输入.json').write_text(json.dumps(dict(summaryheaders=summaryheaders,summaryrows=summaryrows,headers=headers,selected=[values(r)for r in selected],allrows=[values(r)for r in rows]),ensure_ascii=False))
print('SCOPE',out['scope']);print('COUNTS',out['counts']);print('DIRECTION',out['direction_summary'])
for s in classes:print('CLASS',s['label'],s['total'],s['passed'],s['review'],s['bad'],s['hold'],'bad',s['bad_records'],'hold',s['hold_records'])
for r in selected:print('SELECTED',r['record'],r['action'],r['reasons'],'gaps',r.get('stage_details',{}).get('out',{}).get('max_gap_ms'),r.get('stage_details',{}).get('back',{}).get('max_gap_ms'),'R',r.get('direction',{}).get('common_rate'),'four',r.get('direction',{}).get('marked'))
