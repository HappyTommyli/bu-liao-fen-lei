"""Portable, checksummed processed segments; never re-crop a loaded snapshot."""
from pathlib import Path
import uuid
import shutil
from datetime import datetime
import numpy as np
from .config import DataConfig
from .data import Record,Trial,file_hash,metadata_hash,protocol_for,required_axes
from .folds import make_folds,validate_folds
from .storage import write_json,read_json

FORMAT='cloth-dataset-v3'
ARRAY_FIELDS=('values','times','raw_t','raw_x','raw_values','raw_valid')


def save_dataset(directory,records,config,seed=42,screening=None,folds=None,verify_sources=True,signals=None,admission=None,split_plan=None,selection=None,cancelled=lambda:False):
    directory=Path(directory)
    if directory.exists():raise ValueError('数据集版本已存在，不能覆盖；请使用新目录')
    if not records:raise ValueError('没有可冻结记录')
    for r in records if verify_sources else []:
        if cancelled():raise ValueError('已取消数据保存')
        if file_hash(r.path)!=r.sha256 or metadata_hash(r.path)!=r.metadata_sha256:raise ValueError('冻结前CSV或元数据已改变，请重新导入')
    scope=tuple(signals) if signals is not None else ({('y',):('by',),('x',):('bx',),('x','y'):('bxby',)}[tuple(records[0].axes)])
    if any(set(r.axes)!=set(required_axes(scope)) for r in records):raise ValueError('快照通道/有效点协议与所选输入不一致，请重新检查原始数据')
    from .admission import package,review_batch,require_admission,verify_batch_sources
    if admission is None:
        batches=(screening or {}).get('review_batches')
        if not batches:
            if any(r.audit.get('batch_id') for r in records):raise ValueError('缺少原固定审查批次，不能从选择子集重算')
            batches=[review_batch(records,[],[r.path for r in records],config,scope)]
        admission=package(batches)
    require_admission(records,admission,config)
    if verify_sources:verify_batch_sources(admission['batches'],config,cancelled)
    selected_folds=folds if folds is not None else make_folds(records,seed)
    validate_folds(records,selected_folds,len(selected_folds));protocol=protocol_for(records,config)
    if split_plan is not None:
        from .modern_split import validate_plan
        validate_plan(records,split_plan)
        if split_plan['seed']!=seed:raise ValueError('固定划分种子与快照不一致')
    try:
        (directory/'arrays').mkdir(parents=True)
        sources=[]
        for i,r in enumerate(records):
            if cancelled():raise ValueError('已取消数据保存')
            filename=f'arrays/{i:05d}_{r.sha256[:12]}.npz';values={}
            for prefix,trials in [('',r.trials)]+[(signal+'_',tt) for signal,tt in r.views.items()]:
                for j,t in enumerate(trials):
                    for key in ARRAY_FIELDS:values[f'{prefix}{j}_{key}']=getattr(t,key)
            np.savez_compressed(directory/filename,**values)
            sources.append(dict(r.manifest(),array=filename,array_sha256=file_hash(directory/filename)))
        manifest=dict(format=FORMAT,processing_revision=2,admission=admission,dataset_id=uuid.uuid4().hex,created_at=datetime.now().isoformat(),data=config.__dict__,protocol=protocol,
                      signals=scope,split_seed=seed,folds=selected_folds,sources=sources,screening=screening or {},contains_raw_csv=False,
                      note='处理后快照，含预览数组；不是原始CSV备份。修改选择/协议时建立新版本。')
        if split_plan is not None:manifest['split_plan']=split_plan
        if selection is not None:manifest['selection']=selection
        write_json(directory/'dataset.json',manifest)
        (directory/'dataset.sha256').write_text(file_hash(directory/'dataset.json'),encoding='ascii')
    except Exception:
        shutil.rmtree(directory);raise
    return directory


def load_dataset(directory,cancelled=lambda:False):
    directory=Path(directory).resolve()
    if file_hash(directory/'dataset.json')!=(directory/'dataset.sha256').read_text(encoding='ascii').strip():raise ValueError('数据集清单校验失败，文件已改变')
    manifest=read_json(directory/'dataset.json')
    if manifest.get('format')!=FORMAT:raise ValueError('不是支持的处理后数据快照')
    cfg=DataConfig.from_dict(manifest['data']);records=[]
    for source in manifest['sources']:
        if cancelled():raise ValueError('已取消数据载入')
        array=(directory/source['array']).resolve()
        if not array.is_relative_to(directory) or file_hash(array)!=source['array_sha256']:raise ValueError('片段数组校验失败或越界')
        with np.load(array,allow_pickle=False) as archive:
            def restore(infos,prefix=''):
                trials=[]
                for j,info in enumerate(infos):
                    a={k:archive[f'{prefix}{j}_{k}'].copy() for k in ARRAY_FIELDS};axes=tuple(info.get('axes',source['axes']))
                    if axes not in (('x',),('y',),('x','y')) or a['values'].ndim!=2 or a['values'].shape[1]!=len(axes) or not np.isfinite(a['values']).all() or len(a['values'])!=info['points'] or len(a['times'])!=info['points']:raise ValueError('片段通道、形状或有限值校验失败')
                    trials.append(Trial(info['id'],source['path'],source['sha256'],source['group'],source['label'],info['direction'],**a,bounds=tuple(info['bounds']),axes=axes))
                if [t.direction for t in trials]!=['out','back']:raise ValueError('快照缺完整去回配对')
                return trials
            trials=restore(source['trials']);views={signal:restore(infos,signal+'_') for signal,infos in source.get('views',{}).items()}
            if tuple(source['axes'])==('x','y') and set(views)!={'by','bx'}:raise ValueError('旧双通道快照缺少独立单通道片段，请重新导入原始CSV')
        if [t.direction for t in trials]!=['out','back']:raise ValueError('快照缺完整去回配对')
        records.append(Record(source['path'],source['sha256'],source.get('metadata_sha256'),source['group'],source['label'],trials,source['warnings'],source['ppmm'],source['speed'],
                              axes=tuple(source['axes']),quality=source.get('quality','normal'),quality_notes=source.get('quality_notes',[]),diagnostics=source.get('diagnostics',{}),label_confirmed=source.get('label_confirmed',True),views=views,audit=source.get('audit',{})))
    validate_folds(records,manifest['folds'],len(manifest['folds']))
    if protocol_for(records,cfg)!=manifest['protocol']:raise ValueError('快照处理协议与片段不一致')
    if 'split_plan' in manifest:
        from .modern_split import validate_plan
        validate_plan(records,manifest['split_plan'])
        if manifest['split_plan']['seed']!=manifest['split_seed']:raise ValueError('快照固定划分种子不一致')
    return records,manifest


def derive_dataset(source,destination,excluded_labels,split_plan=None):
    """Create a class-filtered snapshot without reprocessing or repartitioning retained data."""
    source=Path(source).resolve();destination=Path(destination).resolve()
    if destination==source or destination.is_relative_to(source):raise ValueError('新数据版本不能覆盖或位于原快照内部')
    if destination.exists():raise ValueError('数据集版本已存在，不能覆盖')
    records,parent=load_dataset(source);excluded=set(excluded_labels)
    if parent.get('split_plan') is not None and split_plan is not None and parent['split_plan']!=split_plan:raise ValueError('原快照划分已冻结，不能覆盖')
    if not excluded or not excluded<={r.label for r in records}:raise ValueError('请选择原版本中存在的待移除类别')
    retained=[i for i,r in enumerate(records) if r.label not in excluded];mapping={old:new for new,old in enumerate(retained)}
    def remap(indices):return [mapping[i] for i in indices if i in mapping]
    children=[records[i] for i in retained];folds=[remap(f) for f in parent['folds']]
    if len({r.label for r in children})<2:raise ValueError('派生分类数据集至少保留2类')
    validate_folds(children,folds,len(folds))
    from .modern_split import validate_plan
    plan=parent.get('split_plan') or split_plan
    if plan is None:raise ValueError('原数据版本缺少固定开发/测试划分，请提供原实验划分')
    validate_plan(records,plan)
    if plan['seed']!=parent['split_seed']:raise ValueError('原实验划分种子与快照不一致')
    new_plan=dict(plan,dev=remap(plan['dev']),test=remap(plan['test']),
                  cv=[dict(train=remap(f['train']),val=remap(f['val'])) for f in plan['cv']])
    validate_plan(children,new_plan)
    cfg=DataConfig.from_dict(parent['data'])
    save_dataset(destination,children,cfg,parent['split_seed'],screening=parent['screening'],folds=folds,verify_sources=False,
                 signals=parent['signals'],admission=parent['admission'])
    try:
        manifest=read_json(destination/'dataset.json')
        previous=parent.get('selection',{})
        manifest.update(split_plan=new_plan,selection=dict(parent_dataset_id=parent['dataset_id'],parent_manifest_sha256=file_hash(source/'dataset.json'),
                        excluded_labels=sorted(excluded|set(previous.get('excluded_labels',[]))),excluded_records=previous.get('excluded_records',[])+[dict(path=r.path,sha256=r.sha256,label=r.label,group=r.group) for r in records if r.label in excluded],
                        reason='用户明确移除类别；原始CSV与历史实验保留；保留记录的开发/测试归属和折号不变'))
        write_json(destination/'dataset.json',manifest)
        (destination/'dataset.sha256').write_text(file_hash(destination/'dataset.json'),encoding='ascii')
        load_dataset(destination)
    except Exception:
        shutil.rmtree(destination);raise
    return destination
