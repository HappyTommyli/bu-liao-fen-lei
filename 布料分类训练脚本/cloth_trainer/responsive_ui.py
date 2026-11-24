"""Focused desktop background I/O/cache; original legacy controllers stay available."""
from pathlib import Path
from dataclasses import replace
import time
import uuid
from PyQt5 import QtWidgets as W,QtCore as C
from .desktop_jobs import DesktopJobs
from .storage import write_json


class ResponsiveMixin:
    def __init__(self,*args,**kwargs):
        self.desktop_jobs=None;self._catalog=[];self._metadata={};self._selection_cache=None;self._selection_wanted=None;self._plan_cache=None;self._closing=False
        super().__init__(*args,**kwargs)
        self.desktop_jobs=DesktopJobs(self);self.desktop_jobs.changed.connect(self.notify);self.desktop_jobs.changed.connect(lambda _:self.update_selection());self.desktop_jobs.failed.connect(self.notify)
        self.cancel_wait=W.QToolButton(self);self.cancel_wait.setText('取消等待');self.cancel_wait.clicked.connect(self.desktop_jobs.cancel_all)
        layout=self.centralWidget().layout();layout.insertWidget(layout.count()-1,self.cancel_wait,0,C.Qt.AlignRight);self.cancel_wait.hide()
        self._job_ui_timer=C.QTimer(self);self._job_ui_timer.setInterval(200);self._job_ui_timer.timeout.connect(lambda:self.cancel_wait.setVisible(self.desktop_jobs.running()));self._job_ui_timer.start()
        for plot in (self.result_curve,self.matrix_plot,self.class_plot,self.prefix_plot,self.test_plot):plot.defer_when_hidden=True
        self.refresh_history();self.update_selection()

    def task_catalog(self):return self._catalog
    def task_metadata(self,directory):return self._metadata.get(str(Path(directory).resolve()),{})
    def refresh_history(self):
        if self.desktop_jobs is None or self._closing:return
        def loaded(rows):
            self._catalog=rows
            for row in rows:self._metadata[row['directory']]=row
            super(ResponsiveMixin,self).refresh_history()
        self.desktop_jobs.submit('catalog','catalog',dict(root=str(self.store)),loaded,'加载模型列表')

    def show_results(self,task):
        if self.desktop_jobs is None:return
        path=str(Path(task).resolve());self.notify('正在加载实验详情…')
        def loaded(meta):
            self._metadata[path]=meta;super(ResponsiveMixin,self).show_results(path)
        self.desktop_jobs.submit('result','metadata',dict(path=str(Path(path)/'metadata.json')),loaded,'加载实验曲线与结果')

    def online_model_choices(self):
        task=self.online_experiment.currentData()
        if not task or self.desktop_jobs is None:return super().online_model_choices()
        if self.task_metadata(task):return super().online_model_choices()
        path=str(Path(task).resolve())
        def loaded(meta):
            self._metadata[path]=meta
            if self.online_experiment.currentData()==task:super(ResponsiveMixin,self).online_model_choices()
        self.desktop_jobs.submit('online-model','metadata',dict(path=str(Path(path)/'metadata.json')),loaded,'加载所选模型信息')

    def update_selection(self,*args):
        # UI preflight is advisory; the backend always validates the full frozen audit.
        if not hasattr(self,'editor') or self._closing:return
        from collections import Counter
        from .modern_split import make_plan
        selected=self.selected_records();labels=Counter(r.label for r in selected)
        self.data_summary.setText(f'已载入{len(self.records)} CSV · 已选{len(selected)} CSV · {len(labels)}类 · {len(selected)*2}单程')
        self.training_summary.setText(self.data_summary.text());self.training_summary.setToolTip(self.data_summary.text())
        key=(tuple((r.path,r.label,r.group) for r in selected),self.config.split_seed,self.config.cv_folds,id(self.snapshot_manifest))
        if self._selection_cache is None or self._selection_cache[0]!=key:
            self._selection_cache=(key,None,'正在后台检查采集组划分…')
            self._selection_wanted=(key,list(selected),self.config)
            C.QTimer.singleShot(250,self._run_selection_check)
        valid,message=self._selection_cache[1:];self.partition_summary.setText(message)
        self.start_button.setEnabled(bool(valid) and not self.busy() and not self.import_jobs and not self.project_read_only)
        self.cancel_button.setEnabled(self.busy() and not self.cancelled);self.update_task_count()

    def _run_selection_check(self):
        if self.desktop_jobs is None or self._selection_wanted is None or self._closing:return
        if 'preflight' in self.desktop_jobs.active:
            C.QTimer.singleShot(200,self._run_selection_check);return
        key,selected,cfg=self._selection_wanted
        if self._selection_cache and self._selection_cache[0]==key and self._selection_cache[1] is not None:return
        def check(cancelled):
            try:
                if not selected:raise ValueError('请先导入数据或载入数据版本。')
                if self.project_read_only:raise ValueError('历史项目只读，请新建前缀CERA实验。')
                if self.snapshot_manifest:
                    plan=super(ResponsiveMixin,self).selected_plan(selected,cfg.split_seed)
                else:
                    from .modern_split import make_plan
                    plan=make_plan(selected,cfg.split_seed,cfg.cv_folds)
                part=plan['cv'][0];return True,f"固定测试{len(plan['test'])} CSV · 训练{len(part['train'])} / 验证{len(part['val'])} CSV",plan
            except ValueError as e:return False,str(e),None
        def done(value):
            if self._selection_wanted and self._selection_wanted[0]==key:
                valid,message,plan=value;self._selection_cache=(key,valid,message);self._plan_cache=(key,plan);self.update_selection()
        self.desktop_jobs.submit_thread('preflight',check,done,'检查数据分组')

    def selected_plan(self,records,seed):
        key=(tuple((r.path,r.label,r.group) for r in records),seed,self.config.cv_folds,id(self.snapshot_manifest))
        if self._plan_cache and self._plan_cache[0]==key and self._plan_cache[1] is not None:return self._plan_cache[1]
        return super().selected_plan(records,seed)

    def busy(self):
        jobs=getattr(self,'desktop_jobs',None)
        return super().busy() or bool(jobs and any(k in jobs.active for k in ('prepare-training','launch','dataset','finish-task')))

    def start_training(self):
        if self.busy() or self.import_jobs:return
        records=list(self.selected_records());cfg=self.controls_config();batches=self.batches;screening=self.screening;exclusions=list(self.exclusions);snapshot=self.snapshot_path;manifest=self.snapshot_manifest;store=str(self.store)
        if not self._selection_cache or not self._selection_cache[1]:self.notify('请等待后台分组检查完成。');return
        def prepare(cancelled):
            from .admission import require_admission,package
            from .data import protocol_for
            require_admission(records,package(batches),cfg.data)
            if cancelled():return None
            if snapshot:
                if [[r.path,r.sha256,r.metadata_sha256,r.label,r.group,r.label_confirmed] for r in records]!=[[v['path'],v['sha256'],v.get('metadata_sha256'),v['label'],v['group'],v.get('label_confirmed',True)] for v in manifest['sources']]:raise ValueError('快照选择已更改，请先保存新版本')
                return dict(snapshot=str(snapshot),config=cfg.to_dict(),output=store)
            return dict(paths=[r.path for r in records],sources=[r.manifest() for r in records],review_batches=batches,protocol=protocol_for(records,cfg.data),overrides={r.path:r.label for r in records},config=cfg.to_dict(),output=store,screening=dict(screening,excluded=exclusions))
        def done(request):
            if request is not None:self.live_history=[];self.training_log.clear();self.navigation.setCurrentRow(1);self.launch(request,'train')
        self.desktop_jobs.submit_thread('prepare-training',prepare,done,'检查训练数据并冻结配置')

    def read_project(self,path):
        from .storage import read_json
        value=read_json(path)
        self._project_snapshot_selection={x['path'] for x in value.get('sources',[]) if x.get('selected')} if value.get('snapshot') else None
        try:return super().read_project(path)
        finally:self._project_snapshot_selection=None

    def load_dataset_version(self,path):
        if self.busy() or self.import_jobs:raise ValueError('请等待当前训练或导入完成')
        from .dataset import load_dataset
        from .config import DataConfig
        path=Path(path).resolve();chosen=getattr(self,'_project_snapshot_selection',None)
        def load(cancelled):return load_dataset(path,cancelled=cancelled)
        def apply(value):
            records,manifest=value
            if manifest.get('signals')!=['by'] or manifest['data'].get('preprocess_version')!='abs-v3':raise ValueError('当前工作区只支持abs-v3 / |dBy|快照')
            self.snapshot_path=path;self.snapshot_manifest=manifest;self.batches=manifest.get('admission',{}).get('batches',[]);self.import_reports=[]
            changes=dict(cv_folds=len(manifest['split_plan']['cv'])) if manifest.get('split_plan') else {}
            self.config=replace(self.config,data=DataConfig.from_dict(manifest['data']),split_seed=manifest['split_seed'],**changes)
            self.records=[];self.errors=[];self.overrides={};self.data_table.setRowCount(0);self.screening=manifest['screening'];self.sync_controls()
            paths={r.path for r in records};removed=set(manifest.get('selection',{}).get('excluded_labels',[]))
            missing={e['path']:dict(e,message='；'.join(e['audit']['reasons']) or '本版本未选择该记录的处理片段',review_only=True) for b in self.batches for e in b['entries'] if e['path'] not in paths and e.get('label') not in removed}
            self.add_records(records,list(missing.values()))
            if chosen is not None:
                for i,r in enumerate(self.records):self.data_table.item(i,0).setCheckState(C.Qt.Checked if r.path in chosen else C.Qt.Unchecked)
                self.update_selection()
            self.notify(f"已载入数据版本{manifest['dataset_id'][:10]} · {len(records)}CSV")
        self.desktop_jobs.submit_thread('dataset',load,apply,'核验并载入数据版本')

    def save_dataset_version(self,path):
        if self.busy() or self.import_jobs:raise ValueError('请等待当前训练或导入完成')
        from .dataset import save_dataset
        from .folds import make_folds
        from .admission import package
        selected=list(self.selected_records());cfg=self.controls_config();plan=self.selected_plan(selected,cfg.split_seed);batches=self.batches;screening=self.screening
        def save(cancelled):
            if cancelled():return
            save_dataset(path,selected,cfg.data,cfg.split_seed,screening,make_folds(selected,cfg.split_seed,max(2,cfg.cv_folds)),signals=cfg.signals,admission=package(batches),split_plan=plan,cancelled=cancelled)
            return path
        self.desktop_jobs.submit_thread('dataset',save,lambda p:self.load_dataset_version(p) if p else None,'核验并保存数据版本')

    def launch(self,request,kind):
        if self.desktop_jobs is None:return super().launch(request,kind)
        requests=self.store.parent/'requests';requests.mkdir(parents=True,exist_ok=True);path=requests/(uuid.uuid4().hex+'.json')
        def prepare(cancelled):
            if cancelled():return
            write_json(path,request)
            return path
        def launch(path):
            if path is None:return
            import sys
            self.worker_kind=kind;self.cancelled=False;self.active_task=None;self.buffer='';self.stderr='';self.active_request=path
            self.process.setProgram(sys.executable);self.process.setArguments([str(Path(__file__).resolve().parents[1]/'train_cloth.py'),'--worker',str(path),'--kind',kind]);self.process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]));self.process.start()
            self.progress.setRange(0,0);self.training_state.setText('后台任务准备中…');self.update_selection()
        self.desktop_jobs.submit_thread('launch',prepare,launch,'保存冻结任务配置')

    def worker_finished(self,code,status):
        self.read_output();self.read_error()
        if not self.active_task:return super().worker_finished(code,status)
        task=self.active_task;success=code==0 and not self.cancelled
        def done(meta):
            self._metadata[str(task.resolve())]=meta;ok=success and meta.get('status')=='completed'
            self.progress.setRange(0,100);self.progress.setValue(100 if ok else 0)
            self.training_state.setText('已完成并保存' if ok else '任务取消或失败，详见日志')
            if ok:
                super(ResponsiveMixin,self).show_results(task);self.navigation.setCurrentRow(2)
            else:self.training_log.appendPlainText(meta.get('error') or self.stderr or '任务已取消')
            self.refresh_history();self.update_selection()
        self.desktop_jobs.submit('finish-task','finish-task',dict(path=str(task/'metadata.json'),cancelled=self.cancelled,error=self.stderr),done,'读取任务完成状态')

    def export_figures(self):
        for plot in (self.result_curve,self.matrix_plot,self.class_plot):plot.flush_pending()
        return super().export_figures()

    def history_load(self):
        if not self.result_task:return
        task=str(self.result_directory);key=self.result_selector.currentData()
        self.navigation.setCurrentRow(3)
        index=self.online_experiment.findData(task)
        if index<0:self.notify('该历史模型不支持当前在线输入协议；请使用历史离线工具。');return
        self.online_experiment.setCurrentIndex(index)
        model=self.online_model.findData(str(Path(task)/str(key)/'model.pt'))
        if model>=0:self.online_model.setCurrentIndex(model)
        self.online_source_mode.setCurrentIndex(self.online_source_mode.findData('independent'))

    def closeEvent(self,event):
        self._closing=True
        if self.desktop_jobs is not None and self.desktop_jobs.running():
            self.desktop_jobs.cancel_all();self.notify('正在取消后台读取，请稍候…');event.ignore();C.QTimer.singleShot(100,self.close);return
        if self.process.state()!=C.QProcess.NotRunning:
            self.cancel_worker();event.ignore();C.QTimer.singleShot(200,self.close);return
        super().closeEvent(event)
