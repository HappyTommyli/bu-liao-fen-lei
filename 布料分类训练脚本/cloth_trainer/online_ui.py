"""Online replay and opt-in loopback receiver; inference stays outside Qt's GUI thread."""
from dataclasses import asdict
from pathlib import Path
import csv
import json
import time
import threading
import math
from PyQt5 import QtWidgets as W,QtCore as C
from .online import OnlineConfig,OnlinePredictor,csv_packets,csv_metadata,tcp_packets,summarize_records,freeze_policy,validate_policy,claim_holdout_policy
from .storage import list_tasks,read_json
from .ui import label,button,table,row_values

ROLE_TITLES={"independent":"独立数据","independent-exploratory":"独立数据（探索）","validation-tuning":"验证集调参","training-replay":"训练数据重放","holdout-protected":"受保护测试集","holdout-frozen":"冻结测试评价","live-exploratory":"实时识别（探索）"}


def online_row_supported(row):
    candidate=row.get('candidate') or {};window=candidate.get('window_size')
    revision=candidate.get('spec',{}).get('revision')
    expected={'tcn':('configurable-tcn-v1',),'cera':('configurable-cera-v1','configurable-cera-prefix-v1')}.get(row.get('architecture'))
    return row.get('status')=='completed' and row.get('signal')=='by' and expected is not None and revision in expected and type(window) is int and window>0


class OnlineWorker(C.QThread):
    observed=C.pyqtSignal(dict)
    record=C.pyqtSignal(dict)
    loaded=C.pyqtSignal(dict)
    failed=C.pyqtSignal(str)

    def __init__(self,model_path,paths,config,source='independent',live=False,speed=0.,parent=None,policy=None,heldout=False,include_return=None,live_truth=''):
        super().__init__(parent);self.model_path=str(model_path);self.paths=list(paths);self.config=config
        self.source=source;self.live=live;self.speed=speed;self.cancel=threading.Event();self.paused=threading.Event();self.policy=policy;self.heldout=heldout;self.include_return=config.include_return if include_return is None else include_return;self.live_truth=live_truth

    def stop(self):self.cancel.set()
    def set_paused(self,value):self.paused.set() if value else self.paused.clear()
    def _wait(self,seconds=0.):
        if seconds>0 and self.cancel.wait(seconds):return False
        while self.paused.is_set():
            if self.cancel.wait(.05):return False
        return not self.cancel.is_set()

    def run(self):
        try:
            from .modern_engine import load_online_bundle
            bundle=load_online_bundle(self.model_path)
            if self.policy:
                frozen_config=validate_policy(self.policy,self.model_path)
                if self.config!=frozen_config or self.include_return!=frozen_config.include_return:raise ValueError('运行设置与冻结在线策略不一致')
            if self.heldout:
                if not self.policy:raise ValueError('留出评价需要冻结在线策略')
                claim_holdout_policy(self.model_path,self.policy,bundle)
                self.paths=[bundle['sources'][i]['path'] for i in bundle['split_audit']['test']]
            self.loaded.emit(dict(model_source=bundle.get('model_source',{}),split_audit=bundle.get('split_audit',{}),labels=bundle['labels']))
            if self.live:self._stream(bundle,tcp_packets(cancelled=self.cancel.is_set),'TCP 127.0.0.1:8766',self.live_truth)
            else:
                for path in self.paths:
                    if not self._wait():break
                    from .data import file_hash,metadata_hash
                    digest=file_hash(path)
                    known=[(i,r) for i,r in enumerate(bundle.get('sources',[])) if r.get('sha256')==digest or Path(r['path']).resolve()==Path(path).resolve()]
                    protected=any(i in bundle.get('split_audit',{}).get('test',[]) for i,_ in known)
                    if not self.heldout and self.source=='validation-tuning' and (not known or any(i not in bundle.get('split_audit',{}).get('val',[]) for i,_ in known)):raise ValueError('输入不属于当前模型冻结验证子集')
                    if protected and not self.heldout:raise ValueError('受保护留出集不能用于在线调参（包含复制改名文件）')
                    if self.heldout and not protected:raise ValueError('留出来源不匹配')
                    if (self.heldout or self.source=='validation-tuning') and any(r['sha256']!=digest or r.get('metadata_sha256')!=metadata_hash(path) for _,r in known):raise ValueError('冻结CSV或元数据已改变')
                    self.current_source='holdout-frozen' if self.heldout else 'validation-tuning' if self.source=='validation-tuning' else 'training-replay' if known else 'independent-exploratory'
                    meta,_,_=csv_metadata(path)
                    self._stream(bundle,csv_packets(path,self.cancel.is_set),path,str(meta.get('label','')))
        except Exception as e:self.failed.emit(str(e))

    def _stream(self,bundle,packets,path,truth):
        predictor=OnlinePredictor(bundle,config=self.config);active=None;previous_t=None;segment_start=None;effective_start=None;trace=[];last_effective_t=None
        def flush_trace():
            nonlocal trace
            if trace:self.observed.emit(dict(type='trace',path=str(path),stage=(active or ('','step_003'))[1],points=trace,labels=bundle['labels']));trace=[]
        def finish():
            if active is None:return
            flush_trace();end=predictor.feed(dict(type='end'))[-1];prediction=end['prediction']
            first=end.get('first_confirmation')
            row=dict(path=str(path),direction='out' if active[1]=='step_003' else 'back',source=getattr(self,'current_source','live-exploratory'),policy_fingerprint=(self.policy or {}).get('fingerprint'),
                     true_label=truth,prediction=prediction,status='abstain' if prediction is None else ('correct' if truth==prediction else 'error') if truth else 'unlabelled',
                     first_confirmation=first,confirmation_elapsed_s=(first['t_s']-segment_start) if first and segment_start is not None else None,
                     flip_count=end['flip_count'],quality_resets=end['quality_resets'],warmup_points=end['warmup_points'],sample_count=end['sample_count'],completion=end.get('completion'),effective_confirmation_s=(first['t_s']-effective_start) if first and effective_start is not None else None,lead_before_crop_end_s=(last_effective_t-first['t_s']) if first and last_effective_t is not None and not self.cancel.is_set() else None,scan_finished=not self.cancel.is_set())
            self.record.emit(row);self.observed.emit(end)
        for packet in packets:
            if not self._wait():break
            kind=packet.get('type');stage=packet.get('stage');identity=(packet.get('capture_id'),stage)
            if kind=='reset':
                if active is not None and packet.get('capture_id',active[0])==active[0] and packet.get('stage',active[1])==active[1]:
                    if not predictor.locked:
                        predictor.feed(dict(type='reset'));self.observed.emit(dict(type='quality',status='error',path=str(path),stage=active[1],reason=packet.get('reason','data_reset')))
                    previous_t=None;continue
                finish();active=None;previous_t=None
            if kind=='sample' and stage=='step_004' and not self.include_return:
                finish();active=None;previous_t=None;predictor.feed(dict(type='reset',reason='return_disabled'));continue
            if kind=='sample' and identity!=active:
                finish();active=identity if stage in ('step_003','step_004') else None;segment_start=packet.get('t_s');previous_t=None;effective_start=None;last_effective_t=None
            if kind=='sample' and self.speed and previous_t is not None:
                delay=max(0.,min(1.,float(packet['t_s'])-previous_t))/self.speed
                if not self._wait(delay):break
            if kind=='sample':previous_t=float(packet['t_s'])
            if kind=='end':finish();active=None;continue
            if kind=='sample' and stage in ('step_003','step_004'):
                low,high=(predictor.data.out_low,predictor.data.out_high) if stage=='step_003' else (predictor.data.back_low,predictor.data.back_high)
                if low<=float(packet['x_mm'])<=high and all(math.isfinite(float(packet[k])) for k in ('t_s','x_mm','by_uT','baseline_by_uT')):
                    if effective_start is None:effective_start=float(packet['t_s'])
                    last_effective_t=float(packet['t_s'])
                    trace.append([float(packet['t_s']),float(packet['x_mm']),abs(float(packet['by_uT'])-float(packet['baseline_by_uT']))])
                    if len(trace)>=64:flush_trace()
            started=time.perf_counter()
            for event in predictor.feed(packet):
                event=dict(event,path=str(path),source=getattr(self,'current_source','live-exploratory'),inference_latency_ms=(time.perf_counter()-started)*1000)
                self.observed.emit(event)
        if active is not None and (predictor.locked or not self.cancel.is_set()):finish()


class OnlineMixin:
    def build_online(self):
        self.online_paths=[];self.online_records=[];self.online_events=[];self.online_worker=None;self.online_source='independent';self.online_policy=None;self.online_window=900
        page=W.QWidget();page.setObjectName('tcnOnline');layout=W.QVBoxLayout(page);layout.setContentsMargins(8,8,8,8)
        notice=label('扫描途中识别 · 达到最少观察量后持续判断，满足确认条件后锁定。模型概率不等于实际正确率。','notice');notice.setWordWrap(True);layout.addWidget(notice)
        models=W.QHBoxLayout();self.online_experiment=W.QComboBox();self.online_model=W.QComboBox()
        for widget in (self.online_experiment,self.online_model):
            widget.setMinimumContentsLength(10);widget.setSizeAdjustPolicy(W.QComboBox.AdjustToMinimumContentsLengthWithIcon);widget.setMinimumWidth(0);widget.setSizePolicy(W.QSizePolicy.Expanding,W.QSizePolicy.Fixed)
        models.addWidget(label('实验'));models.addWidget(self.online_experiment,1);models.addWidget(label('模型'));models.addWidget(self.online_model,2);layout.addLayout(models)
        self.online_model_details=label('选择已完成训练的TCN或CERA。','muted');self.online_model_details.setWordWrap(True);layout.addWidget(self.online_model_details)
        source_row=W.QHBoxLayout();source_row.addWidget(label('数据来源'));self.online_source_mode=W.QComboBox()
        for title,key in [('验证集调试','validation-tuning'),('新数据回放','independent'),('实时采集','live')]:self.online_source_mode.addItem(title,key)
        source_row.addWidget(self.online_source_mode,1);layout.addLayout(source_row)
        self.online_source_hint=label('','muted');self.online_source_hint.setWordWrap(True);layout.addWidget(self.online_source_hint)
        self.online_input_group=W.QWidget();inputs=W.QVBoxLayout(self.online_input_group);inputs.setContentsMargins(0,0,0,0)
        self.online_choose=W.QToolButton();self.online_choose.setText('选择数据');self.online_choose.setPopupMode(W.QToolButton.InstantPopup)
        menu=W.QMenu(self.online_choose);menu.addAction('选择CSV文件',self.choose_online_files);menu.addAction('选择文件夹',self.choose_online_directory);self.online_choose.setMenu(menu);inputs.addWidget(self.online_choose)
        self.online_queue=table(['输入CSV','用途 / 数据来源']);self.online_queue.setMinimumHeight(60);self.online_queue.setMaximumHeight(110);inputs.addWidget(self.online_queue);layout.addWidget(self.online_input_group)
        self.online_truth_group=W.QWidget();truth_row=W.QHBoxLayout(self.online_truth_group);truth_row.setContentsMargins(0,0,0,0)
        truth_row.addWidget(label('本次真实布料'));self.online_live_label=W.QLineEdit();self.online_live_label.setPlaceholderText('可留空；仅用于统计正确/错误，不参与预测');truth_row.addWidget(self.online_live_label,1);layout.addWidget(self.online_truth_group)
        gates=W.QGridLayout()
        self.online_probability=W.QDoubleSpinBox();self.online_probability.setRange(0,1);self.online_probability.setSingleStep(.01);self.online_probability.setValue(.95)
        self.online_stability=W.QDoubleSpinBox();self.online_stability.setRange(0,60);self.online_stability.setSingleStep(.1);self.online_stability.setValue(2.7)
        gates.addWidget(label('最低概率'),0,0);gates.addWidget(self.online_probability,0,1);gates.addWidget(label('稳定时间 / 秒'),0,2);gates.addWidget(self.online_stability,0,3)
        self.online_advanced_toggle=W.QCheckBox('高级设置');layout.addWidget(self.online_advanced_toggle)
        self.online_advanced=W.QGroupBox('识别参数');advanced=W.QFormLayout(self.online_advanced);advanced.setRowWrapPolicy(W.QFormLayout.WrapLongRows)
        gate_box=W.QWidget();gate_box.setLayout(gates);advanced.addRow(gate_box)
        self.online_consecutive=W.QSpinBox();self.online_consecutive.setRange(1,20);self.online_consecutive.setValue(2)
        self.online_lock=W.QCheckBox('达标后锁定（采集继续）');self.online_lock.setChecked(True)
        advanced.addRow('连续达标次数',self.online_consecutive);advanced.addRow(self.online_lock)
        self.online_stride=W.QComboBox()
        for n in (900,450,225):self.online_stride.addItem(str(n),n)
        self.online_margin=W.QDoubleSpinBox();self.online_margin.setRange(0,1);self.online_margin.setSingleStep(.01);self.online_margin.setValue(.20)
        self.online_smoothing=W.QCheckBox('概率平滑');self.online_smoothing.setChecked(False);self.online_return=W.QCheckBox('另评回程');self.online_return.setChecked(False)
        self.online_speed=W.QComboBox();self.online_speed.addItem('最快回放',0.);self.online_speed.addItem('实时速度',1.);self.online_speed.addItem('4倍速度',4.)
        for title,widget in [('预测步长 / 点',self.online_stride),('前两类概率差≥',self.online_margin),('回放速度',self.online_speed),('',self.online_smoothing),('',self.online_return)]:advanced.addRow(title,widget)
        layout.addWidget(self.online_advanced);self.online_advanced.hide();self.online_advanced_toggle.toggled.connect(self.online_advanced.setVisible)
        controls=W.QHBoxLayout();self.online_start=button('开始',self.start_online,True);self.online_stop=button('停止',self.stop_online);self.online_export=button('导出结果',self.export_online)
        for widget in (self.online_start,self.online_stop):controls.addWidget(widget)
        self.online_export.hide()
        self.online_more=W.QToolButton();self.online_more.setText('更多');self.online_more.setPopupMode(W.QToolButton.InstantPopup)
        menu=W.QMenu(self.online_more);self.online_pause=menu.addAction('暂停回放',self.pause_online);menu.addAction('导出结果',self.export_online);self.online_clear=menu.addAction('清空输入',self.clear_online_inputs);self.online_more.setMenu(menu);controls.addWidget(self.online_more);layout.addLayout(controls)
        self.online_state=label('等待开始','muted');layout.addWidget(self.online_state)
        self.online_current=label('尚未识别','section');self.online_current.setWordWrap(True);layout.addWidget(self.online_current)
        self.online_tabs=W.QTabWidget();self.online_predictions=table(['时间 / 秒','位置 / mm','当前预测','概率','差值','状态','首次确认 / 反转','推理ms'])
        self.online_tabs.addTab(self.online_predictions,'逐窗口明细');self.online_results=table(['采集 / 方向','真实布料','锁定结果','评价','单程用时/s','有效用时/s','有效段提前/s'])
        records_page=W.QWidget();records_layout=W.QVBoxLayout(records_page);records_layout.setContentsMargins(0,0,0,0)
        pages=W.QHBoxLayout();pages.addWidget(label('每页100条 · 点击记录回看过程'));pages.addStretch();self.online_page=W.QSpinBox();self.online_page.setRange(1,1);pages.addWidget(label('页'));pages.addWidget(self.online_page);records_layout.addLayout(pages);records_layout.addWidget(self.online_results);self.online_page.valueChanged.connect(self._render_online_records)
        self.online_tabs.addTab(records_page,'识别结果');self.online_summary=W.QPlainTextEdit();self.online_summary.setReadOnly(True);self.online_tabs.addTab(self.online_summary,'统计汇总');self.online_class_table=table(['布料','精确率','召回率','F1','样本数']);self.online_tabs.addTab(self.online_class_table,'各类表现')
        from .plots import Plot
        matrix_page=W.QWidget();ml=W.QVBoxLayout(matrix_page);ml.addWidget(label('矩阵仅统计已输出且有已知真实标签的记录；未确认单独计数。','muted'));self.online_matrix=Plot();ml.addWidget(self.online_matrix);self.online_tabs.addTab(matrix_page,'混淆矩阵');self.online_tabs.currentChanged.connect(self._draw_online_matrix)
        self._online_summary_timer=C.QTimer(self);self._online_summary_timer.setSingleShot(True);self._online_summary_timer.setInterval(250);self._online_summary_timer.timeout.connect(self._render_online_summary)
        self.online_tabs.setCurrentIndex(1);self.online_tabs.setMinimumHeight(180)
        from .early_timeline import EarlyTimeline
        self.online_timeline=EarlyTimeline();self.online_stack=W.QStackedWidget();self.online_stack.addWidget(self.online_timeline);self.online_stack.addWidget(self.online_tabs)
        views=W.QHBoxLayout();self.online_view=W.QComboBox();self.online_view.addItem('当前过程',0);self.online_view.addItem('本批结果',1);views.addWidget(self.online_view);views.addStretch();layout.addLayout(views);layout.addWidget(self.online_stack,1)
        self.online_view.currentIndexChanged.connect(self.online_stack.setCurrentIndex)
        self.online_results.itemSelectionChanged.connect(self.review_online_record)
        self._online_traces={};self._online_trace_key=None;self._online_display_queue=[];self._online_flush_timer=C.QTimer(self);self._online_flush_timer.setInterval(100);self._online_flush_timer.timeout.connect(self._flush_online_events);self._online_flush_timer.start()
        self.online_formal_toggle=W.QCheckBox('正式测试（门槛确定后使用）');layout.addWidget(self.online_formal_toggle)
        self.online_formal=W.QGroupBox('固定测试集评价');formal=W.QVBoxLayout(self.online_formal)
        text=label('先用验证集确定门槛，再锁定并评价测试集。首次测试策略会持久保存，换门槛不能重新冒充独立测试。','muted');text.setWordWrap(True);formal.addWidget(text)
        formal.addWidget(button('锁定门槛并评价测试集',self.freeze_and_evaluate_online))
        policy_actions=W.QGridLayout()
        for i,(title,callback) in enumerate([('保存门槛',self.freeze_online_policy),('载入门槛',self.load_online_policy),('重新调参',self.unlock_online_policy)]):policy_actions.addWidget(button(title,callback),i//2,i%2)
        formal.addLayout(policy_actions);layout.addWidget(self.online_formal);self.online_formal.hide();self.online_formal_toggle.toggled.connect(self.online_formal.setVisible)
        self.online_experiment.currentIndexChanged.connect(self.online_model_choices);self.online_model.currentIndexChanged.connect(self.online_describe_model)
        self.online_source_mode.currentIndexChanged.connect(self._online_source_changed)
        self.refresh_online_models();self._online_source_changed();return page

    def _online_source_changed(self,*args):
        if self.online_worker and self.online_worker.isRunning():return
        mode=self.online_source_mode.currentData()
        self._set_online_paths([])
        self.online_input_group.setVisible(mode=='independent');self.online_truth_group.setVisible(mode=='live')
        self.online_speed.setEnabled(mode!='live');self.online_pause.setEnabled(mode!='live')
        self.online_source_hint.setText({'validation-tuning':'点击开始后自动载入当前模型的验证子集，用于调整门槛。测试集保持封存。','independent':'选择新采集CSV或文件夹，模拟扫描途中识别；反复用于调门槛的数据属于调参数据。','live':'先在采集程序开启本机实时转发，点击开始后连接127.0.0.1:8766。'}[mode])

    def start_online(self):
        mode=self.online_source_mode.currentData()
        if mode=='validation-tuning':
            if not self.load_online_validation():return
            self._launch_online()
        elif mode=='live':self._launch_online(live=True)
        else:self._launch_online()

    def freeze_and_evaluate_online(self):
        if self.online_worker and self.online_worker.isRunning():return
        if self.online_policy is None:self.freeze_online_policy(after=self.evaluate_online_holdout)
        else:self.evaluate_online_holdout()

    def online_metadata(self,path):
        return self.task_metadata(path) if hasattr(self,'task_metadata') else read_json(Path(path)/'metadata.json')

    def refresh_online_models(self):
        if not hasattr(self,'online_experiment'):return
        if self.online_policy is not None or (self.online_worker and self.online_worker.isRunning()):return
        selected=self.online_experiment.currentData();self.online_experiment.blockSignals(True);self.online_experiment.clear()
        for meta in (self.task_catalog() if hasattr(self,'task_catalog') else list_tasks(self.store)):
            rows=list(meta.get('models',{}).values())+([meta['final_model']] if meta.get('final_model') else [])
            if meta.get('format')=='cloth-classification-v4' and any(online_row_supported(r) for r in rows):
                self.online_experiment.addItem(meta.get('name',meta['directory']),meta['directory'])
        self.online_experiment.setCurrentIndex(max(0,self.online_experiment.findData(selected)));self.online_experiment.blockSignals(False);self.online_model_choices()

    def online_model_choices(self):
        self.online_model.clear();task=self.online_experiment.currentData()
        if not task:return
        meta=self.online_metadata(task);rows=[(k,r) for k,r in meta.get('models',{}).items()]
        if meta.get('final_model'):rows.append(('final',meta['final_model']))
        for key,row in rows:
            if online_row_supported(row):
                phase='最终模型' if key=='final' else '单次最佳' if meta.get('config',{}).get('cv_folds',5)==1 else f'验证折{int(row.get("fold",0))+1}'
                candidate=row['candidate'];window=candidate['window_size'];stride=candidate.get('window_stride',window)
                title=f"{row['architecture'].upper()} · {phase} · "+('累积前缀' if candidate.get('prefix_policy') else '不重叠' if stride==window else f'重叠{1-stride/window:.0%}')
                self.online_model.addItem(title,str(Path(task)/key/'model.pt'))
                self.online_model.setItemData(self.online_model.count()-1,key,C.Qt.ToolTipRole)
        self.online_describe_model()

    def online_describe_model(self):
        path=self.online_model.currentData()
        if path:
            meta=self.online_metadata(Path(path).parent.parent);key=Path(path).parent.name
            model=(meta.get('final_model') if key=='final' else meta.get('models',{}).get(key)) or {}
            self.online_is_prefix=bool((model.get('candidate') or {}).get('prefix_policy'))
            self.online_window=(model.get('candidate') or {}).get('window_size',meta.get('config',{}).get('tcn_window',900))
            self.online_stride.clear()
            if self.online_window:
                for stride in dict.fromkeys((self.online_window,max(1,self.online_window//2),max(1,self.online_window//4))):self.online_stride.addItem(str(stride),stride)
        self.online_model_details.setText((f'累计历史 · 最少{self.online_window}点后开始判断 · 达标后锁定。' if getattr(self,'online_is_prefix',False) else f'历史窗口模型 · 每次仅看最近{self.online_window}点；累积模式需重新训练前缀模型。') if path else '尚无可用窗口模型：请先完成TCN/CERA训练。')
        self.online_model_details.setToolTip(str(path or ''))
        if getattr(self,'online_is_prefix',False) and self.online_policy is None:self.online_lock.setChecked(True);self.online_lock.setEnabled(False)

    def _online_role(self,path):
        task=self.online_experiment.currentData()
        if not task:return 'independent'
        meta=self.online_metadata(task);sources=meta.get('sources',[])
        indices=[i for i,s in enumerate(sources) if Path(s['path']).resolve()==Path(path).resolve()]
        if not indices:return 'independent'
        if any(i in meta.get('split_plan',{}).get('test',[]) for i in indices):return 'holdout-protected'
        return 'training-replay'

    def _set_online_paths(self,paths,source='independent'):
        if self.online_worker and self.online_worker.isRunning():return
        self.online_source=source;self.online_paths=list(dict.fromkeys(str(Path(p).resolve()) for p in paths))
        self.online_queue.setRowCount(len(self.online_paths))
        for i,path in enumerate(self.online_paths):
            role='validation-tuning' if source=='validation-tuning' else self._online_role(path)
            row_values(self.online_queue,i,[path,ROLE_TITLES.get(role,role)])

    def choose_online_files(self):
        paths,_=W.QFileDialog.getOpenFileNames(self,'选择独立采样CSV','','CSV (*.csv)')
        if paths:
            self.online_source_mode.setCurrentIndex(self.online_source_mode.findData('independent'))
            self._set_online_paths((self.online_paths if self.online_source=='independent' else [])+paths)

    def choose_online_directory(self):
        path=W.QFileDialog.getExistingDirectory(self,'选择CSV目录')
        if path:
            self.online_source_mode.setCurrentIndex(self.online_source_mode.findData('independent'))
            self._set_online_paths([str(p) for p in sorted(Path(path).rglob('*.csv'))])

    def clear_online_inputs(self):self._set_online_paths([])

    def load_online_validation(self):
        path=self.online_model.currentData()
        if not path:self.online_state.setText('请先选择已完成训练的模型。');return False
        # Only explicit action reads frozen validation membership, never holdout data.
        try:
            task=Path(path).parent.parent;meta=self.online_metadata(task);key=Path(path).parent.name
            row=meta.get('models',{}).get(key)
            if not row:raise ValueError('最终模型没有独立验证子集；请选择验证折模型。')
            indices=meta['split_plan']['cv'][row['fold']]['val']
            self._set_online_paths([meta['sources'][i]['path'] for i in indices],'validation-tuning');return True
        except Exception as e:self.online_state.setText(str(e));return False

    def _online_config(self):
        return OnlineConfig(window=self.online_window,stride=self.online_stride.currentData(),probability=self.online_probability.value(),margin=self.online_margin.value(),stable_seconds=self.online_stability.value(),smoothing=self.online_smoothing.isChecked(),include_return=self.online_return.isChecked(),consecutive=self.online_consecutive.value(),lock_on_confirm=self.online_lock.isChecked())

    def _launch_online(self,live=False,heldout=False):
        if self.online_worker and self.online_worker.isRunning():return
        path=self.online_model.currentData()
        if not path:self.online_state.setText('请先选择已验证完成的模型。');return
        if live and heldout:raise ValueError('实时连接不能作为固定测试集评价')
        if not live and not heldout and not self.online_paths:self.online_state.setText('请先添加CSV或冻结验证子集。');return
        if not live and not heldout and any(self._online_role(p)=='holdout-protected' for p in self.online_paths):
            self.online_state.setText('已阻止受保护留出集进入在线门槛调参。请使用冻结验证子集或独立CSV。');return
        if not self.online_window:self.online_state.setText('完整单程模型不能使用在线切窗；请选用已训练窗口模型。');return
        self.online_events=[];self.online_records=[];self.online_predictions.setRowCount(0);self.online_results.setRowCount(0);self._online_traces={};self._online_trace_key=None;self._online_display_queue=[];self.online_timeline.clear()
        self.online_run_settings=dict(asdict(self._online_config()),include_return=self.online_return.isChecked());self.online_run_model=path;self.online_run_policy=self.online_policy;self.online_run_metadata={}
        from .online_process import OnlineProcess
        worker=OnlineProcess(path,self.online_paths,self._online_config(),self.online_source,live,self.online_speed.currentData(),self,policy=self.online_policy,heldout=heldout,include_return=self.online_return.isChecked(),live_truth=self.online_live_label.text().strip())
        self.online_worker=worker;worker.loaded.connect(self.online_receive_loaded);worker.observed.connect(self._queue_online_event);worker.record.connect(lambda row:self._queue_online_event(dict(type='record',record=row)))
        worker.failed.connect(lambda text:(self.online_state.setText('错误：'+text),self.online_current.setText('无法完成识别，请查看错误信息')));worker.finished.connect(self._online_finished)
        self._lock_online_policy(True);self.online_live_label.setEnabled(False);self.online_speed.setEnabled(False)
        self.online_current.setText(f'正在积累窗口（需{self.online_window}点）');self.online_summary.clear();self.online_start.setEnabled(False);self.online_source_mode.setEnabled(False);self.online_choose.setEnabled(False);self.online_clear.setEnabled(False);self.online_pause.setEnabled(not live);self.online_state.setText('连接中…' if live else '正在回放…');worker.start()

    def start_online_replay(self):self._launch_online()
    def connect_online_live(self):self._launch_online(True)
    def pause_online(self):
        worker=self.online_worker
        if worker and worker.isRunning():
            value=not worker.paused.is_set();worker.set_paused(value);self.online_pause.setText('继续' if value else '暂停')
    def stop_online(self):
        if self.online_worker:self.online_worker.stop()
    def _online_finished(self):
        self._online_finish_pending=True;self.online_start.setEnabled(False)
        if self._online_display_queue:
            self.online_state.setText('计算已结束，正在整理识别过程…');return
        self._finalize_online_display()

    def _finalize_online_display(self):
        self._online_finish_pending=False;self._render_online_records();self._render_online_summary()
        self._lock_online_policy(self.online_policy is not None);self.online_live_label.setEnabled(True);self.online_speed.setEnabled(self.online_source_mode.currentData()!='live')
        self.online_start.setEnabled(True);self.online_source_mode.setEnabled(True);self.online_choose.setEnabled(True);self.online_clear.setEnabled(True);self.online_pause.setText('暂停回放')
        if not self.online_state.text().startswith('错误：'):self.online_state.setText('已停止 / 已断开 · '+self.online_state.text())

    def online_receive_loaded(self,metadata):
        self.online_run_metadata=metadata

    def _queue_online_event(self,event):
        self._online_display_queue.append(event)

    def _flush_online_events(self,all_events=False):
        count=min(150,len(self._online_display_queue));self._online_batch_display=True
        events=self._online_display_queue[:count];del self._online_display_queue[:count]
        try:
            for event in events:self.online_receive_event(event)
        finally:self._online_batch_display=False
        if events:self._render_online_records()
        if not self._online_display_queue and getattr(self,'_online_finish_pending',False):self._finalize_online_display()

    def review_online_record(self):
        row=self.online_results.currentRow()
        if 0<=row and (self.online_page.value()-1)*100+row<len(self.online_records):
            record=self.online_records[(self.online_page.value()-1)*100+row];key=(record['path'],'step_003' if record['direction']=='out' else 'step_004')
            if key in self._online_traces:self.online_timeline.load(self._online_traces[key]);self.online_view.setCurrentIndex(0)
            else:self.online_state.setText('该条过程已释放，可选择原CSV单独回放查看；识别结果仍保留。')

    def online_receive_event(self,event):
        if event.get('type')=='record':self.online_receive_record(event['record']);return
        key=(event.get('path',''),event.get('stage','step_003'))
        if event.get('type') in ('trace','prediction'):
            trace=self._online_traces.setdefault(key,dict(points=[],predictions=[],labels=[],completion=None))
            if len(self._online_traces)>300:self._online_traces.pop(next(iter(self._online_traces)))
            if key!=self._online_trace_key:self._online_trace_key=key;self.online_timeline.load(trace)
            if event.get('type')=='trace':
                trace['points'].extend(event['points'])
                if len(trace['points'])>6000:trace['points']=trace['points'][::2]
                self.online_timeline.append_points(event['points']);return
            trace['predictions'].append(event);trace['labels']=getattr(self,'online_run_metadata',{}).get('labels',event.get('labels',[]));trace['completion']=event.get('completion')
            self.online_timeline.labels=trace['labels'];self.online_timeline.append_prediction(event)
        self.online_events.append(event)
        if event.get('type')=='waiting':
            self.online_state.setText('等待扫描 · 当前尚未进入有效扫描阶段')
            self.online_current.setText('请在采集程序运行原布料流程（“运行并采集”）；仅手动采集不会开始识别。')
            return
        status={'judging':'判断中','confirmed':'已确认','error':'数据错误','end-unconfirmed':'结束未确认','locked':'识别完成，结果已锁定'}.get(event.get('status'),event.get('status',''))
        self.online_state.setText(status+' · '+str(event.get('prediction') or event.get('reason','')))
        if event.get('type')!='prediction':
            if event.get('type') in ('quality','reset'):self.online_current.setText('数据中断或重置，重新积累窗口')
            elif event.get('type')=='end':
                if event.get('completion'):
                    done=event['completion'];self.online_current.setText(f"结果已锁定：{done['label']} · 有效历史{(done.get('effective_elapsed_s') or 0):.2f}秒 · X={done['x_mm']:.1f}mm")
                else:self.online_current.setText(('最终确认：'+str(event.get('prediction'))) if event.get('prediction') else '本次结束：未确认')
            return
        if event.get('status')=='locked':
            completion=event.get('completion') or {};effective=completion.get('effective_elapsed_s');timing=f' · 有效数据{effective:.2f}秒' if effective is not None else ''
            self.online_current.setText(f"结果已锁定：{event['prediction']}{timing} · X={event['x_mm']:.1f}mm")
        else:
            self.online_current.setText(f"正在判断：{event['prediction']} · 概率{event['probability']:.1%} · 连续{event.get('consecutive_count',0)}/{self.online_consecutive.value()}次 · 稳定{event.get('stable_elapsed_s',0):.1f}/{self.online_stability.value():.1f}秒")
        t=self.online_predictions;i=t.rowCount();t.insertRow(i)
        row_values(t,i,[f"{event['t_s']:.3f}",f"{event['x_mm']:.2f}",event['prediction'],f"{event['probability']:.3f}",f"{event['margin']:.3f}",status,f"{event.get('confirmed_label') or '—'} / {event['flip_count']}",f"{event.get('inference_latency_ms',0):.2f}"])
        if i>199:t.removeRow(0)

    def online_receive_record(self,row):
        self.online_records.append(row);self.online_page.setMaximum(max(1,(len(self.online_records)+99)//100))
        if not getattr(self,'_online_batch_display',False):self._render_online_records()
        if not self._online_summary_timer.isActive():self._online_summary_timer.start()

    def _render_online_records(self,*args):
        rows=self.online_records[(self.online_page.value()-1)*100:self.online_page.value()*100];self.online_results.blockSignals(True);self.online_results.setRowCount(len(rows))
        def seconds(value):return '—' if value is None else f'{value:.2f}'
        for i,row in enumerate(rows):
            row_values(self.online_results,i,[Path(row['path']).stem+' / '+{'out':'去','back':'回'}.get(row['direction'],row['direction']),row['true_label'] or '—',row['prediction'] or '—',{'correct':'正确','error':'错误','abstain':'未确认','unlabelled':'无标签'}.get(row['status'],row['status']),seconds(row.get('confirmation_elapsed_s')),seconds(row.get('effective_confirmation_s')),seconds(row.get('lead_before_crop_end_s'))])
        self.online_results.blockSignals(False)

    def _render_online_summary(self):
        summary=summarize_records(self.online_records);counts=summary['counts'];first=summary['first_confirmation_counts']
        lines=[f"累计 {summary['records']} 条单程",f"锁定/最终结果：正确 {counts['correct']} / 错误 {counts['error']} / 未确认 {counts['abstain']} / 无真实标签 {summary['unlabelled']}",f"首次确认：正确 {first['correct']} / 错误 {first['error']} / 未确认 {first['abstain']}",'','单程用时从当前方向开始计算；有效用时从首次进入有效范围计算。提前量相对有效范围末尾，仅在单程结束后生成。','各类表现（包含未确认样本）：']
        self.online_class_table.setRowCount(len(summary['classes']))
        for i,(name,metrics) in enumerate(summary['classes'].items()):
            lines.append(f"{name}：{metrics['count']} 条 · 精确率 {metrics['precision']:.1%} · 召回率 {metrics['recall']:.1%} · F1 {metrics['f1']:.3f}")
            row_values(self.online_class_table,i,[name,f"{metrics['precision']:.1%}",f"{metrics['recall']:.1%}",f"{metrics['f1']:.3f}",metrics['count']])
        self.online_summary.setPlainText('\n'.join(lines));self._draw_online_matrix()

    def _draw_online_matrix(self,*args):
        if not hasattr(self,'online_matrix') or self.online_tabs.currentIndex()!=4:return
        labels=getattr(self,'online_run_metadata',{}).get('labels',[])
        if not labels:self.online_matrix.empty('暂无已知标签结果');return
        matrix=[[0]*len(labels) for _ in labels];lookup={label:i for i,label in enumerate(labels)}
        for row in self.online_records:
            if row['true_label'] in lookup and row['prediction'] in lookup:matrix[lookup[row['true_label']]][lookup[row['prediction']]]+=1
        self.online_matrix.confusion(matrix,labels)

    def export_online(self):
        path,_=W.QFileDialog.getSaveFileName(self,'导出在线结果','online_results.json','JSON (*.json);;CSV (*.csv)')
        if not path:return
        value=dict(settings=getattr(self,'online_run_settings',asdict(OnlineConfig())),model_path=getattr(self,'online_run_model',self.online_model.currentData()),policy=getattr(self,'online_run_policy',self.online_policy),model_audit=getattr(self,'online_run_metadata',{}),records=list(self.online_records),summary=summarize_records(self.online_records),predictions=list(self.online_events),timelines=[dict(path=key[0],stage=key[1],points=list(t['points']),predictions=list(t['predictions']),labels=list(t['labels']),completion=t['completion']) for key,t in self._online_traces.items()])
        def save_json(target,contents,cancelled):
            target=Path(target);tmp=target.with_name(target.name+'.tmp')
            try:
                with tmp.open('w',encoding='utf-8') as f:
                    for n,chunk in enumerate(json.JSONEncoder(ensure_ascii=False,indent=2,allow_nan=False).iterencode(contents)):
                        if n%1024==0 and cancelled():raise ValueError('已取消导出')
                        f.write(chunk)
                if cancelled():raise ValueError('已取消导出')
                tmp.replace(target)
            finally:tmp.unlink(missing_ok=True)
        def save(cancelled):
            target=Path(path)
            if target.suffix.lower()=='.csv':
                rows=[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in value['records']]
                temp=target.with_name(target.name+'.tmp')
                try:
                    with temp.open('w',encoding='utf-8-sig',newline='') as f:
                        fields=list(dict.fromkeys(k for row in rows for k in row)) or ['path','status'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                        for row in rows:
                            if cancelled():raise ValueError('已取消导出')
                            writer.writerow(row)
                    temp.replace(target)
                finally:temp.unlink(missing_ok=True)
                save_json(target.with_suffix('.summary.json'),value,cancelled)
            else:save_json(target,value,cancelled)
            return str(target)
        if hasattr(self,'desktop_jobs') and self.desktop_jobs is not None:self.desktop_jobs.submit_thread('export-online',save,lambda target:self.online_state.setText('已导出：'+target),'导出当前识别结果')
        else:
            try:self.online_state.setText('已导出：'+save(lambda:False))
            except (OSError,ValueError) as e:self.online_state.setText('导出错误：'+str(e))


    def _lock_online_policy(self,locked):
        for widget in (self.online_experiment,self.online_model,self.online_stride,self.online_probability,self.online_margin,self.online_stability,self.online_smoothing,self.online_return,self.online_consecutive,self.online_lock):widget.setEnabled(not locked)
        if not locked and getattr(self,'online_is_prefix',False):self.online_lock.setChecked(True);self.online_lock.setEnabled(False)

    def _submit_online_policy(self,work,apply):
        jobs=getattr(self,'desktop_jobs',None)
        self._online_policy_generation=getattr(self,'_online_policy_generation',0)+1
        generation=self._online_policy_generation;path=self.online_model.currentData();settings=asdict(self._online_config())
        def done(value):
            if generation!=self._online_policy_generation or path!=self.online_model.currentData() or settings!=asdict(self._online_config()) or (self.online_worker and self.online_worker.isRunning()):return
            apply(value)
        self.online_state.setText('正在后台校验策略和模型权重，可继续浏览或取消等待…')
        if jobs is not None:jobs.submit_thread('online-policy',work,done,'校验在线策略与权重')
        else:
            try:done(work(lambda:False))
            except (OSError,ValueError,KeyError,TypeError) as e:self.online_state.setText('策略错误：'+str(e))

    def freeze_online_policy(self,after=None):
        if self.online_worker and self.online_worker.isRunning():self.online_state.setText('请先停止在线验证，再冻结策略。');return
        path=self.online_model.currentData()
        if not path:return
        target,_=W.QFileDialog.getSaveFileName(self,'保存冻结在线策略','online_policy.json','JSON (*.json)')
        if not target:return
        cfg=self._online_config()
        def work(cancelled):
            policy=freeze_policy(path,cfg,cancelled=cancelled)
            if cancelled():return None
            target_path=Path(target);temp=target_path.with_name(target_path.name+'.pending')
            try:
                temp.write_text(json.dumps(policy,ensure_ascii=False,indent=2),encoding='utf-8')
                if not cancelled():temp.replace(target_path)
            finally:temp.unlink(missing_ok=True)
            return policy
        def apply(policy):
            if policy is None:return
            self.online_policy=policy;self._lock_online_policy(True);self.online_state.setText('策略已冻结：'+policy['fingerprint'][:16]+'；可评价固定留出集。')
            if callable(after):after()
        self._submit_online_policy(work,apply)

    def load_online_policy(self):
        if self.online_worker and self.online_worker.isRunning():self.online_state.setText('请先停止在线验证，再载入策略。');return
        path=self.online_model.currentData()
        if not path:return
        target,_=W.QFileDialog.getOpenFileName(self,'载入与当前模型绑定的冻结策略','','JSON (*.json)')
        if not target:return
        window=self.online_window
        def work(cancelled):
            policy=read_json(target);cfg=validate_policy(policy,path,cancelled=cancelled)
            if cfg.window!=window:raise ValueError('冻结策略窗口与当前模型不一致')
            return policy,cfg
        def apply(value):
            policy,cfg=value;index=self.online_stride.findData(cfg.stride)
            if index<0:self.online_stride.addItem(str(cfg.stride),cfg.stride);index=self.online_stride.count()-1
            self.online_stride.setCurrentIndex(index);self.online_probability.setValue(cfg.probability);self.online_margin.setValue(cfg.margin);self.online_stability.setValue(cfg.stable_seconds);self.online_consecutive.setValue(cfg.consecutive);self.online_lock.setChecked(cfg.lock_on_confirm);self.online_smoothing.setChecked(cfg.smoothing);self.online_return.setChecked(cfg.include_return)
            self.online_policy=policy;self._lock_online_policy(True);self.online_state.setText('冻结策略已载入，门槛与模型已锁定。')
        self._submit_online_policy(work,apply)

    def unlock_online_policy(self):
        if self.online_worker and self.online_worker.isRunning():return
        self._online_policy_generation=getattr(self,'_online_policy_generation',0)+1
        jobs=getattr(self,'desktop_jobs',None)
        if jobs is not None:jobs.cancel('online-policy')
        self.online_policy=None;self._lock_online_policy(False);self.online_state.setText('已解除冻结；后续调整为探索性在线结果。')

    def evaluate_online_holdout(self):
        if not self.online_policy:self.online_state.setText('请先导出并冻结在线策略；留出集不会进入门槛调参。');return
        self._launch_online(heldout=True)
