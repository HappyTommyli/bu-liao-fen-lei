"""Focused desktop surface; legacy UI and model lifecycle remain reusable."""
from dataclasses import replace
from pathlib import Path
from PyQt5 import QtWidgets as W,QtCore as C
import torch
from .ui import TrainerWindow,label,button,panel,table,row_values,STATUS
from .modern_config import ModernConfig
from .tcn_models import default_spec,TCNClassifier
from .tcn_study import candidates,candidate_label,study_definition,study_description,description_text,MODE_TITLES,window_slices
from .tcn_settings import TCNParameterEditor,TCNSettingsDialog
from .config import PRESETS
from .plots import Plot
from .storage import list_tasks,read_json
from .tcn_recognition import TCNRecognitionMixin
from .online_ui import OnlineMixin


def focused_config():
    from .cera_models import default_cera_spec,PREFIX_REVISION
    return ModernConfig(architectures=('cera',),signals=('by',),tcn_spec=dict(default_cera_spec(),revision=PREFIX_REVISION),tcn_mode='prefix',tcn_stride=900,cv_folds=1)


class ScrollDetails(W.QTextBrowser):
    def setText(self,text):self.setPlainText(str(text))


class TCNTrainerWindow(OnlineMixin,TCNRecognitionMixin,TrainerWindow):
    def __init__(self,store=None):
        self.project_read_only=False;self._read_only_value=None
        super().__init__(store,config=focused_config())
        online=self.build_online();online.layout().setSizeConstraint(W.QLayout.SetMinimumSize)
        self.online_scroll=W.QScrollArea();self.online_scroll.setWidgetResizable(True);self.online_scroll.setWidget(online)
        self.pages.removeWidget(self.pages.widget(3));self.pages.addWidget(self.online_scroll);self.navigation.item(3).setText('04    在线识别')
        self._layout_timer=C.QTimer(self);self._layout_timer.setSingleShot(True)
        self._layout_timer.timeout.connect(self._responsive)
        self.setWindowTitle('TCN / CERA 布料分类 · |dBy|');self.setMinimumSize(1024,640)
        self.navigation.item(3).setText('04    在线识别')
        self.navigation.item(1).setText('02    模型训练');self.navigation.item(2).setText('03    实验结果')
        sidebar=self.findChild(W.QWidget,'sidebar')
        for text in sidebar.findChildren(W.QLabel):
            if text.text()=='CERA-NET':text.setText('TCN / CERA')
            elif 'GRU · TCN · CERA' in text.text():text.setText('单通道 |dBy|\n333 Hz · GPU训练')
            elif text.text()=='分类训练 · 模型库 · 识别':text.setText('数据 · 训练 · 在线识别')
        rect=W.QApplication.primaryScreen().availableGeometry()
        self.resize(min(1420,rect.width()),min(900,rect.height()))
        self.update_selection();self.navigate(self.navigation.currentRow());self._layout_timer.start(0)

    def navigate(self,index):
        if index<0 or index>4:return
        if index==4:index=3
        self.pages.setCurrentIndex(index)
        self.page_title.setText(['数据','模型训练','实验结果','在线识别'][index])
        self.subtitle.setText(['导入原始CSV，检查单通道dBy与去回程；训练输入取绝对值。',
                               '固定输入 |dBy| · 累积前缀 CERA · 原始采集组不拆分。',
                               '查看训练与验证曲线；单次最佳权重可直接在线验证，开发集重训为可选。',
                               '使用累计历史判断，达到条件后输出并锁定；采集继续正常进行。'][index])
        self._responsive()

    def build_data(self):
        page=super().build_data()
        self.preview_signal.clear();self.preview_signal.addItem('dBy','by')
        self.preview_mode.setItemText(0,'原始dBy（带符号）');self.preview_mode.setItemText(1,'模型输入 |dBy|')
        old=self.data_details;layout=old.parentWidget().layout()
        details=ScrollDetails();details.setMinimumHeight(72);details.setMaximumHeight(110)
        layout.replaceWidget(old,details);old.hide();self.data_details=details
        # Put secondary data operations in a menu, keeping narrow screens usable.
        top=page.layout().itemAt(0).layout()
        for i in range(top.count()-1,-1,-1):
            widget=top.itemAt(i).widget()
            if isinstance(widget,W.QPushButton) and widget.text() in ('指定标签','移除所选'):widget.hide()
        more=W.QPushButton('更多');menu=W.QMenu(more)
        menu.addAction('指定标签',self.label_selected);menu.addAction('移除所选',self.remove_selected)
        menu.addAction('采集协议…',lambda:self.advanced(True));more.setMenu(menu)
        top.insertWidget(3,more)
        return page

    def build_training(self):
        page=W.QWidget();layout=W.QVBoxLayout(page);layout.setContentsMargins(0,0,0,0);layout.setSpacing(10)
        header=W.QHBoxLayout();self.task_name=W.QLineEdit();self.task_name.setPlaceholderText('实验名称，例如：900点 · 128通道')
        header.addWidget(self.task_name,1);self.task_count=label('1项模型训练','muted');self.task_count.setWordWrap(False);header.addWidget(self.task_count);layout.addLayout(header)
        self.training_splitter=W.QSplitter(C.Qt.Horizontal);layout.addWidget(self.training_splitter,1)
        self.parameter_scroll=W.QScrollArea();self.parameter_scroll.setWidgetResizable(True)
        self.parameter_scroll.setHorizontalScrollBarPolicy(C.Qt.ScrollBarAlwaysOff)
        self.editor=TCNParameterEditor(self.config);self.parameter_scroll.setWidget(self.editor)
        self.parameter_scroll.setMinimumWidth(300);self.training_splitter.addWidget(self.parameter_scroll)
        monitor=W.QWidget();ml=W.QVBoxLayout(monitor);ml.setContentsMargins(12,0,0,0)
        self.training_state=label('等待训练','section');ml.addWidget(self.training_state)
        self.monitor_tabs=W.QTabWidget();ml.addWidget(self.monitor_tabs,1)
        curve_page=W.QWidget();curve_layout=W.QVBoxLayout(curve_page);curve_layout.setContentsMargins(0,0,0,0)
        curve_bar=W.QHBoxLayout();curve_bar.addWidget(label('横轴'))
        self.live_axis=W.QComboBox();self.live_axis.addItem('训练轮数','epoch');self.live_axis.addItem('累计更新次数','updates');self.live_axis.setCurrentIndex(1)
        curve_bar.addWidget(self.live_axis);curve_bar.addStretch();curve_bar.addWidget(button('导出当前曲线',self.export_live_curve));curve_layout.addLayout(curve_bar)
        self.live_plot=Plot();self.live_plot.training_x_axis='updates';curve_layout.addWidget(self.live_plot,1)
        self.live_axis.currentIndexChanged.connect(lambda:self.live_plot.set_training_axis(self.live_axis.currentData()))
        self.monitor_tabs.addTab(curve_page,'曲线')
        task_page=W.QWidget();task_layout=W.QVBoxLayout(task_page)
        self.candidate_preview=ScrollDetails();task_layout.addWidget(self.candidate_preview)
        self.monitor_tabs.addTab(task_page,'任务与配置')
        self.training_log=W.QPlainTextEdit();self.training_log.setReadOnly(True);self.training_log.setMaximumBlockCount(2000)
        self.monitor_tabs.addTab(self.training_log,'日志')
        self.training_splitter.addWidget(monitor);self.training_splitter.setSizes([365,750])
        summary,sl=panel('数据与配置')
        self.training_summary=label('尚未选择数据','muted');sl.addWidget(self.training_summary)
        self.partition_summary=label('导入数据后检查分组覆盖','muted');sl.addWidget(self.partition_summary)
        self.model_summary=label('TCN · |dBy|','muted');sl.addWidget(self.model_summary)
        # Summary itself scrolls instead of squeezing long audit/configuration messages.
        self.summary_scroll=W.QScrollArea();self.summary_scroll.setWidgetResizable(True);self.summary_scroll.setWidget(summary)
        self.summary_scroll.setMinimumHeight(84);self.summary_scroll.setMaximumHeight(122);layout.addWidget(self.summary_scroll)
        actions=W.QHBoxLayout();self.check_button=button('检查配置',self.check_configuration)
        self.start_button=button('开始训练',self.start_training,True);self.cancel_button=button('取消',self.cancel_worker);self.cancel_button.setEnabled(False)
        self.new_project_button=button('以TCN新建实验',self.new_tcn_from_project);self.new_project_button.hide()
        for widget in (self.check_button,self.start_button,self.cancel_button,self.new_project_button):actions.addWidget(widget)
        self.progress=W.QProgressBar();self.progress.setRange(0,100);self.progress.setValue(0);actions.addWidget(self.progress,1)
        layout.addLayout(actions)
        # Retain controller hooks without exposing irrelevant architecture/input choices.
        self.arch_checks={'tcn':W.QCheckBox(self)};self.arch_checks['tcn'].setChecked(True);self.arch_checks['tcn'].hide()
        self.signal_checks={'by':W.QCheckBox(self)};self.signal_checks['by'].setChecked(True);self.signal_checks['by'].hide()
        self.last_readout=W.QCheckBox(self);self.last_readout.hide()
        self.preset_selector=W.QComboBox(self)
        for key,title in PRESETS.items():self.preset_selector.addItem(title,key)
        self.preset_selector.hide()
        self.log_toggle=W.QCheckBox(self);self.log_toggle.hide()
        self.log_toggle.toggled.connect(lambda shown:self.monitor_tabs.setCurrentWidget(self.training_log) if shown else None)
        self.editor.changed.connect(self.parameters_changed)
        return page

    def build_history(self):
        page=super().build_history()
        self.final_button.setText('可选：开发集重训')
        self.history_splitter=page.layout().itemAt(0).widget()
        self.library_tabs.setTabText(0,'模型实验');self.library_tabs.setTabVisible(1,False)
        self.detail_tabs.setTabVisible(1,False);self.detail_tabs.setTabText(2,'完整记录');self.detail_tabs.setTabText(3,'验证汇总')
        self.spec_details=ScrollDetails();self.detail_tabs.addTab(self.spec_details,'参数详情')
        windows=W.QWidget();wl=W.QVBoxLayout(windows)
        self.window_notice=label('相关窗口辅助结果，不能视为新增独立采集。','muted');wl.addWidget(self.window_notice)
        self.window_page=W.QSpinBox();self.window_page.setRange(1,1);self.window_page.setPrefix('页 ');wl.addWidget(self.window_page)
        self.window_table=table(['采集 / 方向','窗口索引','有效点数','真实布料','窗口预测','最高概率']);wl.addWidget(self.window_table,1);self.window_page.valueChanged.connect(self._draw_window_page)
        self.detail_tabs.addTab(windows,'窗口结果')
        axis_bar=W.QHBoxLayout();axis_bar.addWidget(label('曲线横轴'))
        self.result_axis=W.QComboBox();self.result_axis.addItem('训练轮数','epoch');self.result_axis.addItem('累计更新次数','updates');self.result_axis.setCurrentIndex(1)
        self.result_curve.training_x_axis='updates';self.result_axis.currentIndexChanged.connect(lambda:self.result_curve.set_training_axis(self.result_axis.currentData()))
        axis_bar.addWidget(self.result_axis);axis_bar.addStretch()
        result_layout=self.detail_tabs.parentWidget().layout();result_layout.insertLayout(result_layout.indexOf(self.detail_tabs),axis_bar)
        self.result_tabs.setTabVisible(4,False);self.result_partition.hide();self.result_aggregate.hide()
        self.comparison.setHorizontalHeaderLabels(['配置','参数量','验证F1','训练F1','感受野'])
        self.comparison.setMinimumHeight(150);self.comparison.setMaximumHeight(210)
        self.comparison.horizontalHeader().setSectionResizeMode(0,W.QHeaderView.Stretch)
        for col in range(1,5):self.comparison.horizontalHeader().setSectionResizeMode(col,W.QHeaderView.ResizeToContents)
        for widget in page.findChildren(W.QPushButton):
            if widget.text() in ('导入成果包','重命名'):widget.hide()
            elif widget.text()=='加载识别':widget.setText('在线识别')
        # The results panel has many plots and controls; give it its own scrollbar.
        right=self.history_splitter.widget(1);right.setParent(None);right.setObjectName('tcnResults')
        right.layout().setSizeConstraint(W.QLayout.SetMinimumSize)
        scroll=W.QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(right)
        self.history_splitter.addWidget(scroll);self.history_splitter.setSizes([225,900])
        return page

    def selected_signals(self):return ('by',)

    def controls_config(self):
        if self.project_read_only:raise ValueError('旧项目在TCN工作区只读；请以TCN新建实验')
        return replace(self.editor.value(self.config),device='cuda',signals=('by',),
                       readouts=('mean',),name=self.task_name.text().strip()).validate()

    def parameters_changed(self):
        try:self.config=self.editor.value(self.config)
        except ValueError:pass
        self.update_selection()

    def update_selection(self,*args):
        super().update_selection(*args)
        if not hasattr(self,'editor'):return
        self.training_summary.setText(self.data_summary.text().split('\n')[-1])
        self.training_summary.setToolTip(self.data_summary.text())
        if not self.records and not self.errors and not self.batches:
            self.partition_summary.setText('请先导入CSV或载入 |dBy| 数据版本。')
        self.partition_summary.setToolTip(self.partition_summary.text())
        if self.project_read_only:
            self.start_button.setEnabled(False);self.training_summary.setText('已打开不兼容的旧项目，仅供查看；可新建TCN实验。')
        self.update_task_count()

    def update_task_count(self,*args):
        if not hasattr(self,'editor'):return
        try:
            cfg=self.editor.value(self.config);cc=candidates(cfg);n=len(cc)*cfg.cv_folds*cfg.repetitions
            self.task_count.setText(f'{n}项训练 · 每折最多{cfg.epochs}轮')
            self.task_count.setMinimumWidth(self.task_count.fontMetrics().horizontalAdvance(self.task_count.text())+6)
            selected=self.selected_records()
            classes=len({r.label for r in selected}) or 15
            parent_lengths=[len(t.values) for r in selected for t in (r.views.get('by') or r.trials)]
            lines=[]
            for candidate in cc:
                from .model_summary import parameter_summary
                summary=parameter_summary(candidate['spec'],classes);count=summary['parameters'];rf=summary['receptive_field']
                lines.append(f"{candidate_label(candidate)}\n  参数 {count:,} · 感受野 {rf}点 / 约{rf/333:.2f}秒")
                spec=candidate['spec']
                lines.append(f"  卷积核 {spec['kernel_size']} · Dropout {spec['dropout']} · 汇总 {spec['pooling']}\n  膨胀：{', '.join(map(str,spec['dilations']))}")
                window=candidate['window_size']
                count_windows=sum(len(window_slices(length,window,candidate.get('window_stride',0))) for length in parent_lengths)
                lines.append(f"  {len(parent_lengths)}条原始单程 → {count_windows}个窗口 / 片段")
                if window:lines.append(f"  窗口时长约{window/333:.2f}秒；尾窗按实际有效长度。")
                elif parent_lengths:lines.append(f"  单程时长约{min(parent_lengths)/333:.2f}–{max(parent_lengths)/333:.2f}秒。")
                if cfg.architectures==('cera',):lines.append('  GRU及累积模块使用当前窗口历史；上方感受野仅指TCN分支。')
                if candidate['window_size'] and rf>candidate['window_size']:
                    lines.append('  感受野超过窗口；窗口外上下文为左侧补零。')
            suffix='' if self.selected_records() else '（暂按15类估算）'
            self.model_summary.setText(MODE_TITLES[cfg.tcn_mode]+' · '+lines[0].split('\n')[1].strip()+suffix)
            heading=description_text(study_description(study_definition(cfg)))
            dataset=(f'数据集版本：{self.snapshot_manifest["dataset_id"][:10]}' if self.snapshot_manifest else '数据集：当前导入选择，训练前冻结')
            validation='单次划分（60%训练 / 20%验证 / 20%测试）' if cfg.cv_folds==1 else f'{cfg.cv_folds}折开发集验证（20%测试封存）'
            overview=f'\n\n{dataset} · {len({r.label for r in selected})}类 · {len(selected)}次采集\n任务：{len(cc)}组 × {cfg.cv_folds}次划分 × {cfg.repetitions}次 = {n}项\n{validation}\n最多{cfg.epochs}轮 · 早停{cfg.patience}轮（'+('开启' if cfg.early_stopping else '关闭')+f'）\n学习率 {cfg.lr:g} · 有效批 {cfg.batch_size}条原始单程/更新 · 微批 {cfg.micro_batch or "自动预检"}'
            if not self.busy():self.candidate_preview.setPlainText(heading+'\n\n实际配置明细\n\n'+'\n\n'.join(lines)+overview+'\n窗口独立监督，单程按有效点数加权平均预测概率；所有组共享数据和训练参数。'+suffix)
        except (ValueError,TypeError) as exc:
            self.task_count.setText('配置待修正');self.model_summary.setText(str(exc))
            if not self.busy():self.candidate_preview.setPlainText(str(exc))
            self.start_button.setEnabled(False)

    def export_live_curve(self):
        path,_=W.QFileDialog.getSaveFileName(self,'导出当前训练曲线','tcn-training.png','PNG (*.png)')
        if path:self.live_plot.figure.savefig(path,dpi=180);self.notify('已导出当前横轴的训练曲线')

    def check_configuration(self):
        try:
            cfg=self.controls_config();self.notify('模型配置有效；'+self.task_count.text())
            self.monitor_tabs.setCurrentWidget(self.candidate_preview.parentWidget())
        except ValueError as exc:self.notify(str(exc))

    def sync_controls(self):
        if hasattr(self,'editor'):self.editor.set_config(self.config)
        if hasattr(self,'task_name'):self.task_name.setText(self.config.name)
        self.update_selection()

    def advanced(self,protocol=False):
        if protocol:
            if self.project_read_only:self.notify('非TCN旧项目仅供查看；请新建TCN实验后导入CSV。');return
            return super().advanced(True)
        if self.busy() or self.import_jobs:return
        dialog=TCNSettingsDialog(self.controls_config(),self)
        if dialog.exec_()==W.QDialog.Accepted:
            self.config=dialog.result_config;self.sync_controls()

    def _responsive(self):
        if not hasattr(self,'training_splitter'):return
        narrow=self.pages.width()<1000
        for name in ('training_splitter','history_splitter'):
            splitter=getattr(self,name,None)
            if splitter is not None:
                # 300px parameters + monitor still fit at the supported 1024px minimum.
                # Stacking both at 640px height made the parameter viewport unusably short.
                vertical=narrow if name=='history_splitter' else self.pages.width()<740
                direction=C.Qt.Vertical if vertical else C.Qt.Horizontal
                if splitter.orientation()!=direction:
                    splitter.setOrientation(direction)
                    splitter.setSizes(([160,600] if vertical else [225,900]) if name=='history_splitter' else ([300,500] if vertical else [365,800]))
        for splitter in self.pages.widget(0).findChildren(W.QSplitter):
            splitter.setOrientation(C.Qt.Vertical if narrow else C.Qt.Horizontal)

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,'_layout_timer'):self._layout_timer.start(0)

    def on_event(self,event):
        event=dict(event)
        if event.get('model'):
            task=Path(event['task']) if event.get('task') else self.active_task
            if event.get('type')=='phase' or getattr(self,'_event_model_key',None)!=(str(task),event['model']):
                self._event_model_key=(str(task),event['model'])
                self._event_metadata=self.task_metadata(task) if task else None
            meta=self._event_metadata
            study=(meta or {}).get('study')
            if study:
                desc=(meta or {}).get('study_description') or study_description(study)
                identity=event.get('configuration_id',event['model'])
                match=next((g for g in desc['groups'] if g['candidate_id'] in identity),None)
                count=(self.result_task or {}).get('config',{}).get('cv_folds',self.config.cv_folds)
                scope='最终重训' if event.get('scope')=='final' else ('单次训练' if count==1 else f"第{event.get('fold',0)+1}/{count}折")
                context=f"{desc['title']} · {match['group']+'组' if match else event['model']} · {scope}"
                event['message']=context+' · '+event.get('message','')
                if event.get('type')=='phase':
                    cfg=meta['config']
                    budget=(f'最终重训固定{event["epochs"]}轮；验证F1不适用，早停不适用' if event.get('scope')=='final' else f'最多{cfg["epochs"]}轮 · 早停{cfg["patience"]}轮（'+('开启' if cfg['early_stopping'] else '关闭')+'）')
                    frozen=f'数据集版本：{meta["dataset_id"][:10]} · {len(meta["labels"])}类 · {meta["records"]}次采集\n任务 {event.get("job",1)}/{event.get("jobs",1)}\n'+budget+f'\n学习率 {cfg["lr"]:g} · 有效批 {cfg["batch_size"]}条单程'
                    self.candidate_preview.setPlainText('当前运行：'+context+'\n\n'+description_text(desc)+'\n\n'+frozen)
        super().on_event(event)
        self.training_state.setText(self.training_state.text().replace('完整段','TCN'))
        self.training_state.setToolTip(self.training_state.text())

    def refresh_history(self):
        super().refresh_history()
        if hasattr(self,'online_experiment'):self.refresh_online_models()
        self.refresh_recognition_models()
        for widget in (self.history,self.legacy_history,self.task_history):
            for i in range(widget.count()-1,-1,-1):
                item=widget.item(i)
                meta=self.task_metadata(Path(item.data(C.Qt.UserRole)))
                config=meta.get('config',{})
                if config.get('architectures') not in (['tcn'],['cera']) or config.get('signals')!=['by']:
                    widget.takeItem(i)
        self.legacy_history.clear()

    def show_results(self,task):
        super().show_results(task)
        from .ui import row_values
        self.experiment_title.setText(self.result_task['name'])
        for i,row in enumerate(self.catalog_rows):
            results=row['results'];first=results[0]
            train=sum(r['train_metrics']['macro_f1'] for r in results)/len(results) if all(r.get('train_metrics') for r in results) else None
            val='—' if row['val_mean'] is None else f"{row['val_mean']:.3f} ±{row['val_std']:.3f}"
            row_values(self.comparison,i,[first.get('candidate_label','TCN · 完整单程'),f"{first.get('parameters',0):,}",val,
                                          '—' if train is None else f'{train:.3f}',str(first.get('receptive_field','—'))+'点'])
        self.comparison.setColumnHidden(4,False)

    def draw_result(self,*args):
        super().draw_result(*args)
        if not hasattr(self,'spec_details') or not self.result_model:return
        row=self.result_model;candidate=row.get('candidate')
        source=(self.result_task.get('holdout') or {}).get('predictions',[]) if self.artifact_scope.currentIndex()==1 else row.get('val_predictions',[])
        window_rows=[(parent,window) for parent in source for window in parent.get('windows',[])]
        self._window_rows=window_rows;self.window_page.setMaximum(max(1,(len(window_rows)+99)//100));self._draw_window_page()
        self.window_notice.setText(f'{len(source)}条原始单程 · {len(window_rows)}个前缀/窗口；每页100条。主指标以原单程为单位。')
        if candidate is None:
            self.spec_details.setPlainText('旧版TCN：完整结构见完整记录。');return
        spec=candidate['spec'];config=self.result_task['config']
        study=self.result_task.get('study')
        context=description_text(self.result_task.get('study_description') or study_description(study))+'\n\n' if study else ''
        if study and study.get('mode')=='overlap_pair':
            delta=self.result_task.get('contrasts',{}).get('B-A',{}).get('mean')
            if delta is not None:context+=f'配对验证：B−A（重叠−不重叠）单程 Macro-F1 差值 {delta:+.4f}\n\n'
        pooling={'mean':'有效特征平均','mean_std':'平均＋标准差','max':'有效特征最大值'}[spec['pooling']]
        self.spec_details.setPlainText(
            context+f"{candidate_label(candidate)}\n\n"
            f"通道列表：{', '.join(map(str,spec['channels']))}\n卷积核：{spec['kernel_size']}\n"
            f"膨胀系数：{', '.join(map(str,spec['dilations']))}\nDropout：{spec['dropout']}\n特征汇总：{pooling}\n"
            f"参数量：{row['parameters']:,}\n感受野：{row['receptive_field']}点\n\n"
            f"初始学习率：{config['lr']}\n权重衰减：{config['weight_decay']}\n有效批：{config['batch_size']}条原始单程\n"
            f"梯度裁剪上限：{config.get('grad_clip_norm',1)}\n" +
            ('训练：每单程随机取最多2个累积前缀，单程内等权；每父批累积后一次更新。\n评价：使用最后一个累积前缀；主指标以原单程为单位。' if candidate.get('prefix_policy') else '训练：每窗交叉熵按有效长度加权；每父批累积后一次更新。\n评价：窗口概率按有效点数平均；主指标以原单程为单位。'))

    def _draw_window_page(self,*args):
        rows=getattr(self,'_window_rows',[]);start=(self.window_page.value()-1)*100;visible=rows[start:start+100];self.window_table.setRowCount(len(visible))
        for i,(parent,window) in enumerate(visible):
            labels=self.result_task['labels'];predicted=window['predicted']
            row_values(self.window_table,i,[Path(parent['path']).stem+(' / 去' if parent['direction']=='out' else ' / 回'),f"{window['start']}:{window['end']}",window['points'],parent['label'] or '未知',labels[predicted],f"{window['scores'][predicted]:.1%}"])

    def read_project(self,path):
        value=read_json(path);config=value.get('config',{})
        if (config.get('architectures') not in (['tcn'],['cera']) or config.get('signals')!=['by'] or config.get('data',{}).get('preprocess_version')!='abs-v3'
                or (config.get('architectures')==['cera'] and config.get('tcn_spec') is None)):
            self.project_read_only=True;self._read_only_value=value;self.new_project_button.show()
            self.navigation.setCurrentRow(1);self.update_selection();self.notify('旧项目只读；以TCN新建实验后重新审查数据。');return
        self.project_read_only=False;self.new_project_button.hide()
        super().read_project(path)
        if self.config.tcn_spec is None:
            spec=default_spec(self.config.hidden_size);spec['dropout']=self.config.dropout
            self.config=replace(self.config,tcn_spec=spec)
        self.sync_controls()

    def new_tcn_from_project(self):
        value=self._read_only_value
        if value is None:return
        self.project_read_only=False;self.new_project_button.hide();self.config=focused_config();self.sync_controls()
        self.records=[];self.errors=[];self.batches=[];self.import_reports=[];self.snapshot_path=None;self.snapshot_manifest=None;self.data_table.setRowCount(0)
        self.overrides=value.get('overrides',{});sources=value.get('sources',[])
        self.import_paths(value.get('review_candidates') or [s['path'] for s in sources],
                          [s['path'] for s in sources if s.get('selected')])

    def load_dataset_version(self,path):
        manifest=read_json(Path(path)/'dataset.json')
        if manifest.get('signals')!=['by'] or manifest.get('data',{}).get('preprocess_version')!='abs-v3':
            raise ValueError('当前工作区只支持abs-v3 / |dBy|快照；其他快照保留原样')
        super().load_dataset_version(path)

    def resume_selected(self):
        if self.result_task and (self.result_task.get('config',{}).get('architectures') not in (['tcn'],['cera']) or self.result_task.get('config',{}).get('signals')!=['by']):
            self.notify('当前工作区仅继续TCN/CERA / |dBy|任务');return
        super().resume_selected()

    def closeEvent(self,event):
        if getattr(self,'online_worker',None) is not None and self.online_worker.isRunning():
            self.stop_online()
            if self.online_worker.isRunning():
                self.notify('正在停止在线验证…');event.ignore();C.QTimer.singleShot(200,self.close);return
        super().closeEvent(event)


# Keep the legacy controller implementation, with nonblocking focused overrides.
from .responsive_ui import ResponsiveMixin
LegacyTCNTrainerWindow=TCNTrainerWindow
class TCNTrainerWindow(ResponsiveMixin,LegacyTCNTrainerWindow):
    pass
