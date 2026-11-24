"""Four-page classification workbench; workers keep the desktop responsive."""
from pathlib import Path
from dataclasses import replace
from collections import Counter
from datetime import datetime
import json
import sys
import uuid
from PyQt5 import QtWidgets as W,QtCore as C,QtGui as G
import PyQt5

# Qt5's wheel prefix lookup can corrupt non-ASCII virtualenv paths. Register
# the real Unicode directory before constructing QApplication (also for tests).
C.QCoreApplication.addLibraryPath(str(Path(PyQt5.__path__[0])/'Qt5/plugins'))
from .config import DataConfig,TrainConfig,SIGNALS,VARIANTS,ARCHITECTURES,READOUTS,PRESETS,input_label
from .modern_config import ModernConfig,ARCHITECTURES
from .modern_split import make_plan,validate_plan
from .settings_ui import SettingsDialog
from .catalog import configurations
from .data import import_dataset,prepare_trials,protocol_for,required_axes,discover_report
from .folds import make_folds,fold_partition
from .dataset import save_dataset,load_dataset
from .quality import QUALITY
from .quality import current_rule
from .admission import review_batch,package,require_admission
from .evaluation import classification_metrics
from .storage import read_json,write_json,list_tasks,export_task,import_task
from .plots import Plot
from .theme import STYLE

ROOT=Path(__file__).resolve().parents[1]
STATUS={'completed':'已完成','failed':'失败','cancelled':'已取消','running':'运行中／中断待检查',
        'insufficient_evidence':'风险目标证据不足','certified':'通过风险审查','fixed_length':'固定长度对照','exploratory':'探索性，无风险认证'}

def label(text,name=None):
    w=W.QLabel(text);w.setWordWrap(True)
    if name:w.setObjectName(name)
    return w

def button(text,fn,primary=False):
    w=W.QPushButton(text);w.clicked.connect(fn);w.setCursor(C.Qt.PointingHandCursor)
    if primary:w.setObjectName('primary')
    return w

def table(headers):
    w=W.QTableWidget(0,len(headers));w.setHorizontalHeaderLabels(headers);w.verticalHeader().hide()
    w.setAlternatingRowColors(True);w.setSelectionBehavior(W.QAbstractItemView.SelectRows);w.setSelectionMode(W.QAbstractItemView.ExtendedSelection)
    w.setEditTriggers(W.QAbstractItemView.NoEditTriggers);w.horizontalHeader().setSectionResizeMode(W.QHeaderView.Stretch);w.verticalHeader().setDefaultSectionSize(38)
    return w

def row_values(w,row,values):
    for j,v in enumerate(values):
        item=W.QTableWidgetItem(str(v));item.setToolTip(str(v));w.setItem(row,j,item)

def panel(title):
    w=W.QFrame();w.setObjectName('panel');layout=W.QVBoxLayout(w);layout.setContentsMargins(16,14,16,14);layout.addWidget(label(title,'section'));return w,layout

def percent(v):return 'N/A' if v is None else f'{v:.1%}'

def audit_text(d):
    return f'完整差异率{percent(d.get("full_rate"))} · 共同差异率{percent(d.get("common_rate"))} · 同向位置箱{percent(d.get("same_sign_fraction"))}\n共同幅值差{d.get("common_difference","无法判断")} µT · 阈值{d.get("amplitude_threshold_uT","缺噪声")} µT · 共同箱{d.get("bins",0)}'

class ImportThread(C.QThread):
    imported=C.pyqtSignal(object,object)
    progress=C.pyqtSignal(int,int)
    def __init__(self,paths,config,overrides,parent,signals=None,batch_scopes=None,initial_ignored=None):
        super().__init__(parent);self.paths=paths;self.config=config;self.overrides=overrides;self.signals=signals;self.batch_scopes=batch_scopes;self.initial_ignored=initial_ignored or []
    def run(self):
        try:
            records=[];reports=[]
            for scope in self.batch_scopes or [self.paths]:
                rr,rp=import_dataset(scope,self.config,self.signals,self.overrides,self.progress.emit);records.extend(rr);reports.append(rp)
            errors=dict(candidate_count=sum(r['candidate_count'] for r in reports),usable_count=len(records),ignored_count=sum(r['ignored_count'] for r in reports))
            for key in ('errors','ignored','duplicates','review_batches'):errors[key]=[e for r in reports for e in r.get(key,[])]
            errors['ignored'].extend(self.initial_ignored);errors['ignored_count']=len(errors['ignored'])
        except Exception as exc:records,errors=[],dict(errors=[dict(path='导入',message=str(exc))])
        self.imported.emit(records,errors)

class TrainerWindow(W.QMainWindow):
    def __init__(self,store=None,config=None):
        super().__init__();self.store=Path(store or ROOT/'outputs/models');self.store.mkdir(parents=True,exist_ok=True)
        self.config=config or ModernConfig();self.records=[];self.errors=[];self.overrides={};self.screening={};self.batches=[];self.import_reports=[];self.exclusions=[];self.snapshot_path=None;self.snapshot_manifest=None;self.import_jobs=[];self.pending_close=False
        self.active_request=None;self.active_task=None;self.cancelled=False;self.worker_kind=None;self.buffer='';self.stderr='';self.live_history=[]
        self.result_task=None;self.result_model=None;self.result_rows=[];self.prediction_data=None;self.prediction_rows=[];self.prediction_path=None
        self.recognition_paths=[];self.process=C.QProcess(self);self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.readyReadStandardError.connect(self.read_error);self.process.finished.connect(self.worker_finished);self.process.errorOccurred.connect(self.worker_error)
        self.setWindowTitle('布料分类训练 · 第一阶段');self.setStyleSheet(STYLE);self.resize(1420,920);self.setMinimumSize(1120,760)
        shell=W.QWidget();self.setCentralWidget(shell);outer=W.QVBoxLayout(shell);outer.setContentsMargins(0,0,0,0);outer.setSpacing(0)
        body=W.QHBoxLayout();body.setSpacing(0);outer.addLayout(body,1)
        sidebar=W.QWidget();sidebar.setObjectName('sidebar');sidebar.setFixedWidth(190);sl=W.QVBoxLayout(sidebar);sl.setContentsMargins(14,24,14,18)
        sl.addWidget(label('CERA-NET','eyebrow'));sl.addWidget(label('布料分类','brand'));sl.addWidget(label('分类训练 · 模型库 · 识别','muted'))
        self.navigation=W.QListWidget();self.navigation.setObjectName('navigation');self.navigation.addItems(['01    数据集','02    训练','03    模型库','04    模型识别']);sl.addWidget(self.navigation,1)
        sl.addWidget(label('GRU · TCN · CERA\n完整接触段分类','muted'));sl.addWidget(label('离线工作台\n原始CSV只读','muted'));body.addWidget(sidebar)
        content=W.QWidget();cl=W.QVBoxLayout(content);cl.setContentsMargins(24,22,24,16);body.addWidget(content,1)
        header=W.QHBoxLayout();self.page_title=label('数据集','title');header.addWidget(self.page_title);header.addStretch()
        header.addWidget(button('打开项目',self.open_project));header.addWidget(button('保存项目',self.save_project));cl.addLayout(header)
        self.subtitle=label('导入原始CSV，自动检查去回程与有效接触段。','subtitle');cl.addWidget(self.subtitle)
        self.pages=W.QStackedWidget();cl.addWidget(self.pages,1)
        for builder in [self.build_data,self.build_training,self.build_history,self.build_prediction]:self.pages.addWidget(builder())
        self.status=label('就绪 · 导入数据后即可准备训练','status');outer.addWidget(self.status)
        self.navigation.currentRowChanged.connect(self.navigate);self.navigation.setCurrentRow(0);self.refresh_history();self.update_selection()

    def task_catalog(self):return list_tasks(self.store)
    def task_metadata(self,directory):return read_json(Path(directory)/'metadata.json')

    def busy(self):return self.process.state()!=C.QProcess.NotRunning
    def notify(self,text):self.status.setText(str(text))
    def error(self,text):self.notify(text);W.QMessageBox.warning(self,'操作未完成',str(text))
    def navigate(self,i):
        if i<0:return
        self.pages.setCurrentIndex(i);self.page_title.setText(['数据集','训练','模型库','模型识别'][i])
        self.subtitle.setText(['导入后自动检查；同一CSV的两个单程始终同组。','第一阶段：只训练完整接触段分类。','按实验查看模型配置；复验归组，第二阶段后续开放。','选择一个具体模型，按保存协议识别新CSV。'][i])

    def build_data(self):
        page=W.QWidget();layout=W.QVBoxLayout(page);layout.setContentsMargins(0,0,0,0)
        top=W.QHBoxLayout();top.addWidget(button('＋ 导入CSV',self.choose_files,True));top.addWidget(button('导入文件夹',self.choose_directory))
        top.addWidget(button('按类别选择',self.choose_classes));top.addWidget(button('指定标签',self.label_selected));top.addWidget(button('移除所选',self.remove_selected));top.addStretch();layout.addLayout(top)
        self.data_summary=label('尚未导入数据','section');layout.addWidget(self.data_summary)
        splitter=W.QSplitter();layout.addWidget(splitter,1)
        left,ll=panel('CSV清单');self.data_table=table(['参与训练','采集记录','布料','单程','审查标记'])
        for col in [0,3,4]:self.data_table.horizontalHeader().setSectionResizeMode(col,W.QHeaderView.ResizeToContents)
        self.data_table.itemChanged.connect(self.update_selection);self.data_table.itemSelectionChanged.connect(self.preview_data);ll.addWidget(self.data_table);splitter.addWidget(left)
        right,rl=panel('去回程预览');row=W.QHBoxLayout();self.preview_signal=W.QComboBox()
        for key,text in SIGNALS.items():self.preview_signal.addItem(text,key)
        self.preview_mode=W.QComboBox();self.preview_mode.addItems(['原始磁响应与位置裁剪','模型输入（因果预处理后）'])
        row.addWidget(self.preview_signal);row.addWidget(self.preview_mode);rl.addLayout(row);self.data_plot=Plot();rl.addWidget(self.data_plot,1)
        self.preview_signal.currentIndexChanged.connect(self.preview_data);self.preview_mode.currentIndexChanged.connect(self.preview_data)
        self.data_details=label('选中文件查看处理规则。','muted');rl.addWidget(self.data_details);splitter.addWidget(right);splitter.setSizes([560,580])
        self.import_diagnostics=label('历史相对起点裁剪规则，详情见采集协议。' if self.config.data.contact_rule=='relative-035-v1' else f'固定X范围：去程{self.config.data.out_low:g}～{self.config.data.out_high:g}mm；回程{self.config.data.back_high:g}→{self.config.data.back_low:g}mm。TCN默认仅检查By。','notice');layout.addWidget(self.import_diagnostics)
        bottom=W.QHBoxLayout();bottom.addWidget(button('保存数据集版本',self.choose_save_dataset));bottom.addWidget(button('载入数据集版本',self.choose_load_dataset));bottom.addWidget(button('导出筛查报告',self.export_screening));bottom.addStretch();bottom.addWidget(button('下一步：训练 →',lambda:self.navigation.setCurrentRow(1),True));layout.addLayout(bottom);return page

    def choose_files(self):
        paths,_=W.QFileDialog.getOpenFileNames(self,'选择原始磁采样CSV','','CSV (*.csv)')
        if paths:self.import_paths(paths)
    def choose_directory(self):
        path=W.QFileDialog.getExistingDirectory(self,'选择布料采集目录')
        if path:self.import_paths([path])
    def import_paths(self,paths,selected_paths=None,batch_scopes=None):
        if self.import_jobs:self.notify('正在导入，请等待本次检查完成');return
        if self.busy():self.notify('训练或识别进行中，请完成后再更换数据');return
        if self.snapshot_path:
            self.records=[];self.errors=[];self.batches=[];self.import_reports=[];self.data_table.setRowCount(0);self.snapshot_path=None;self.snapshot_manifest=None
        ignored=[]
        if batch_scopes is None and self.batches:
            found,ignored,discovery_errors=discover_report(paths);incoming={str(p) for p in found}|{e['path'] for e in discovery_errors}
            touched=[b['candidates'] for b in self.batches if incoming.intersection(b['candidates'])]
            if touched:
                known={p for b in self.batches for p in b['candidates']};new=sorted(incoming-known)
                batch_scopes=touched+([new] if new else [])
            else:ignored=[]
        job=ImportThread(paths,self.config.data,dict(self.overrides),self,signals=self.selected_signals(),batch_scopes=batch_scopes,initial_ignored=ignored);self.import_jobs.append(job);self.update_selection()
        self.notify('正在检查原始CSV…')
        job.progress.connect(lambda a,b:self.notify(f'导入检查 {a}/{b}'))
        def done(records,report):
            self.screening=report;self.import_reports.append(report);self.batches.extend(report.get('review_batches',[]));self.add_records(records,report['errors'])
            if selected_paths is not None:
                chosen=set(selected_paths)
                self.data_table.blockSignals(True)
                for i,r in enumerate(self.records):self.data_table.item(i,0).setCheckState(C.Qt.Checked if r.path in chosen else C.Qt.Unchecked)
                self.data_table.blockSignals(False);self.update_selection()
        def finished():
            self.import_jobs.remove(job);job.deleteLater();self.update_selection()
            self.notify(f'审查完成 · 可训练{sum(r.audit.get("eligible",False) for r in self.records)} · {len(self.records)+len(self.errors)}条记录均保留显示')
            if self.selected_signals() and job.signals!=self.selected_signals():C.QTimer.singleShot(0,self.inputs_changed)
            if self.pending_close and not self.import_jobs:self.close()
        job.imported.connect(done);job.finished.connect(finished);job.start()

    def add_records(self,records,errors):
        old_checked={r.path for r in self.selected_records()};old_paths={r.path for r in self.records}
        attempted={r.path for r in records}|{e['path'] for e in errors};failed={e['path'] for e in errors}
        by_path={r.path:r for r in self.records if r.path not in failed}
        for record in records:
            if record.path in self.overrides:record.label=self.overrides[record.path]
            by_path[record.path]=record
        unique=[];hashes={};conflicts=set()
        for record in by_path.values():
            if record.sha256 in hashes:
                old=hashes[record.sha256]
                if record.label!=old.label or record.metadata_sha256!=old.metadata_sha256:
                    conflicts.add(record.sha256);errors.extend(dict(path=r.path,message='重复内容的标签或元数据冲突，请明确正确来源') for r in (old,record))
                else:self.screening.setdefault('duplicates',[]).append(dict(path=record.path,canonical=old.path))
                continue
            hashes[record.sha256]=record;unique.append(record)
        unique=[r for r in unique if r.sha256 not in conflicts]
        self.records=unique;success={r.path for r in self.records}
        combined=[e for e in self.errors if e['path'] not in attempted]+errors;self.errors=[];known=set()
        for error in combined:
            if error['path'] not in success and error['path'] not in known:self.errors.append(error);known.add(error['path'])
        if not self.snapshot_path:
            fresh=[r for r in self.records if not r.audit.get('batch_id')];bad=[e for e in self.errors if not e.get('audit',{}).get('batch_id')]
            if fresh or bad:
                axes=set().union(*(set(r.axes) for r in fresh)) if fresh else set(required_axes(self.selected_signals() or ('by',)))
                signals=('bxby',) if axes=={'x','y'} else ('by',) if 'y' in axes else ('bx',)
                self.batches.append(review_batch(fresh,bad,[r.path for r in fresh]+[e['path'] for e in bad],self.config.data,signals,overrides=self.overrides))
            active={r.audit.get('batch_id') for r in self.records}|{e.get('audit',{}).get('batch_id') for e in self.errors}
            self.batches=[b for b in self.batches if b['batch_id'] in active]
        self.data_table.blockSignals(True);self.data_table.setRowCount(len(self.records)+len(self.errors))
        for i,r in enumerate(self.records):
            row_values(self.data_table,i,['',Path(r.path).stem,r.label or '待指定',2,QUALITY[r.quality]])
            self.data_table.item(i,4).setToolTip('；'.join(r.quality_notes+[str(r.diagnostics)]))
            eligible=self.record_eligible(r);check=W.QTableWidgetItem('是' if eligible else '否');check.setFlags(C.Qt.ItemIsEnabled|C.Qt.ItemIsSelectable|(C.Qt.ItemIsUserCheckable if eligible else C.Qt.NoItemFlags))
            check.setCheckState(C.Qt.Checked if eligible and (r.path not in old_paths or r.path in old_checked) else C.Qt.Unchecked);self.data_table.setItem(i,0,check)
            self.data_table.item(i,1).setToolTip(r.path);self.data_table.item(i,4).setForeground(G.QColor('#66d9c4' if eligible else '#f08eab'))
        for i,e in enumerate(self.errors,len(self.records)):
            row_values(self.data_table,i,['否',Path(e['path']).stem,e.get('label') or '—','—',QUALITY.get(e.get('audit',{}).get('status'),'不可用·数据错误')]);self.data_table.item(i,4).setToolTip(e['message']);self.data_table.item(i,4).setForeground(G.QColor('#f08eab'))
        self.data_table.blockSignals(False)
        if self.records:self.data_table.selectRow(0)
        classes=[f'{name}：大幅异常{c["abnormal_groups"]}/{c["assessable_groups"]}组，无法判断{c["undetermined_groups"]}组'+(' · 本批次该类暂不启用' if c['disabled'] else ' · 不足10组，仅提示复核' if c['review_small_batch'] else '') for b in self.batches for name,c in b['classes'].items()]
        self.import_diagnostics.setText(('；'.join(c for c in classes if '暂不启用' in c or '仅提示复核' in c) or '审查批次已固定，取消勾选不改变类别异常比例。')+'\n'+('\n'.join(f'{Path(e["path"]).name}：{e["message"]}' for e in self.errors[:2]) if self.errors else '完整审查指标见记录详情和导出报告。'))
        self.import_diagnostics.setToolTip('\n'.join(classes)+'\n每次增量导入建立新批次；类别统计不随勾选或移除显示行改变。')
        self.update_selection()

    def record_eligible(self,r):
        return bool(r.audit.get('eligible')) and (not self.snapshot_path or self.snapshot_manifest.get('admission',{}).get('rule')==current_rule())
    def selected_records(self):
        return [r for i,r in enumerate(self.records) if self.record_eligible(r) and self.data_table.item(i,0) and self.data_table.item(i,0).checkState()==C.Qt.Checked]
    def selected_rows(self):return sorted({ix.row() for ix in self.data_table.selectedIndexes()})
    def assign_labels(self,rows,text):
        text=text.strip()
        if not text:raise ValueError('标签不能为空')
        if self.snapshot_path:raise ValueError('快照标签已冻结，修改标签需重新导入原CSV审查')
        for i in rows:
            if i<len(self.records):self.records[i].label=text;self.records[i].label_confirmed=True;self.overrides[self.records[i].path]=text;self.data_table.item(i,2).setText(text)
        self.update_selection()
        self.reimport_review_batches()
    def label_selected(self):
        rows=self.selected_rows()
        if not rows:self.notify('请先选择需要指定标签的CSV');return
        value,ok=W.QInputDialog.getText(self,'批量指定布料','布料名称')
        if ok:
            try:self.assign_labels(rows,value)
            except ValueError as exc:self.error(exc)
    def choose_classes(self):
        if not self.records:return
        d=W.QDialog(self);d.setWindowTitle('选择参与训练的布料');layout=W.QVBoxLayout(d);checks={};counts=Counter(r.label for r in self.records);selected={r.label for r in self.selected_records()}
        for c,n in sorted(counts.items()):
            check=W.QCheckBox(f'{c or "待指定标签"} · {n} CSV'+(' · 五折覆盖需检查' if n<5 else ''));check.setChecked(c in selected);layout.addWidget(check);checks[c]=check
        controls=W.QDialogButtonBox(W.QDialogButtonBox.Ok|W.QDialogButtonBox.Cancel);controls.accepted.connect(d.accept);controls.rejected.connect(d.reject);layout.addWidget(controls)
        if d.exec_()==W.QDialog.Accepted:
            for i,r in enumerate(self.records):self.data_table.item(i,0).setCheckState(C.Qt.Checked if checks[r.label].isChecked() else C.Qt.Unchecked)
    def remove_selected(self):
        if self.busy() or self.import_jobs:self.notify('请等待当前任务完成');return
        rows=set(self.selected_rows());self.exclusions.extend(dict(path=e['path'],reason='用户明确移除待处理记录：'+e['message']) for i,e in enumerate(self.errors,len(self.records)) if i in rows);keep=[r for i,r in enumerate(self.records) if i not in rows]
        self.errors=[e for i,e in enumerate(self.errors,len(self.records)) if i not in rows];self.records=[];self.data_table.setRowCount(0);self.add_records(keep,[])
    def preview_data(self,*args):
        i=self.data_table.currentRow()
        if i<0:return
        if i>=len(self.records):
            if i-len(self.records)<len(self.errors):
                e=self.errors[i-len(self.records)];self.data_plot.empty('本记录未保存可训练片段；审查结论保留')
                self.data_details.setText(e['message']+'\n'+audit_text(e.get('diagnostics',{}).get('direction_quality',{})))
                self.data_details.setToolTip(json.dumps(e,ensure_ascii=False,indent=2))
            return
        r=self.records[i]
        try:self.data_plot.curve(r,self.preview_signal.currentData(),self.preview_mode.currentIndex()==1)
        except (ValueError,IndexError):self.data_plot.empty('此版本未载入所选通道；请重新导入原始CSV检查')
        details=audit_text(r.diagnostics.get('direction_quality',{}))
        contact=r.diagnostics.get('contact_rule',{})
        if contact.get('mode')=='start-relative':details+=f'\n相对起点裁剪 · 起点X={contact["origin_x_mm"]:g}mm · 绝对X窗{r.trials[0].bounds[0]:g}～{r.trials[0].bounds[1]:g}mm · 去程44～144mm / 回程移动11～111mm'
        self.data_details.setText(f'{r.label or "无标签"} · {r.speed:g} mm/s · {QUALITY[r.quality]}\n'+details+'\n'+('；'.join(r.quality_notes) or '必要检查通过，未触发异常规则')+('\n'+'；'.join(r.warnings) if r.warnings else ''))
        self.data_details.setToolTip(json.dumps(r.manifest(),ensure_ascii=False,indent=2))

    def selected_signals(self):
        return tuple(k for k,c in self.signal_checks.items() if c.isChecked()) if hasattr(self,'signal_checks') else self.config.signals
    def reimport_review_batches(self):
        scopes=[b['candidates'] for b in self.batches]
        if not scopes:scopes=[[r.path for r in self.records]+[e['path'] for e in self.errors]]
        chosen=[r.path for r in self.selected_records()]
        if any(scopes):self.import_paths([p for scope in scopes for p in scope],chosen,batch_scopes=scopes)
    def inputs_changed(self,*args):
        self.update_selection()
        if self.snapshot_path or self.busy() or self.import_jobs or not (self.records or self.errors):return
        if not self.selected_signals():return
        needed=set(required_axes(self.selected_signals()))
        if any(needed!=set(r.axes) for r in self.records) or self.errors:
            C.QTimer.singleShot(0,self.reimport_review_batches)
    def update_selection(self,*args):
        selected=self.selected_records();counts=Counter(r.label for r in selected)
        quality=Counter(r.quality for r in selected)
        all_quality=Counter(r.quality for r in self.records)+Counter(e.get('audit',{}).get('status','data_error') for e in self.errors)
        scanned=len({p for b in self.batches for p in b['candidates']}) or len(self.records)+len(self.errors)
        self.data_summary.setText(f'扫描{scanned} CSV · 可训练{sum(self.record_eligible(r) for r in self.records)} · 建议复核{all_quality["suggest_review"]} · 排除{all_quality["excluded"]} · 无法判断{all_quality["undetermined"]} · 数据错误{all_quality["data_error"]}\n已选 {len(counts)} 类 · {len(selected)} CSV · {len(selected)*2} 单程')
        if not hasattr(self,'start_button'):return
        valid=True
        try:
            if self.snapshot_path and not self.snapshot_manifest.get('admission'):raise ValueError('旧快照仅供查看；新训练需从原CSV重新审查并保存新版本')
            require_admission(selected,package(self.batches),self.config.data)
            for signal in self.selected_signals():
                for r in selected:
                    if not set(required_axes((signal,))).issubset(r.axes):raise ValueError('所选通道未包含在当前快照/导入中，请重新检查原始CSV')
            if self.snapshot_path and self.selected_signals() and any(set(r.axes)!=set(required_axes(self.selected_signals())) for r in selected):raise ValueError('快照通道/有效点协议已冻结，更改输入范围请重新导入原始CSV建立新版本')
            protocol_for(selected,self.config.data)
            plan=self.selected_plan(selected,self.config.split_seed)
            split=plan['cv'][0];text=f'固定测试 {len(plan["test"])} CSV · 开发集 {len(plan["dev"])} CSV · 开发集第1折：训练 {len(split["train"])} / 验证 {len(split["val"])} CSV'
        except ValueError as exc:valid=False;text=str(exc)
        self.training_summary.setText(self.data_summary.text());self.partition_summary.setText(text)
        if valid:
            details=[]
            for j,part in enumerate(plan['cv']):
                details.append(f'第{j+1}折：'+str({k:len(v) for k,v in part.items()}))
            self.partition_summary.setToolTip('\n'.join(details)+'\n同CSV及会话不拆组；混合会话可能使比例略有不同。')
        self.start_button.setEnabled(valid and bool(self.selected_signals()) and any(c.isChecked() for c in self.arch_checks.values()) and not self.busy() and not self.import_jobs)
        self.update_task_count();self.cancel_button.setEnabled(self.busy() and not self.cancelled)

    def build_training(self):
        page=W.QWidget();layout=W.QVBoxLayout(page);layout.setContentsMargins(0,0,0,0)
        top,tl=panel('第一阶段 · 完整接触段分类');self.training_summary=label('未选择数据','section');tl.addWidget(self.training_summary)
        self.task_name=W.QLineEdit();self.task_name.setPlaceholderText('实验名称（可选）');tl.addWidget(self.task_name)
        row=W.QHBoxLayout();row.addWidget(label('训练方式'));self.preset_selector=W.QComboBox()
        for key,title in PRESETS.items():self.preset_selector.addItem(title,key)
        self.preset_selector.currentIndexChanged.connect(self.apply_preset);row.addWidget(self.preset_selector);row.addStretch();self.task_count=label('6项模型训练','muted');row.addWidget(self.task_count);tl.addLayout(row);layout.addWidget(top)
        tl.addWidget(label('GPU训练 · 自动检查NVIDIA环境与显存','muted'))
        top.setSizePolicy(W.QSizePolicy.Preferred,W.QSizePolicy.Minimum)
        tl.setSizeConstraint(W.QLayout.SetMinimumSize)
        model,ml=panel('模型与磁响应输入');row=W.QHBoxLayout();self.arch_checks={}
        for key,title in ARCHITECTURES.items():
            c=W.QCheckBox(title);c.setChecked(key in self.config.architectures);c.toggled.connect(self.update_selection);row.addWidget(c);self.arch_checks[key]=c
        row.addStretch();ml.addLayout(row);row=W.QHBoxLayout();self.signal_checks={}
        self.research_inputs=W.QWidget();research_layout=W.QHBoxLayout(self.research_inputs);research_layout.setContentsMargins(0,0,0,0)
        for key,title in SIGNALS.items():
            c=W.QCheckBox(('主实验：' if key=='by' else '对照：')+input_label(key));c.setChecked(key in self.config.signals);c.toggled.connect(self.inputs_changed);(row if key in ('by','bxby') else research_layout).addWidget(c);self.signal_checks[key]=c
        ml.addLayout(row);self.research_toggle=W.QCheckBox('显示高级研究对照');ml.addWidget(self.research_toggle);ml.addWidget(self.research_inputs);self.research_inputs.hide();self.research_toggle.toggled.connect(self.research_inputs.setVisible);self.last_readout=W.QCheckBox('同时训练“最后时刻读出”研究对照（任务数翻倍）');self.last_readout.toggled.connect(self.update_task_count);research_layout.addWidget(self.last_readout)
        ml.addWidget(label('默认把整段的有效特征取平均后分类。每个单程只计算一次分类损失，不训练风险头。','muted'));layout.addWidget(model)
        fixed,fl=panel('固定20%测试 · 其余80%开发集做五折比较');self.partition_summary=label('检查五折是否覆盖所有类别','muted');fl.addWidget(self.partition_summary)
        fl.addWidget(label('先比较开发集验证表现，再选定配置训练最终模型，最后运行固定测试。第二阶段后续开放。','muted'))
        row=W.QHBoxLayout();row.addWidget(button('训练参数…',self.advanced));row.addWidget(button('采集协议…',lambda:self.advanced(True)));row.addStretch();fl.addLayout(row);layout.addWidget(fixed)
        actions=W.QHBoxLayout();self.start_button=button('开始第一阶段训练',self.start_training,True);self.cancel_button=button('取消任务',self.cancel_worker);self.cancel_button.setEnabled(False)
        actions.addWidget(self.start_button);actions.addWidget(self.cancel_button);actions.addStretch();self.training_state=label('等待训练','muted');actions.addWidget(self.training_state);layout.addLayout(actions)
        self.progress=W.QProgressBar();self.progress.setValue(0);layout.addWidget(self.progress);self.live_plot=Plot();layout.addWidget(self.live_plot,1)
        self.log_toggle=W.QCheckBox('显示详细日志');layout.addWidget(self.log_toggle);self.training_log=W.QPlainTextEdit();self.training_log.setReadOnly(True);self.training_log.setMaximumBlockCount(700);self.training_log.setMaximumHeight(140);self.training_log.hide();self.log_toggle.toggled.connect(self.training_log.setVisible);layout.addWidget(self.training_log)
        return page

    def update_task_count(self,*args):
        if not hasattr(self,'last_readout'):return
        n=sum(c.isChecked() for c in self.signal_checks.values())*sum(c.isChecked() for c in self.arch_checks.values())*(2 if self.last_readout.isChecked() else 1)*self.config.repetitions*5
        self.task_count.setText(f"{n}项模型训练 · 每折{'最多' if self.config.early_stopping else '预定'}{self.config.epochs}轮"+(' · 试跑，不作为正式成果' if self.config.preset=='quick' else ''))
    def apply_preset(self,*args):
        name=self.preset_selector.currentData()
        if name and name!='custom':self.config=self.config.with_preset(name)
        self.update_task_count()
    def sync_controls(self):
        for k,c in self.signal_checks.items():c.setChecked(k in self.config.signals)
        for k,c in self.arch_checks.items():c.setChecked(k in self.config.architectures)
        self.last_readout.setChecked('last' in self.config.readouts)
        self.preset_selector.blockSignals(True);self.preset_selector.setCurrentIndex(self.preset_selector.findData(self.config.preset));self.preset_selector.blockSignals(False);self.update_task_count()
    def advanced(self,protocol=False):
        if self.busy() or self.import_jobs:self.notify('当前任务完成后才能更改设置');return
        d=SettingsDialog(self.controls_config(),self,protocol=protocol)
        if d.exec_()!=W.QDialog.Accepted:return
        changed=d.result_config.data!=self.config.data
        if changed and self.snapshot_path:self.notify('载入的快照协议已冻结；修改协议请重新导入原始CSV建立新版本');return
        self.config=d.result_config;self.sync_controls()
        if changed and (self.records or self.errors):self.reimport_review_batches()
        self.update_selection();self.notify('设置已保存；用于下一次训练')
    def controls_config(self):
        return replace(self.config,device="cuda",signals=tuple(k for k,v in self.signal_checks.items() if v.isChecked()),
                       architectures=tuple(k for k,v in self.arch_checks.items() if v.isChecked()),
                       readouts=('mean','last') if self.last_readout.isChecked() else ('mean',),name=self.task_name.text().strip())
    def make_training_request(self):
        records=self.selected_records();config=self.controls_config().validate();self.selected_plan(records,config.split_seed)
        require_admission(records,package(self.batches),config.data)
        if self.snapshot_path:
            if [[r.path,r.sha256,r.metadata_sha256,r.label,r.group,r.label_confirmed] for r in records]!=[[v['path'],v['sha256'],v.get('metadata_sha256'),v['label'],v['group'],v.get('label_confirmed',True)] for v in self.snapshot_manifest['sources']]:raise ValueError('快照选择或标签已更改，请先保存新数据集版本')
            return dict(snapshot=str(self.snapshot_path),config=config.to_dict(),output=str(self.store))
        return dict(paths=[r.path for r in records],sources=[r.manifest() for r in records],review_batches=self.batches,protocol=protocol_for(records,config.data),overrides={r.path:r.label for r in records},config=config.to_dict(),output=str(self.store),screening=dict(self.screening_value(),excluded=self.exclusions+[dict(path=r.path,reason='用户未勾选或审查不准入') for r in self.records if r.path not in {x.path for x in records}]))
    def start_training(self):
        if self.busy() or self.import_jobs:return
        try:req=self.make_training_request()
        except ValueError as exc:self.error(exc);return
        self.live_history=[];self.live_plot.empty('准备完整段分类训练…');self.training_log.clear();self.navigation.setCurrentRow(1);self.launch(req,'train')
    def start_imported_training(self,exclude_labels=()):
        if self.busy() or self.import_jobs:return False
        if exclude_labels:
            for i,r in enumerate(self.records):
                if r.label in exclude_labels:self.data_table.item(i,0).setCheckState(C.Qt.Unchecked)
        self.update_selection()
        if not self.start_button.isEnabled():self.notify('未启动训练：'+self.partition_summary.text());return False
        self.start_training();return self.busy()
    def launch(self,request,kind):
        self.worker_kind=kind;self.cancelled=False;self.active_task=None;self.buffer='';self.stderr=''
        requests=self.store.parent/'requests';requests.mkdir(parents=True,exist_ok=True);self.active_request=requests/(uuid.uuid4().hex+'.json');write_json(self.active_request,request)
        self.process.setProgram(sys.executable);self.process.setArguments([str(ROOT/'train_cloth.py'),'--worker',str(self.active_request),'--kind',kind]);self.process.setWorkingDirectory(str(ROOT));self.process.start()
        self.progress.setRange(0,0);self.training_state.setText('准备中…');self.update_selection();self.predict_button.setEnabled(False);self.notify('后台任务已启动')
    def read_output(self):
        self.buffer+=bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace')
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1)
            try:self.on_event(json.loads(line))
            except ValueError:
                if line:self.training_log.appendPlainText(line)
    def read_error(self):
        self.stderr+=bytes(self.process.readAllStandardError()).decode('utf-8',errors='replace');self.stderr=self.stderr[-12000:]
    def on_event(self,event):
        message=event.get('message','');self.training_log.appendPlainText(message);self.notify(message);self.training_state.setText(message)
        if event.get('task'):self.active_task=Path(event['task'])
        if event.get('type')=='phase':self.live_history=[]
        if event.get('type') in ('epoch','restored'):
            if event['type']=='restored':self.live_history=list(event['history'])
            else:self.live_history.append(event)
            self.live_plot.training(self.live_history)
            jobs=event.get('jobs',1);job=event.get('job',1);self.progress.setRange(0,100);self.progress.setValue(min(95,int(((job-1)+event['epoch']/event.get('epochs',self.config.epochs))/jobs*100)))
        if event.get('type')=='prediction_done':self.prediction_path=Path(event['destination'])
    def worker_error(self,error):
        if error==C.QProcess.FailedToStart:
            self.progress.setRange(0,100);self.progress.setValue(0);self.notify('后台进程无法启动：'+self.process.errorString());self.update_selection()
    def worker_finished(self,code,status):
        self.read_output();self.read_error();ok=code==0 and not self.cancelled
        self.progress.setRange(0,100);self.progress.setValue(100 if ok else 0)
        if self.active_task and (self.active_task/'metadata.json').exists():
            meta=self.task_metadata(self.active_task)
            if meta.get('status')=='running':
                meta.update(status='cancelled' if self.cancelled else 'failed',error='用户取消' if self.cancelled else self.stderr or '后台进程异常退出');write_json(self.active_task/'metadata.json',meta)
            ok=ok and meta.get('status')=='completed'
        if ok and self.worker_kind in ('train','test','final') and self.active_task:
            self.show_results(self.active_task);self.navigation.setCurrentRow(2)
        elif ok and self.worker_kind=='predict' and self.prediction_path:self.show_prediction(self.prediction_path)
        elif not ok:self.training_log.appendPlainText(self.stderr or '任务已取消／失败，请检查日志');self.log_toggle.setChecked(True)
        self.training_state.setText('已完成并保存' if ok else ('已取消' if self.cancelled else '失败，查看日志'))
        self.notify(self.training_state.text());self.refresh_history();self.update_selection();self.predict_button.setEnabled(bool(self.recognition_paths) and self.model_selector.count()>0)
    def cancel_worker(self):
        if not self.busy():return
        self.cancelled=True;self.active_request.with_suffix('.cancel').touch();self.cancel_button.setEnabled(False);self.notify('正在取消，保留日志…')
        C.QTimer.singleShot(8000,lambda:self.process.kill() if self.busy() and self.cancelled else None)

    def build_history(self):
        page=W.QWidget();layout=W.QVBoxLayout(page);layout.setContentsMargins(0,0,0,0)
        splitter=W.QSplitter();layout.addWidget(splitter,1)
        left,ll=panel('实验');self.experiment_search=W.QLineEdit();self.experiment_search.setPlaceholderText('查找实验…');self.experiment_search.textChanged.connect(self.filter_experiments);ll.addWidget(self.experiment_search);self.library_tabs=W.QTabWidget();self.library_tabs.setUsesScrollButtons(False);self.library_tabs.setStyleSheet('QTabBar::tab {padding:10px 7px;font-size:12px;}');ll.addWidget(self.library_tabs)
        self.history=W.QListWidget();self.legacy_history=W.QListWidget();self.library_tabs.addTab(self.history,'分类实验');self.library_tabs.addTab(self.legacy_history,'旧版记录');self.task_history=W.QListWidget();self.library_tabs.addTab(self.task_history,'任务记录')
        for w in (self.history,self.legacy_history,self.task_history):w.currentItemChanged.connect(self.experiment_selected)
        splitter.addWidget(left);right=W.QWidget();rl=W.QVBoxLayout(right);rl.setContentsMargins(6,0,0,0);splitter.addWidget(right);splitter.setSizes([245,880])
        self.experiment_title=label('选择实验','section');rl.addWidget(self.experiment_title)
        self.final_summary=label('最终模型尚未训练','notice');rl.addWidget(self.final_summary)
        scope=W.QHBoxLayout();scope.addWidget(label('查看'));self.artifact_scope=W.QComboBox();self.artifact_scope.addItems(['开发集交叉验证','最终模型与固定测试']);self.artifact_scope.currentIndexChanged.connect(self.draw_result);scope.addWidget(self.artifact_scope);rl.addLayout(scope)
        self.comparison=table(['模型','输入','验证F1','旧版折外准确率','第二阶段']);self.comparison.setMinimumHeight(195);self.comparison.setMaximumHeight(235);self.comparison.itemSelectionChanged.connect(self.result_selected);rl.addWidget(self.comparison)
        self.detail_tabs=W.QTabWidget();rl.addWidget(self.detail_tabs,1)
        classification=W.QWidget();dl=W.QVBoxLayout(classification);self.detail_tabs.addTab(classification,'分类')
        row=W.QHBoxLayout();row.addWidget(label('具体模型'));self.result_selector=W.QComboBox();self.result_selector.currentIndexChanged.connect(self.draw_result);row.addWidget(self.result_selector,1)
        self.result_partition=W.QComboBox();self.result_partition.addItems(['验证结果','测试结果']);self.result_partition.currentIndexChanged.connect(self.draw_result);row.addWidget(self.result_partition);self.result_aggregate=W.QComboBox();self.result_aggregate.addItems(['当前折具体权重','同种子合并折外']);self.result_aggregate.currentIndexChanged.connect(self.draw_result);row.addWidget(self.result_aggregate);dl.addLayout(row)
        self.result_notice=label('训练完成后先查看验证结果。','notice');dl.addWidget(self.result_notice)
        self.result_tabs=W.QTabWidget();dl.addWidget(self.result_tabs,1);self.result_curve=Plot();self.matrix_plot=Plot();self.class_plot=Plot();self.prefix_plot=Plot()
        for w,title in [(self.result_curve,'训练曲线'),(self.matrix_plot,'混淆矩阵'),(self.class_plot,'类别表现')]:self.result_tabs.addTab(w,title)
        detail=W.QWidget();el=W.QVBoxLayout(detail);self.errors_only=W.QCheckBox('只显示错分');self.errors_only.toggled.connect(self.fill_test_rows);el.addWidget(self.errors_only)
        self.test_table=table(['文件／方向','真实布料','预测布料']);self.test_table.setMaximumHeight(180);self.test_table.itemSelectionChanged.connect(self.preview_test);el.addWidget(self.test_table);self.test_plot=Plot();el.addWidget(self.test_plot,1);self.result_tabs.addTab(detail,'逐条预测')
        self.result_tabs.addTab(self.prefix_plot,'旧版前缀')
        risk=W.QWidget();kl=W.QVBoxLayout(risk);kl.addWidget(label('第二阶段 · 风险与提前停止（后续开放）','section'))
        kl.addWidget(label('本次只实现第一阶段分类。后续将从一个明确的分类模型开始，冻结主干，一次执行风险头训练与独立校准。当前模型只支持完整扫描识别。','notice'))
        self.stage2_button=button('第二阶段后续开放',lambda:None);self.stage2_button.setEnabled(False);kl.addWidget(self.stage2_button);self.calibration_text=W.QPlainTextEdit();self.calibration_text.setReadOnly(True);kl.addWidget(self.calibration_text,1);self.detail_tabs.addTab(risk,'风险与提前停止')
        self.record_text=W.QPlainTextEdit();self.record_text.setReadOnly(True);self.detail_tabs.addTab(self.record_text,'训练记录');self.oof_text=W.QPlainTextEdit();self.oof_text.setReadOnly(True);self.detail_tabs.addTab(self.oof_text,'折外汇总')
        row=W.QHBoxLayout();self.final_button=button('训练最终模型',self.confirm_final);self.final_button.setEnabled(False);row.addWidget(self.final_button);self.test_button=button('固定留出测试',self.confirm_test);row.addWidget(self.test_button);row.addWidget(button('加载识别',self.history_load,True));row.addWidget(button('导出图表',self.export_figures));rl.addLayout(row)
        actions=W.QHBoxLayout()
        actions.addWidget(button('继续未完成任务',self.resume_selected))
        for text,fn in [('导出成果包',self.history_export),('导入成果包',self.history_import),('重命名',self.history_rename),('打开文件夹',self.history_open)]:actions.addWidget(button(text,fn))
        layout.addLayout(actions);layout.addWidget(label('每行是一种模型配置；各折及复验收在详情中。未测试不代表0%。快速检查和失败记录不能用于正式识别。','muted'));return page
    def filter_experiments(self,text):
        for w in (self.history,self.legacy_history,self.task_history):
            for i in range(w.count()):w.item(i).setHidden(text.casefold() not in w.item(i).text().casefold())
    def experiment_selected(self,item,*args):
        if item:self.show_results(item.data(C.Qt.UserRole))
    def show_results(self,task):
        self.result_directory=Path(task);self.result_task=self.task_metadata(self.result_directory);self.catalog_rows=configurations(self.result_task)
        legacy=self.result_task.get('format')=='cera-net-v1';self.experiment_title.setText(self.result_task['name']+(' · 旧版只读' if self.result_task.get('format') not in ('cloth-classification-v3','cloth-classification-v4') else '')+(' · 修复验证（非正式）' if self.result_task.get('validation_kind') else ' · 快速检查' if self.result_task.get('purpose')=='quick' else ''))
        self.comparison.blockSignals(True);self.comparison.setRowCount(len(self.catalog_rows))
        for i,r in enumerate(self.catalog_rows):
            name=(VARIANTS.get(self.result_task['models'][r['keys'][0]].get('variant'),'旧版CERA') if legacy else r['architecture'].upper())+(' · 最后时刻' if r['readout']=='last' else '')+(f' · {len({v.get("fold") for v in r["results"]})}折' if any('fold' in v for v in r['results']) else f' ×{len(r["keys"])}' if len(r['keys'])>1 else '')
            val='—' if r['val_mean'] is None else f'{r["val_mean"]:.3f}'+(f' ±{r["val_std"]:.3f}' if len(r['keys'])>1 else '')
            test='未测试' if r['test_mean'] is None else percent(r['test_mean'])+(f' ±{r["test_std"]:.1%}' if len({v.get('seed') for v in r['results']})>1 else '')
            row_values(self.comparison,i,[name,input_label(r['signal'],self.result_task['format']),val,test,'旧版联合训练' if legacy else '后续开放'])
        self.comparison.blockSignals(False);self.result_model=None;self.result_selector.clear();self.prefix_plot.setVisible(legacy);self.result_tabs.setTabVisible(4,legacy)
        modern=self.result_task.get('format')=='cloth-classification-v4';final=self.result_task.get('final_model')
        self.artifact_scope.setEnabled(modern);self.final_button.setEnabled(modern and self.result_task.get('status')=='completed' and not final and not self.busy())
        self.final_summary.setText(('最终模型：'+final['configuration_id']+f' · {final["epochs"]}轮 · '+('固定测试已完成' if self.result_task.get('holdout') else '固定测试未运行')) if modern and final else '开发集比较完成后，选定配置训练最终模型' if modern else '旧版实验保留原流程')
        self.test_button.setEnabled(self.result_task.get('format') in ('cloth-classification-v3','cloth-classification-v4') and (bool(final) if modern else True) and self.result_task.get('status')=='completed' and self.result_task.get('purpose')!='quick' and not self.busy())
        self.record_text.setPlainText(json.dumps({k:v for k,v in self.result_task.items() if k not in ('models','sources','holdout')},ensure_ascii=False,indent=2))
        if self.catalog_rows:
            self.comparison.blockSignals(True);self.comparison.selectRow(0);self.comparison.blockSignals(False);self.result_selected()
        else:self.result_notice.setText('尚无完整模型，请查看训练记录。')
        if self.result_task.get('status') in ('failed','cancelled'):
            self.result_notice.setText(self.result_task.get('error','任务未完成')+'；使用“继续未完成任务”恢复，有恢复点时从最近完整轮继续，无恢复点时该折从第1轮重跑。')
    def result_selected(self):
        i=self.comparison.currentRow()
        if not self.result_task or not 0<=i<len(self.catalog_rows):return
        self.result_selector.blockSignals(True);self.result_selector.clear()
        for key in self.catalog_rows[i]['keys']:
            r=self.result_task['models'][key];self.result_selector.addItem((f'第{r["fold"]+1}折 · ' if 'fold' in r else '')+f'随机编号 {r["seed"]} · {r["model_id"][:8]}' if 'model_id' in r else key,key)
        self.result_selector.blockSignals(False);self.draw_result()
    def draw_result(self,*args):
        if not self.result_task:return
        if self.result_task.get('format')=='cloth-classification-v4' and self.artifact_scope.currentIndex()==1:
            self.draw_final_result();return
        if self.result_selector.currentData() is None:return
        self.result_selector.setEnabled(True);self.result_partition.setEnabled(True)
        r=self.result_task['models'][self.result_selector.currentData()];self.result_model=r;legacy=self.result_task.get('format')=='cera-net-v1'
        self.record_text.setPlainText(json.dumps(dict(实验={k:v for k,v in self.result_task.items() if k not in ('models','sources','holdout')},当前模型={k:v for k,v in r.items() if k not in ('val_predictions','test_predictions','history','runtime')}),ensure_ascii=False,indent=2))
        self.oof_text.setPlainText(json.dumps(self.result_task.get('summaries',{}).get(r.get('configuration_id'),{}),ensure_ascii=False,indent=2) or '未测试')
        self.result_curve.training(r['history']);is_test=self.result_partition.currentIndex()==1 or legacy
        seed_summary=self.result_task.get('summaries',{}).get(r.get('configuration_id'),{}).get('by_seed',{}).get(str(r.get('seed')))
        self.result_aggregate.setEnabled(bool(seed_summary) and is_test and not legacy)
        aggregate=self.result_aggregate.isEnabled() and self.result_aggregate.currentIndex()==1
        metrics=r.get('metrics') if legacy else r.get('test_metrics') if is_test else r.get('val_metrics')
        if aggregate:metrics=seed_summary['oof_metrics']
        if metrics:
            self.matrix_plot.confusion(metrics['confusion'],self.result_task['labels']);self.class_plot.classes(metrics)
            self.result_notice.setText(('旧版测试' if legacy else '留出测试' if is_test else '验证结果')+f'：{metrics["correct"]}/{metrics["total"]} 单程正确 · Macro-F1 {metrics["macro_f1"]:.3f} · 最佳第{r["best_epoch"]}轮')
        else:self.matrix_plot.empty('尚未运行留出测试');self.class_plot.empty('尚未测试');self.result_notice.setText('未测试：请先依据验证结果选定配置，再手动测试。')
        if aggregate:
            stats=seed_summary['fold_statistics'];self.result_notice.setText(f'合并折外：{metrics["correct"]}/{metrics["total"]}单程正确 · Macro-F1 {metrics["macro_f1"]:.3f} · 逐折准确率均值±标准差 {percent(stats["accuracy_mean"])} ±{percent(stats["accuracy_std"])}')
        if legacy:self.prefix_plot.prefix(r);self.calibration_text.setPlainText(json.dumps({k:r.get(k) for k in ('calibration','stopping','group_risk','brier','reliability')},ensure_ascii=False,indent=2))
        else:self.calibration_text.setPlainText('后续开放。该分类模型没有风险头和停止阈值。')
        if not legacy:
            direction=seed_summary['direction_metrics'] if aggregate else r.get('direction_metrics' if is_test else 'val_direction_metrics',{})
            if direction:self.result_notice.setText(self.result_notice.text()+' · 去程 '+percent(direction['out']['accuracy'])+' / 回程 '+percent(direction['back']['accuracy']))
        self.fill_test_rows()
    def draw_final_result(self):
        m=self.result_task;r=m.get('final_model');self.result_model=None;self.result_rows=[];self.test_table.setRowCount(0)
        self.result_selector.setEnabled(False);self.result_partition.setEnabled(False);self.result_aggregate.setEnabled(False)
        if not r:
            self.result_curve.empty('尚未训练最终模型');self.matrix_plot.empty('尚未运行固定测试');self.class_plot.empty('尚未运行固定测试');self.result_notice.setText('先选择开发集配置并点击训练最终模型');return
        self.result_model=r;self.result_curve.training(r['history']);h=m.get('holdout')
        self.result_notice.setText(f'最终模型 · {r["configuration_id"]} · 全开发集{r["train_records"]} CSV · 固定{r["epochs"]}轮，无测试集选轮')
        if h:
            metrics=h['metrics'];self.matrix_plot.confusion(metrics['confusion'],m['labels']);self.class_plot.classes(metrics)
            self.result_notice.setText(self.result_notice.text()+f' · 固定测试 {metrics["correct"]}/{metrics["total"]} 单程 · F1 {metrics["macro_f1"]:.3f} · 往返准确率 {percent(h["roundtrip_metrics"]["accuracy"])}');self.fill_test_rows()
        else:self.matrix_plot.empty('固定测试未运行');self.class_plot.empty('固定测试未运行')

    def fill_test_rows(self,*args):
        if not self.result_model:return
        r=self.result_model;legacy=self.result_task.get('format')=='cera-net-v1';source=r.get('test_predictions',[]) if self.result_partition.currentIndex()==1 or legacy else r.get('val_predictions',[])
        if self.result_task.get('format')=='cloth-classification-v4' and self.artifact_scope.currentIndex()==1:source=(self.result_task.get('holdout') or {}).get('predictions',[])
        if self.result_aggregate.isEnabled() and self.result_aggregate.currentIndex()==1:
            source=[x for candidate in self.result_task['models'].values() if candidate.get('configuration_id')==r.get('configuration_id') and candidate.get('seed')==r.get('seed') for x in candidate.get('test_predictions',[])]
        pred=lambda x:x['predicted'] if 'predicted' in x else x['checkpoints'][-1]['predicted']
        self.result_rows=[x for x in source if not self.errors_only.isChecked() or x['true']!=pred(x)];self.test_table.setRowCount(len(self.result_rows))
        for i,x in enumerate(self.result_rows):row_values(self.test_table,i,[Path(x['path']).stem+' / '+('去' if x['direction']=='out' else '回'),x['label'],self.result_task['labels'][pred(x)]])
        if self.result_rows:self.test_table.selectRow(0)
        else:self.test_plot.empty('没有符合条件的预测记录')
    def preview_test(self):
        i=self.test_table.currentRow()
        if 0<=i<len(self.result_rows):self.test_plot.trial(self.result_rows[i],self.result_task['labels'])
    def export_figures(self):
        if not self.result_model:return
        dest=W.QFileDialog.getExistingDirectory(self,'选择图表目录')
        if dest:
            for name,plot in [('training',self.result_curve),('confusion',self.matrix_plot),('classes',self.class_plot)]:plot.figure.savefig(Path(dest)/(name+'.png'),dpi=180)
            self.notify('图表已导出')
    def confirm_final(self):
        if not self.result_task:return
        row=self.comparison.currentRow()
        if not 0<=row<len(self.catalog_rows):return
        candidate=self.catalog_rows[row]['id']
        if W.QMessageBox.question(self,'锁定最终配置',f'使用 {candidate} 在全部开发集重训；轮数取各折最佳轮数中位数。配置锁定后不能更换，测试集不参与。继续？')==W.QMessageBox.Yes:self.run_final(candidate)
    def run_final(self,configuration_id=None):
        if self.busy() or not self.result_task:return
        self.launch(dict(task=str(self.result_directory),configuration_id=configuration_id),'final')

    def confirm_test(self):
        if not self.result_task:return
        if self.result_task.get('format')=='cloth-classification-v4':
            if W.QMessageBox.question(self,'固定内部测试','使用已锁定的最终模型评估预留数据。查看后不能据此更换候选，继续？')==W.QMessageBox.Yes:self.run_test()
            return
        rows=sorted({x.row() for x in self.comparison.selectedIndexes()});keys=[k for i in rows for k in self.catalog_rows[i]['keys']]
        if not keys:self.notify('先选择要测试的模型配置');return
        answer=W.QMessageBox.question(self,'锁定测试候选',f'将锁定所选配置的全部折和随机种子，并查看折外测试结果（当前选择{len(keys)}份权重）。查看后不能追加其他配置；测试不用于选轮次或参数。是否运行？')
        if answer==W.QMessageBox.Yes:self.run_test(keys)
    def run_test(self,keys=None):
        if self.busy() or not self.result_task:return
        self.launch(dict(task=str(self.result_directory),keys=None if self.result_task.get('format')=='cloth-classification-v4' else keys or list(self.result_task['models'])),'test')
    def refresh_history(self):
        self.history.blockSignals(True);self.legacy_history.blockSignals(True);self.task_history.blockSignals(True);self.history.clear();self.legacy_history.clear();self.task_history.clear()
        selected=self.prediction_experiment.currentData();self.prediction_experiment.blockSignals(True);self.prediction_experiment.clear()
        for task in self.task_catalog():
            legacy=task.get('format') not in ('cloth-classification-v3','cloth-classification-v4');w=self.legacy_history if legacy else self.task_history if task.get('status')!='completed' or task.get('purpose')=='quick' else self.history
            text=task['name']+'\n'+STATUS.get(task['status'],task['status'])+f' · {task.get("records",0)} CSV'+(' · 修复验证（非正式）' if task.get('validation_kind') else ' · 试跑' if task.get('purpose')=='quick' else '')
            item=W.QListWidgetItem(text);item.setData(C.Qt.UserRole,task['directory']);w.addItem(item)
            if task['status']=='completed' and task.get('purpose')!='quick' and (task.get('format')!='cloth-classification-v4' or task.get('final_model')):self.prediction_experiment.addItem(task['name']+('（旧版）' if legacy else ''),task['directory'])
        index=self.prediction_experiment.findData(selected) if selected else 0;self.prediction_experiment.setCurrentIndex(max(0,index));self.prediction_experiment.blockSignals(False);self.prediction_configs()
        self.history.blockSignals(False);self.legacy_history.blockSignals(False);self.task_history.blockSignals(False);self.filter_experiments(self.experiment_search.text())
    def history_choice(self):
        return (str(self.result_directory),self.result_selector.currentData()) if self.result_task else None
    def history_load(self):
        choice=self.history_choice()
        if not choice or not choice[1]:return
        if self.result_task.get('format')=='cloth-classification-v4':choice=(choice[0],'final')
        ix=self.prediction_experiment.findData(choice[0])
        if ix<0:self.notify('请选择已完成的正式模型');return
        self.prediction_experiment.setCurrentIndex(ix);self.prediction_configs()
        for j in range(self.prediction_config.count()):
            if choice[1] in self.prediction_config.itemData(j):self.prediction_config.setCurrentIndex(j);break
        self.prediction_runs();self.model_selector.setCurrentIndex(self.model_selector.findData(choice));self.navigation.setCurrentRow(3);self.notify('已选择具体模型，可添加CSV识别')
    def history_export(self):
        choice=self.history_choice()
        if not choice:return
        path,_=W.QFileDialog.getSaveFileName(self,'导出成果包',Path(choice[0]).name+'.zip','模型成果包 (*.zip)')
        if path:
            try:export_task(choice[0],path);self.notify('成果包已导出')
            except (ValueError,OSError) as exc:self.error(exc)
    def history_import(self):
        path,_=W.QFileDialog.getOpenFileName(self,'导入成果包','','模型成果包 (*.zip)')
        if path:
            try:import_task(path,self.store);self.refresh_history();self.notify('成果包已导入')
            except Exception as exc:self.error(exc)
    def history_rename(self):
        choice=self.history_choice()
        if not choice:return
        p=Path(choice[0])/'metadata.json';m=read_json(p);text,ok=W.QInputDialog.getText(self,'重命名实验','名称',text=m['name'])
        if ok and text.strip():m['name']=text.strip();write_json(p,m);self.refresh_history();self.show_results(choice[0])
    def history_open(self):
        choice=self.history_choice()
        if choice:G.QDesktopServices.openUrl(C.QUrl.fromLocalFile(choice[0]))

    def build_prediction(self):
        page=W.QWidget();layout=W.QVBoxLayout(page);layout.setContentsMargins(0,0,0,0)
        self.prediction_experiment=W.QComboBox();self.prediction_config=W.QComboBox();self.model_selector=W.QComboBox()
        for title,w in [('实验',self.prediction_experiment),('模型配置',self.prediction_config),('具体模型',self.model_selector)]:
            row=W.QHBoxLayout();row.addWidget(label(title));row.addWidget(w,1);layout.addLayout(row)
        self.prediction_experiment.currentIndexChanged.connect(self.prediction_configs);self.prediction_config.currentIndexChanged.connect(self.prediction_runs)
        actions=W.QHBoxLayout();actions.addWidget(button('添加CSV',self.choose_recognition));actions.addWidget(button('添加文件夹',self.choose_recognition_dir));actions.addWidget(button('清空',self.clear_recognition))
        actions.addStretch();self.predict_button=button('开始识别',self.start_prediction,True);self.predict_button.setEnabled(False);actions.addWidget(self.predict_button);actions.addWidget(button('导出结果',self.export_predictions));layout.addLayout(actions)
        self.prediction_summary=label('添加新CSV后，按模型已保存的配置处理。','notice');layout.addWidget(self.prediction_summary)
        self.prediction_table=table(['输入／方向','文件','真实标签','预测布料','扫描进度','结果']);self.prediction_table.itemSelectionChanged.connect(self.preview_prediction);layout.addWidget(self.prediction_table,1)
        self.prediction_tabs=W.QTabWidget();layout.addWidget(self.prediction_tabs,2);self.prediction_plot=Plot();self.prediction_tabs.addTab(self.prediction_plot,'单程类别得分')
        matrix=W.QWidget();ml=W.QVBoxLayout(matrix);controls=W.QHBoxLayout();self.matrix_model=W.QComboBox();self.matrix_scope=W.QComboBox();self.matrix_direction=W.QComboBox()
        for title,key in [('仅新数据','new'),('仅回放数据','replay'),('全部（含回放）','all')]:self.matrix_scope.addItem(title,key)
        for title,key in [('去回程合计','all'),('仅去程','out'),('仅回程','back')]:self.matrix_direction.addItem(title,key)
        controls.addWidget(self.matrix_model,1);controls.addWidget(self.matrix_scope);controls.addWidget(self.matrix_direction);controls.addWidget(button('导出混淆矩阵',self.choose_export_matrix));ml.addLayout(controls)
        self.matrix_notice=label('有真实标签的识别结果才能计算混淆矩阵；默认不计回放。','notice');ml.addWidget(self.matrix_notice);self.recognition_matrix=Plot();ml.addWidget(self.recognition_matrix,1);self.prediction_tabs.addTab(matrix,'识别混淆矩阵')
        self.recognition_metrics=classification_metrics([],[])
        for w in (self.matrix_model,self.matrix_scope,self.matrix_direction):w.currentIndexChanged.connect(self.draw_recognition_matrix)
        return page
    def prediction_configs(self,*args):
        directory=self.prediction_experiment.currentData();self.prediction_config.blockSignals(True);self.prediction_config.clear()
        if directory:
            meta=self.task_metadata(directory)
            rows=configurations(meta)
            if meta.get('format')=='cloth-classification-v4':
                f=meta.get('final_model');rows=[dict(architecture=f['architecture'],signal=f['signal'],readout=f['readout'],keys=['final'])] if f else []
            for row in rows:
                self.prediction_config.addItem(row['architecture'].upper()+' / '+input_label(row['signal'],meta['format'])+' / '+READOUTS.get(row['readout'],'旧版'),row['keys'])
        self.prediction_config.blockSignals(False);self.prediction_runs()
    def prediction_runs(self,*args):
        self.model_selector.clear();directory=self.prediction_experiment.currentData()
        if not directory:return
        meta=self.task_metadata(directory)
        for key in self.prediction_config.currentData() or []:
            r=meta['final_model'] if key=='final' else meta['models'][key];self.model_selector.addItem((f'第{r["fold"]+1}折 · ' if r.get('fold') is not None else '最终模型 · ' if key=='final' else '')+f'随机编号 {r.get("seed","旧版")} · {r.get("model_id",key)[:12]}',(directory,key))
    def choose_recognition(self):
        paths,_=W.QFileDialog.getOpenFileNames(self,'选择待识别CSV','','CSV (*.csv)')
        if paths:self.add_recognition(paths)
    def choose_recognition_dir(self):
        path=W.QFileDialog.getExistingDirectory(self,'选择待识别目录')
        if path:self.add_recognition([path])
    def add_recognition(self,paths):
        if self.busy():return
        self.recognition_paths=list(dict.fromkeys(self.recognition_paths+paths));self.prediction_summary.setText(f'已添加 {len(self.recognition_paths)} 个文件／目录');self.predict_button.setEnabled(self.model_selector.count()>0)
    def clear_recognition(self):
        if self.busy():return
        self.recognition_paths=[];self.prediction_table.setRowCount(0);self.prediction_rows=[];self.prediction_data=None;self.predict_button.setEnabled(False);self.prediction_summary.setText('待添加新CSV');self.matrix_model.clear();self.recognition_metrics=classification_metrics([],[]);self.recognition_matrix.empty('待识别有真实标签的数据');self.matrix_notice.setText('尚无可统计记录')
    def start_prediction(self):
        if self.busy() or not self.recognition_paths or not self.model_selector.currentData():return
        directory,key=self.model_selector.currentData();meta=self.task_metadata(directory)
        keys=[key]
        dest=Path(directory)/'recognition'/('识别_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4]+'.json')
        self.prediction_path=None;self.launch(dict(models=[str(Path(directory)/k/'model.pt') for k in keys],paths=self.recognition_paths,destination=str(dest)),'predict')
    def show_prediction(self,path):
        self.prediction_data=read_json(path);self.prediction_rows=[];errors=[];correct=total=0
        for result in self.prediction_data['results']:
            errors+=result['errors']
            for row in result['predictions']:
                self.prediction_rows.append((result,row))
                if row['true']>=0 and not row['training_overlap']:
                    total+=1;correct+=row['predicted']==row['true']
        self.prediction_table.setRowCount(len(self.prediction_rows))
        for i,(result,row) in enumerate(self.prediction_rows):
            o=row.get('outcome');text=('回放 · ' if row['training_overlap'] else '')+('整段分类' if o is None else '旧版接受' if o['accepted'] else '旧版完整长度未接受')
            row_values(self.prediction_table,i,[input_label(result['signal'],result.get('format','legacy-v2'))+' / '+('去' if row['direction']=='out' else '回'),Path(row['path']).stem,row['label'] or '无标签',result['labels'][row['predicted']],percent(o['stop_fraction']) if o else '100%',text])
        message=f'识别 {len(self.prediction_rows)} 个单程／模型结果'
        if len(self.prediction_data['results'])>1:message+=' · 多模型分别统计，请在混淆矩阵中选择结果来源'
        elif total:message+=f' · 新数据已知标签正确 {correct}/{total}'
        else:message+=' · 回放或无标签数据不计新数据准确率'
        if errors:message+='\n'+ '; '.join(Path(e['path']).name+'：'+e['message'] for e in errors[:3])
        self.prediction_summary.setText(message);self.navigation.setCurrentRow(3)
        self.matrix_model.blockSignals(True);self.matrix_model.clear()
        for i,r in enumerate(self.prediction_data['results']):
            name=r.get('experiment_name') or Path(r['model']).parent.parent.name or '已保存结果'
            self.matrix_model.addItem(name+' / '+Path(r['model']).parent.name+' / '+input_label(r['signal'],r.get('format','legacy-v2')),i)
            self.matrix_model.setItemData(i,r['model'],C.Qt.ToolTipRole)
        self.matrix_model.blockSignals(False);self.draw_recognition_matrix()
        if self.prediction_rows:self.prediction_table.selectRow(0)
    def draw_recognition_matrix(self,*args):
        if not self.prediction_data or self.matrix_model.currentData() is None:
            self.recognition_metrics=classification_metrics([],[]);self.recognition_matrix.empty('尚无识别结果');return
        result=self.prediction_data['results'][self.matrix_model.currentData()];scope=self.matrix_scope.currentData();direction=self.matrix_direction.currentData()
        rows=[r for r in result['predictions'] if (scope=='all' or bool(r.get('training_overlap'))==(scope=='replay')) and (direction=='all' or r['direction']==direction)]
        self.recognition_metrics=classification_metrics(rows,result['labels']);m=self.recognition_metrics;unknown=sum(r['true']<0 for r in rows);self.recognition_unknown=unknown
        self.matrix_notice.setText('结果来源：'+self.matrix_model.currentText()+' · '+str(result.get('model_id',''))[:8]+'\n'+f'{self.matrix_scope.currentText()} · {self.matrix_direction.currentText()} · 已知标签{m["total"]}单程 · 无标签/标签集外{unknown}单程未计入'+(f' · 准确率{percent(m["accuracy"])} · Macro-F1 {m["macro_f1"]:.3f}' if m['total'] else ' · 无可评价样本，准确率不适用'))
        if m['total']:self.recognition_matrix.confusion(m['confusion'],result['labels'])
        else:self.recognition_matrix.empty('没有符合筛选条件的真实标签，无法计算混淆矩阵')
    def choose_export_matrix(self):
        if not self.prediction_data:return
        path,_=W.QFileDialog.getSaveFileName(self,'导出当前混淆矩阵（PNG、CSV、JSON）','识别混淆矩阵.png','PNG (*.png)')
        if path:self.export_recognition_matrix(Path(path).with_suffix(''));self.notify('已导出当前模型与筛选范围的混淆矩阵')
    def export_recognition_matrix(self,base):
        if not self.prediction_data or self.matrix_model.currentData() is None:raise ValueError('尚无识别结果')
        import csv
        base=Path(base);base.parent.mkdir(parents=True,exist_ok=True);result=self.prediction_data['results'][self.matrix_model.currentData()]
        self.recognition_matrix.figure.savefig(str(base)+'.png',dpi=180)
        with Path(str(base)+'.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(['真实\预测']+result['labels'])
            for label,row in zip(result['labels'],self.recognition_metrics['confusion']):writer.writerow([label]+row)
        write_json(str(base)+'.json',dict(format='cloth-recognition-matrix-v1',model=result['model'],model_id=result.get('model_id'),weight_sha256=result.get('weight_sha256'),experiment_name=result.get('experiment_name'),fold=result.get('fold'),seed=result.get('seed'),excluded_unknown_labels=self.recognition_unknown,labels=result['labels'],scope=self.matrix_scope.currentData(),direction=self.matrix_direction.currentData(),metrics=self.recognition_metrics,description=self.matrix_notice.text()))
    def preview_prediction(self):
        i=self.prediction_table.currentRow()
        if 0<=i<len(self.prediction_rows):result,row=self.prediction_rows[i];self.prediction_plot.trial(row,result['labels'])
    def export_predictions(self):
        if not self.prediction_data:return
        path,_=W.QFileDialog.getSaveFileName(self,'导出识别结果','CERA识别结果.json','JSON (*.json)')
        if path:write_json(path,self.prediction_data);self.notify('识别结果已导出')

    def screening_value(self):
        return dict(self.screening,review_batches=self.batches,imports=self.import_reports,original_errors=self.screening.get('errors',[]),excluded=self.exclusions,selected_csv=len(self.selected_records()),errors=self.errors)
    def export_screening(self):
        path,_=W.QFileDialog.getSaveFileName(self,'导出筛查报告','导入筛查.json','JSON (*.json)')
        if path:write_json(path,self.screening_value());self.notify('已导出完整筛查与排除原因')
    def choose_save_dataset(self):
        parent=W.QFileDialog.getExistingDirectory(self,'选择数据集版本的父目录（自动创建新版本，不覆盖已有目录）')
        if parent:
            try:self.save_dataset_version(Path(parent)/('数据集_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4]))
            except Exception as exc:self.error(exc)
    def save_dataset_version(self,path):
        if self.busy() or self.import_jobs:raise ValueError('任务进行中，请等待后保存数据版本')
        if self.snapshot_path:load_dataset(self.snapshot_path)
        selected=self.selected_records();fixed=(self.snapshot_manifest or {}).get('split_plan');plan=self.selected_plan(selected,self.config.split_seed) if fixed or self.config.tcn_spec is not None else None
        folds=None
        if fixed:
            mapping={s['path']:i for i,s in enumerate(self.snapshot_manifest['sources'])};remap={mapping[r.path]:i for i,r in enumerate(selected)}
            folds=[[remap[i] for i in fold if i in remap]for fold in self.snapshot_manifest['folds']]
        else:folds=make_folds(selected,self.config.split_seed,max(2,getattr(self.config,'cv_folds',5)))
        save_dataset(path,selected,self.config.data,self.config.split_seed,self.screening_value(),folds=folds,verify_sources=not bool(self.snapshot_path),signals=self.selected_signals(),admission=package(self.batches),split_plan=plan,selection=(self.snapshot_manifest or {}).get('selection'))
        self.load_dataset_version(path);self.notify('数据集版本已冻结并载入；新增采集需新建版本')

    def selected_plan(self,records,seed):
        fixed=(self.snapshot_manifest or {}).get('split_plan')
        if fixed is None:return make_plan(records,seed,getattr(self.config,'cv_folds',5))
        if seed!=fixed['seed']:raise ValueError('当前数据版本的划分种子已冻结；请保留原种子')
        if len(fixed['cv'])!=getattr(self.config,'cv_folds',5):raise ValueError('数据版本的验证划分已冻结；更改折数需从原CSV新建数据版本')
        paths={s['path']:i for i,s in enumerate(self.snapshot_manifest['sources'])}
        if any(r.path not in paths for r in records):raise ValueError('所选记录不属于固定数据版本')
        mapping={paths[r.path]:i for i,r in enumerate(records)}
        def remap(indices):return [mapping[i]for i in indices if i in mapping]
        plan=dict(fixed,seed=seed,dev=remap(fixed['dev']),test=remap(fixed['test']),cv=[dict(train=remap(f['train']),val=remap(f['val']))for f in fixed['cv']])
        validate_plan(records,plan);return plan
    def choose_load_dataset(self):
        path=W.QFileDialog.getExistingDirectory(self,'选择含dataset.json的数据集版本目录')
        if path:
            try:self.load_dataset_version(path)
            except Exception as exc:self.error(exc)
    def load_dataset_version(self,path):
        if self.busy() or self.import_jobs:raise ValueError('请等待当前任务完成')
        records,manifest=load_dataset(path);self.snapshot_path=Path(path).resolve();self.snapshot_manifest=manifest
        self.batches=manifest.get('admission',{}).get('batches',[]);self.import_reports=[]
        validation=dict(cv_folds=len(manifest['split_plan']['cv'])) if manifest.get('split_plan') else {}
        self.config=replace(self.config,data=DataConfig.from_dict(manifest['data']),split_seed=manifest['split_seed'],signals=tuple(manifest.get('signals',self.config.signals)),**validation);self.sync_controls()
        paths={r.path for r in records};removed=set(manifest.get('selection',{}).get('excluded_labels',[]))
        missing={e['path']:dict(e,message='；'.join(e['audit']['reasons']) or '本版本未选择该记录的处理片段',review_only=True) for b in self.batches for e in b['entries'] if e['path'] not in paths and e.get('label') not in removed}
        self.records=[];self.errors=[];self.overrides={};self.data_table.setRowCount(0);self.screening=manifest['screening'];self.add_records(records,list(missing.values()))
        self.notify(f'已载入冻结版本 {manifest["dataset_id"][:10]} · {len(records)}CSV'+(' · 旧审查规则，仅供查看' if not manifest.get('admission') else ''))
    def resume_selected(self):
        if self.busy() or not self.result_task:return
        if self.result_task.get('format') not in ('cloth-classification-v3','cloth-classification-v4') or self.result_task.get('status')=='completed':self.notify('请选择任务记录中未完成的新版五折实验');return
        self.navigation.setCurrentRow(1);self.live_history=[];self.launch(dict(resume=str(self.result_directory)),'train')
    def write_project(self,path):
        selected={r.path for r in self.selected_records()};cfg=self.controls_config()
        write_json(path,dict(format='cloth-project-v3',config=cfg.to_dict(),store=str(self.store),overrides=self.overrides,
                             snapshot=str(self.snapshot_path) if self.snapshot_path else None,review_scopes=[b['candidates'] for b in self.batches],review_candidates=sorted({p for b in self.batches for p in b['candidates']}),sources=[dict(path=r.path,selected=r.path in selected,sha256=r.sha256,metadata_sha256=r.metadata_sha256) for r in self.records]))
    def read_project(self,path):
        if self.busy() or self.import_jobs:raise ValueError('当前任务完成后才能打开项目')
        value=read_json(path)
        if value.get('format') not in ('cera-project-v1','cloth-project-v2','cloth-project-v3'):raise ValueError('不支持的项目格式')
        self.config=ModernConfig.from_dict(value['config']) if value['config'].get('pipeline') else ModernConfig.from_dict(dict(value['config'],pipeline='gpu-standard-v4',device='cuda')); self.overrides=value.get('overrides',{});self.store=Path(value['store']);self.store.mkdir(parents=True,exist_ok=True)
        self.records=[];self.errors=[];self.batches=[];self.import_reports=[];self.data_table.setRowCount(0)
        if value.get('format')!='cloth-project-v3':self.config=replace(self.config,data=replace(self.config.data,preprocess_version='abs-v3'));self.notify('旧项目设置已迁移到新版；原模型保持旧协议')
        self.snapshot_path=None;self.snapshot_manifest=None;self.sync_controls();self.refresh_history()
        if value.get('snapshot'):
            self.load_dataset_version(value['snapshot']);chosen={x['path'] for x in value['sources'] if x['selected']}
            for i,r in enumerate(self.records):self.data_table.item(i,0).setCheckState(C.Qt.Checked if r.path in chosen else C.Qt.Unchecked)
            return
        self.import_paths(value.get('review_candidates') or [s['path'] for s in value['sources']],[s['path'] for s in value['sources'] if s['selected']],batch_scopes=value.get('review_scopes'))
    def save_project(self):
        path,_=W.QFileDialog.getSaveFileName(self,'保存项目','布料训练项目.json','JSON (*.json)')
        if path:
            try:self.write_project(path);self.notify('项目已保存')
            except Exception as exc:self.error(exc)
    def open_project(self):
        path,_=W.QFileDialog.getOpenFileName(self,'打开项目','','JSON (*.json)')
        if path:
            try:self.read_project(path)
            except Exception as exc:self.error(exc)
    def closeEvent(self,event):
        if self.import_jobs:self.pending_close=True;self.notify('等待导入线程完成后关闭…');event.ignore();return
        if self.busy():
            self.cancel_worker()
            if not self.process.waitForFinished(2000):self.process.kill();self.process.waitForFinished(2000)
        event.accept()
