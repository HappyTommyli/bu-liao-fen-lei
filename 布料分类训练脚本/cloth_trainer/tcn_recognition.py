"""Independent recognition UI; no training dataset or configuration is mutated."""
import csv
from datetime import datetime
from pathlib import Path
import uuid
from PyQt5 import QtWidgets as W, QtCore as C
from .ui import label, button, table, row_values
from .data import discover_report
from .storage import read_json
from .tcn_study import candidate_label


class TCNRecognitionMixin:
    def build_prediction(self):
        content=super().build_prediction()
        content.setObjectName('tcnRecognition')
        self.recognition_overrides={}
        self.recognition_input_issues=[]
        self.recognition_notice=label('独立新采集数据，仅用于识别；不会加入训练、交叉验证或调参。模型参数及预处理沿用已保存的最终模型。','notice')
        layout=content.layout();layout.insertWidget(0,self.recognition_notice)
        self.recognition_model_details=label('尚无可用最终模型：请在实验结果中选定配置并训练最终模型。','muted')
        layout.insertWidget(4,self.recognition_model_details)
        self.recognition_queue=table(['待识别CSV','真实标签（仅用于评价）'])
        self.recognition_queue.setMinimumHeight(125);self.recognition_queue.setMaximumHeight(190)
        layout.insertWidget(6,self.recognition_queue)
        actions=W.QHBoxLayout()
        actions.addWidget(button('指定真实标签',self.choose_recognition_label))
        actions.addWidget(button('恢复采集标签',self.reset_recognition_labels))
        actions.addWidget(button('移除所选',lambda:self.remove_recognition()))
        actions.addStretch();actions.addWidget(button('导出明细CSV',self.choose_export_recognition_csv))
        actions.addWidget(button('取消识别',self.cancel_worker));layout.insertLayout(7,actions)
        self.recognition_class_table=table(['布料','精确率','召回率','F1','真实单程数'])
        self.prediction_tabs.addTab(self.recognition_class_table,'各类表现')
        self.recognition_errors=W.QPlainTextEdit();self.recognition_errors.setReadOnly(True)
        self.prediction_tabs.addTab(self.recognition_errors,'导入与识别问题')
        self.prediction_table.setHorizontalHeaderLabels(['方向','文件','真实标签','预测布料','最高概率','数据来源'])
        self.prediction_table.setMinimumHeight(180);self.prediction_tabs.setMinimumHeight(360)
        self.prediction_experiment.currentIndexChanged.connect(self.update_recognition_model_details)
        self.model_selector.currentIndexChanged.connect(self.update_recognition_model_details)
        self.matrix_model.currentIndexChanged.connect(self.draw_recognition_classes)
        self.matrix_scope.currentIndexChanged.connect(self.draw_recognition_classes)
        self.matrix_direction.currentIndexChanged.connect(self.draw_recognition_classes)
        content.layout().setSizeConstraint(W.QLayout.SetMinimumSize)
        self.recognition_scroll=W.QScrollArea();self.recognition_scroll.setWidgetResizable(True)
        self.recognition_scroll.setWidget(content)
        return self.recognition_scroll

    def refresh_recognition_models(self):
        combo=self.prediction_experiment;selected=combo.currentData();combo.blockSignals(True)
        for i in range(combo.count()-1,-1,-1):
            meta=self.task_metadata(Path(combo.itemData(i)));final=meta.get('final_model') or {}
            if (meta.get('format')!='cloth-classification-v4' or meta.get('validation_kind')
                    or final.get('architecture') not in ('tcn','cera') or final.get('signal')!='by'):
                combo.removeItem(i)
        combo.setCurrentIndex(max(0,combo.findData(selected)));combo.blockSignals(False)
        self.prediction_configs();self.update_recognition_model_details()

    def update_recognition_model_details(self,*args):
        if not hasattr(self,'recognition_model_details'):return
        choice=self.model_selector.currentData()
        if choice:
            meta=self.task_metadata(Path(choice[0]));final=meta.get('final_model') or {}
            candidate=final.get('candidate')
            data=meta.get('config',{}).get('data',{})
            self.recognition_model_details.setText(
                f"最终模型：{meta['name']} · {candidate_label(candidate) if candidate else 'TCN'}\n"
                f"类别：{'、'.join(meta.get('labels',[]))}\n"
                f"固定 |dBy| · 预处理 {data.get('preprocess_version','abs-v3')} · 接触段 {data.get('contact_rule','模型保存规则')}；归一化及窗口汇总沿用模型。")
        else:self.recognition_model_details.setText('尚无可用最终模型：请在实验结果中选定配置并训练最终模型。')
        self.predict_button.setEnabled(bool(choice and self.recognition_paths) and not self.busy())

    def add_recognition(self,paths):
        if self.busy():return
        files,ignored,errors=discover_report(paths)
        usable=[str(p) for p in files if p.is_file() and p.suffix.lower()=='.csv']
        missing=[dict(path=str(p),message='文件不存在或不是CSV') for p in files if str(p) not in usable]
        self.recognition_paths=list(dict.fromkeys(self.recognition_paths+usable))
        issues=[e['path']+'：'+e.get('message',e.get('reason','')) for e in errors+missing+ignored]
        self.recognition_input_issues=list(dict.fromkeys(self.recognition_input_issues+issues))
        self.recognition_errors.setPlainText('\n'.join(self.recognition_input_issues))
        self.refresh_recognition_queue()
        if not usable:self.notify('未发现可识别CSV；请查看“导入与识别问题”。')

    def refresh_recognition_queue(self):
        self.recognition_queue.setRowCount(len(self.recognition_paths))
        for i,path in enumerate(self.recognition_paths):
            row_values(self.recognition_queue,i,[path,self.recognition_overrides.get(path,'自动读取采集标签')])
        self.prediction_summary.setText(f'待识别 {len(self.recognition_paths)} 个CSV；真实标签只用于评价，不参与模型预测。')
        self.update_recognition_model_details()

    def selected_recognition_indices(self):
        return sorted({i.row() for i in self.recognition_queue.selectionModel().selectedRows()})

    def choose_recognition_label(self):
        if self.busy():return
        indices=self.selected_recognition_indices()
        if not indices:self.notify('请先选择待识别文件。');return
        text,ok=W.QInputDialog.getText(self,'指定真实标签','输入真实布料名称；留空表示无标签（仅预测）：')
        if ok:self.set_recognition_labels(indices,text)

    def set_recognition_labels(self,indices,text):
        if self.busy():return
        for i in indices:self.recognition_overrides[self.recognition_paths[i]]=text.strip()
        self.refresh_recognition_queue()

    def reset_recognition_labels(self):
        if self.busy():return
        for i in self.selected_recognition_indices():self.recognition_overrides.pop(self.recognition_paths[i],None)
        self.refresh_recognition_queue()

    def remove_recognition(self,indices=None):
        if self.busy():return
        for i in sorted(set(self.selected_recognition_indices() if indices is None else indices),reverse=True):
            self.recognition_overrides.pop(self.recognition_paths.pop(i),None)
        self.refresh_recognition_queue()

    def clear_recognition(self):
        if self.busy():return
        super().clear_recognition();self.recognition_overrides.clear()
        self.recognition_input_issues.clear()
        self.recognition_queue.setRowCount(0);self.recognition_class_table.setRowCount(0)
        self.recognition_errors.clear();self.prediction_plot.empty('待识别新数据');self.prediction_path=None

    def start_prediction(self):
        if self.busy() or not self.recognition_paths or not self.model_selector.currentData():return
        directory,key=self.model_selector.currentData()
        if key!='final':self.notify('独立识别只允许使用最终模型。');return
        dest=Path(directory)/'recognition'/('独立识别_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4]+'.json')
        self.prediction_path=None
        self.launch(dict(models=[str(Path(directory)/'final/model.pt')],paths=list(self.recognition_paths),
                         overrides=dict(self.recognition_overrides),input_issues=list(self.recognition_input_issues),
                         independent=True,destination=str(dest)),'predict')
        self.prediction_summary.setText('正在按最终模型的固定配置识别新数据…')

    def show_prediction(self,path):
        super().show_prediction(path)
        errors=[]
        for i,(result,row) in enumerate(self.prediction_rows):
            overlap=row.get('training_overlap',False)
            row_values(self.prediction_table,i,[{'out':'去程','back':'回程'}[row['direction']],Path(row['path']).name,
                row['label'] or '无标签',result['labels'][row['predicted']],f"{row['scores'][row['predicted']]:.1%}",
                '与原实验重复（排除）' if overlap else '独立新数据'])
        for result in self.prediction_data['results']:errors.extend(result.get('errors',[]))
        issues=self.prediction_data.get('input_issues',[])+[e['path']+'：'+e['message'] for e in errors]
        self.recognition_errors.setPlainText('\n'.join(issues) or '没有导入或识别错误。')
        self.draw_recognition_classes()

    def on_event(self,event):
        super().on_event(event)
        if self.worker_kind=='predict' and event.get('message'):
            self.prediction_summary.setText(event['message'])

    def worker_finished(self,code,status):
        predicting=self.worker_kind=='predict'
        super().worker_finished(code,status)
        if predicting and (code!=0 or self.cancelled):
            self.prediction_summary.setText('识别已取消。' if self.cancelled else '识别失败；请查看“导入与识别问题”。')
            self.recognition_errors.setPlainText(self.stderr or '任务已取消／失败，未生成新识别结果。')

    def draw_recognition_classes(self,*args):
        if not hasattr(self,'recognition_class_table'):return
        metrics=self.recognition_metrics
        self.recognition_class_table.setRowCount(len(metrics.get('per_class',[])))
        for i,row in enumerate(metrics.get('per_class',[])):
            row_values(self.recognition_class_table,i,[row['label'],f"{row['precision']:.1%}",f"{row['recall']:.1%}",f"{row['f1']:.3f}",row['support']])

    def choose_export_recognition_csv(self):
        if not self.prediction_data:return
        path,_=W.QFileDialog.getSaveFileName(self,'导出识别明细','独立识别.csv','CSV (*.csv)')
        if path:self.export_recognition_csv(path);self.notify('识别明细已导出。')

    def export_recognition_csv(self,path):
        with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(['模型ID','权重SHA256','文件','方向','真实标签','预测布料','最高概率','与原实验重复'])
            for result,row in self.prediction_rows:
                writer.writerow([result.get('model_id',''),result.get('weight_sha256',''),row['path'],row['direction'],row['label'],
                                 result['labels'][row['predicted']],row['scores'][row['predicted']],row['training_overlap']])
