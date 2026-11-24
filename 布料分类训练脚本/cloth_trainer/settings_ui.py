"""Plain-language training settings separated from physical acquisition protocol."""
from dataclasses import replace
from PyQt5 import QtWidgets as W
from .config import TrainConfig,DataConfig

HELP={
'epochs':('最多学习几轮',1,1000,'每轮完整看一次训练数据。先用200；曲线仍改善时可增加。'),
'patience':('开启停滞结束后等待几轮',1,1000,'仅在开启训练停滞提前结束时生效；默认开启，验证损失连续30轮没有改善时结束。'),
'batch_size':('每次学习几个单程',1,128,'有效批默认16；显存不足时优先减小微批，保持每次更新的数据量。'),
'hidden_size':('模型宽度',32,128,'默认64；数据少时可试32，128通常需要更多数据。'),
'repetitions':('每项独立训练次数',1,10,'默认1；3次可观察初始化带来的波动，数据划分保持固定。'),
'threads':('使用几个CPU线程',1,32,'默认2，增加后不一定更快。'),
'seed':('训练随机编号',0,999999,'控制初始化和打乱顺序；复验依次加1。'),
'split_seed':('数据划分编号',0,999999,'控制哪些采集组进入训练/验证/测试；比较模型时保持相同。'),
'scheduler_patience':('几轮不降损失就减小步幅',1,1000,'默认8轮；自动把学习率减半。')}
FLOATS={
'lr':('学习步幅（学习率）','默认0.0003。曲线震荡可试0.0001；太小学习会变慢。'),
'weight_decay':('限制模型过度记忆','默认0.0001；训练好但验证差时可试0.001。'),
'dropout':('训练时随机忽略比例','默认0.1。仅训练时启用；过拟合可试0.2。'),
'min_lr':('最小学习步幅','默认0.00001；自动减小步幅时的下限。')}

class SettingsDialog(W.QDialog):
    def __init__(self,config,parent=None,protocol=False):
        super().__init__(parent);self.original=config;self.protocol=protocol;self.fields={};self.contact_rule=config.data.contact_rule;self.sample_quality=config.data.sample_quality
        self.setWindowTitle('采集协议' if protocol else '训练参数');self.resize(670,740)
        layout=W.QVBoxLayout(self);notice=W.QLabel('更改物理裁剪和时间规则后需重新检查CSV。' if protocol else '只影响下一次训练。默认值适合先进行第一阶段比较；将鼠标停在设置上可查看说明。');notice.setWordWrap(True);layout.addWidget(notice)
        scroll=W.QScrollArea();scroll.setWidgetResizable(True);body=W.QWidget();body.setObjectName('settingsBody');body.setStyleSheet('QWidget#settingsBody {background:#141d2b;}');form=W.QFormLayout(body);scroll.setWidget(body);layout.addWidget(scroll)
        if protocol:
            rows=[('out_low','去程开始位置 / mm',0,10000),('out_high','去程结束位置 / mm',0,10000),
                  ('back_low','回程下界 / mm',0,10000),('back_high','回程上界 / mm',0,10000),
                  ('fallback_ppmm','无元数据时脉冲 / mm',.1,100000),('fallback_speed','无元数据时速度 / mm/s',.01,1000),
                  ('source_fs','原始采样率 / Hz',1,10000),('tail_margin','接触段末尾余量 / s',0,10),('max_gap','允许数据缺口 / s',.003,.1)]
            for key,title,low,high in rows:
                w=W.QDoubleSpinBox();w.setDecimals(3);w.setRange(low,high);w.setValue(getattr(config.data,key));self.fields[key]=w;form.addRow(title,w)
            self.protocol_tip=W.QLabel();self.protocol_tip.setWordWrap(True);form.addRow(self.protocol_tip);self.update_protocol_tip()
        else:
            for key,(title,low,high,helptext) in HELP.items():
                if key=='hidden_size':
                    w=W.QComboBox()
                    for n in (32,64,128):w.addItem(str(n),n)
                    w.setCurrentIndex(w.findData(config.hidden_size))
                else:w=W.QSpinBox();w.setRange(low,high);w.setValue(getattr(config,key))
                w.setToolTip(helptext);self.fields[key]=w
                box=W.QWidget();vl=W.QVBoxLayout(box);vl.setContentsMargins(0,0,0,4);vl.addWidget(w);hint=W.QLabel(helptext);hint.setWordWrap(True);vl.addWidget(hint);form.addRow(title,box)
            for key,(title,helptext) in FLOATS.items():
                w=W.QLineEdit(str(getattr(config,key)));w.setToolTip(helptext);self.fields[key]=w;form.addRow(title,w)
            if hasattr(config,'workers'):
                for key,title,upper,hint in [('workers','数据准备进程数',8,'默认2；遇到进程问题可设0，不影响训练设备。'),('micro_batch','GPU每次实际处理几个单程',config.batch_size,'0为最长序列预检后自动确定；累积到有效批再更新。')]:
                    w=W.QSpinBox();w.setRange(0,upper);w.setValue(getattr(config,key));w.setToolTip(hint);self.fields[key]=w;form.addRow(title,w)
            self.early_stopping=W.QCheckBox('验证损失停滞时结束训练（与扫描提前停止不同）');self.early_stopping.setChecked(config.early_stopping);form.addRow(self.early_stopping)
            self.scheduler=W.QCheckBox('验证损失停滞时，自动减小学习步幅');self.scheduler.setChecked(config.scheduler);form.addRow(self.scheduler)
        self.error=W.QLabel();layout.addWidget(self.error)
        controls=W.QDialogButtonBox(W.QDialogButtonBox.Ok|W.QDialogButtonBox.Cancel|W.QDialogButtonBox.RestoreDefaults)
        controls.button(W.QDialogButtonBox.Ok).setText('确定');controls.button(W.QDialogButtonBox.Cancel).setText('取消');controls.button(W.QDialogButtonBox.RestoreDefaults).setText('恢复默认值')
        controls.accepted.connect(self.accept);controls.rejected.connect(self.reject);controls.button(W.QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore);layout.addWidget(controls)
    def restore(self):
        defaults=DataConfig() if self.protocol else type(self.original)()
        if self.protocol:self.contact_rule=defaults.contact_rule;self.sample_quality=defaults.sample_quality;self.update_protocol_tip()
        for k,w in self.fields.items():
            v=getattr(defaults,k)
            if isinstance(w,W.QComboBox):w.setCurrentIndex(w.findData(v))
            elif isinstance(w,W.QLineEdit):w.setText(str(v))
            else:w.setValue(v)
        if not self.protocol:self.scheduler.setChecked(True);self.early_stopping.setChecked(defaults.early_stopping)
    def accept(self):
        try:
            values={k:(w.currentData() if isinstance(w,W.QComboBox) else float(w.text()) if isinstance(w,W.QLineEdit) else w.value()) for k,w in self.fields.items()}
            if self.protocol:self.result_config=replace(self.original,data=replace(self.original.data,contact_rule=self.contact_rule,sample_quality=self.sample_quality,**values)).validate()
            else:self.result_config=replace(self.original,preset='quick' if self.original.preset=='quick' else 'custom',scheduler=self.scheduler.isChecked(),early_stopping=self.early_stopping.isChecked(),**values).validate()
        except (ValueError,TypeError) as exc:self.error.setText(str(exc));return
        super().accept()

    def update_protocol_tip(self):
        rule=('历史相对起点规则：第035次起去程移动44～144mm、回程从转折点移动11～111mm；更早记录使用上方范围。' if self.contact_rule=='relative-035-v1' else '所有采集序号均使用上方固定X范围。默认去程51～140mm，回程120→30mm。')
        quality=('仅检查所选输入通道；TCN仅检查By，忽略其他通道尖峰的连带标记。真实时间缺口、By异常及电机故障仍检查。' if self.sample_quality=='selected-axis-v1' else '历史采样质量规则：沿用采集器的三轴共用无效标记。恢复默认值可采用仅检查By的新规则。')
        self.protocol_tip.setText(rule+' '+quality+' 模型输入固定333Hz，按时间顺序取绝对值；尾部余量与允许缺口分别设置。')
