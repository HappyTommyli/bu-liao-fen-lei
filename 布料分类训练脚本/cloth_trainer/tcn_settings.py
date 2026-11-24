"""TCN parameters shared by the scrollable workbench and settings dialog."""
from dataclasses import replace
import re
from PyQt5 import QtWidgets as W,QtCore as C
from .tcn_models import default_spec,normalize_spec
from .cera_models import MODEL_REVISION as CERA_REVISION,normalize_cera_spec,PREFIX_REVISION,normalize_model_spec
from .tcn_study import MODE_TITLES


class TCNParameterEditor(W.QWidget):
    changed=C.pyqtSignal()

    def __init__(self,config,parent=None):
        super().__init__(parent)
        self.setObjectName('tcnParameters')
        self.fields={};layout=W.QVBoxLayout(self);layout.setContentsMargins(8,8,8,8);layout.setSpacing(12)
        layout.setSizeConstraint(W.QLayout.SetMinimumSize)
        def group(title):
            box=W.QGroupBox(title);form=W.QFormLayout(box)
            form.setFieldGrowthPolicy(W.QFormLayout.AllNonFixedFieldsGrow)
            form.setRowWrapPolicy(W.QFormLayout.WrapLongRows)
            form.setVerticalSpacing(10);layout.addWidget(box)
            return box,form
        def combo(form,key,title,items):
            widget=W.QComboBox()
            for text,data in items:widget.addItem(text,data)
            self.fields[key]=widget;form.addRow(title,widget);return widget
        def spin(form,key,title,low,high,decimals=None):
            widget=W.QSpinBox() if decimals is None else W.QDoubleSpinBox()
            if decimals is not None:widget.setDecimals(decimals)
            widget.setRange(low,high);widget.setKeyboardTracking(False)
            widget.setButtonSymbols(W.QAbstractSpinBox.PlusMinus)
            if decimals is not None:widget.setSingleStep(.0001 if key=='lr' else .1)
            self.fields[key]=widget;form.addRow(title,widget);return widget
        def line(form,key,title,hint):
            widget=W.QLineEdit();widget.setPlaceholderText(hint);widget.setToolTip(hint)
            self.fields[key]=widget;form.addRow(title,widget);return widget
        _,form=group('实验模式')
        combo(form,'architecture','模型',[('TCN','tcn'),('CERA（GRU＋TCN）','cera')])
        combo(form,'tcn_mode','运行方式',[(title,mode) for mode,title in MODE_TITLES.items()])
        spin(form,'cv_folds','验证方式 / 1为单次划分',1,10)
        self.fields['cv_folds'].setToolTip('1：训练60% / 验证20% / 测试20%；2–10：固定20%测试，其余开发集做K折。不会自动增加训练配置。')
        self.input_group,form=group('输入窗口')
        combo(form,'input_mode','输入方式',[('900点等长切片',1),('完整单程',0)])
        spin(form,'window_size','窗口点数',128,8192)
        self.fields['window_size'].setToolTip('窗口长度和训练步长独立设置；333点约1秒。')
        spin(form,'tcn_stride','训练步长 / 点',0,8192)
        self.stride_label=form.labelForField(self.fields['tcn_stride'])
        self.fields['tcn_stride'].setSpecialValueText('不重叠（等于窗口长度）')
        self.fields['tcn_stride'].setToolTip('900点窗口、450点步长为50%重叠。每条原始单程的总训练权重保持为1，尾段保留。')
        self.gru_group,form=group('GRU分支（CERA）')
        spin(form,'gru_hidden','隐藏维度',16,256)
        spin(form,'gru_layers','GRU层数',1,4)
        self.model_group,form=group('TCN结构 / 分支')
        spin(form,'width','通道宽度',16,256)
        spin(form,'depth','残差块数',1,10)
        combo(form,'kernel','卷积核',[(str(v),v) for v in range(3,16,2)])
        spin(form,'dropout','Dropout',0,.99,2)
        self.structure_toggle=W.QCheckBox('展开结构高级设置');layout.addWidget(self.structure_toggle)
        self.structure_group,form=group('结构高级设置')
        line(form,'channels','各块通道','留空使用统一宽度，例如32,64,128')
        line(form,'dilations','膨胀系数','留空使用1,2,4,8…；自定义项数等于块数')
        combo(form,'pooling','特征汇总',[('有效特征平均','mean'),('平均＋标准差','mean_std'),('有效特征最大值','max')])
        self.training_group,form=group('训练参数')
        spin(form,'lr','学习率',.00000001,1.,8)
        spin(form,'weight_decay','权重衰减',0,1.,6)
        spin(form,'batch_size','有效批 / 单程',1,128)
        spin(form,'micro_batch','GPU微批',0,128)
        self.fields['micro_batch'].setSpecialValueText('自动预检')
        self.fields['batch_size'].setToolTip('每次更新的原始单程数；其全部窗口累积梯度后更新一次。')
        self.fields['micro_batch'].setToolTip('同时前向的输入段数；切片模式为窗口数，完整模式为单程数。')
        spin(form,'epochs','最多轮数',1,1000)
        self.advanced_toggle=W.QCheckBox('展开训练高级设置');layout.addWidget(self.advanced_toggle)
        self.advanced_group,form=group('训练高级设置')
        for key,title in [('early_stopping','启用早停'),('scheduler','学习率调度')]:
            widget=W.QCheckBox(title);self.fields[key]=widget;form.addRow(widget)
        spin(form,'patience','早停等待轮数',1,1000)
        spin(form,'scheduler_patience','调度等待轮数',1,1000)
        spin(form,'scheduler_factor','学习率衰减倍率',.01,.99,2)
        spin(form,'min_lr','最低学习率',.00000001,1.,8)
        spin(form,'grad_clip_norm','梯度裁剪上限',.001,100000,3)
        spin(form,'seed','训练种子',0,999999)
        spin(form,'split_seed','划分种子',0,999999)
        spin(form,'repetitions','重复次数',1,10)
        spin(form,'workers','数据准备进程',0,8)
        spin(form,'threads','CPU线程',1,32)
        self.structure_toggle.toggled.connect(self.structure_group.setVisible)
        self.advanced_toggle.toggled.connect(self.advanced_group.setVisible)
        self.structure_group.hide();self.advanced_group.hide()
        for widget in self.fields.values():
            if isinstance(widget,W.QComboBox):widget.currentIndexChanged.connect(self._changed)
            elif isinstance(widget,W.QLineEdit):widget.textChanged.connect(self._changed)
            elif isinstance(widget,W.QCheckBox):widget.toggled.connect(self._changed)
            else:widget.valueChanged.connect(self._changed)
        self.set_config(config)

    def _changed(self,*args):
        self._mode_visibility();self.changed.emit()

    def _mode_visibility(self):
        cera=self.fields['architecture'].currentData()=='cera'
        mode_box=self.fields['tcn_mode']
        if not cera and mode_box.currentData()=='prefix':
            blocker=C.QSignalBlocker(mode_box);mode_box.setCurrentIndex(mode_box.findData('single'));del blocker
        mode_box.model().item(mode_box.findData('prefix')).setEnabled(cera)
        if cera and mode_box.currentData() in ('four_way','window_pool'):
            blocker=C.QSignalBlocker(mode_box);mode_box.setCurrentIndex(mode_box.findData('single'));del blocker
        for mode in ('four_way','window_pool'):mode_box.model().item(mode_box.findData(mode)).setEnabled(not cera)
        self.gru_group.setVisible(cera)
        pool=self.fields['pooling']
        if cera:
            blocker=C.QSignalBlocker(pool);pool.setCurrentIndex(pool.findData('mean'));del blocker
        pool.setEnabled(not cera)
        mode=self.fields['tcn_mode'].currentData();pair=mode=='overlap_pair';prefix=mode=='prefix'
        for key in ('architecture','tcn_mode','input_mode'):
            widget=self.fields[key];widget.setVisible(not prefix);widget.parentWidget().layout().labelForField(widget).setVisible(not prefix)
        self.fields['window_size'].parentWidget().layout().labelForField(self.fields['window_size']).setText('最少观察量 / 点' if prefix else '窗口点数')
        self.fields['window_size'].setToolTip('达到该有效点数才开始判断；每次使用从有效起点到当前的全部历史，333点约1秒。' if prefix else '窗口长度和训练步长独立设置；333点约1秒。')
        four=mode in ('four_way','window_pool')
        self.input_group.setVisible(not four);self.model_group.setVisible(not four)
        self.structure_toggle.setVisible(not four)
        self.structure_group.setVisible(not four and self.structure_toggle.isChecked())
        self.fields['input_mode'].setEnabled(not (pair or prefix))
        if pair or prefix:
            blocker=C.QSignalBlocker(self.fields['input_mode']);self.fields['input_mode'].setCurrentIndex(0);del blocker
        windowed=bool(self.fields['input_mode'].currentData())
        self.fields['window_size'].setEnabled(windowed)
        self.fields['tcn_stride'].setEnabled(windowed)
        stride=self.fields['tcn_stride'];size=self.fields['window_size'].value()
        self.stride_label.setText('前缀增长间隔 / 点' if prefix else 'B组步长 / 点（A组不重叠）' if pair else '训练步长 / 点')
        default_pair=pair and stride.value()==0;default_prefix=prefix and stride.value()==0
        blocker=C.QSignalBlocker(stride)
        stride.setRange(1 if pair or prefix else 0,8192 if prefix else size-1 if pair else size)
        if default_pair:stride.setValue(size//2)
        if default_prefix:stride.setValue(size)
        del blocker
        stride.setToolTip('训练前缀从有效起点不断增长；每轮每条单程抽取最多2个长度，总权重为1。' if prefix else ('B组训练步长；A组固定不重叠。' if pair else '')+'900点窗口、450点步长为50%重叠；每条单程总权重为1。')
        self.fields['input_mode'].setItemText(0,f"{self.fields['window_size'].value()}点等长切片")

    def set_config(self,config):
        self.base=config
        spec=config.tcn_spec or default_spec(config.hidden_size)
        values={key:getattr(config,key) for key in ('lr','weight_decay','batch_size','micro_batch','epochs','early_stopping','scheduler','patience',
                'scheduler_patience','scheduler_factor','min_lr','grad_clip_norm','seed','split_seed','repetitions','workers','threads','cv_folds','tcn_stride')}
        values.update(architecture=config.architectures[0],gru_hidden=spec.get('gru_hidden',128),gru_layers=spec.get('gru_layers',1),tcn_mode=config.tcn_mode,input_mode=int(config.tcn_window!=0),window_size=config.tcn_window or 900,
                      width=spec['channels'][0],depth=len(spec['channels']),kernel=spec['kernel_size'],dropout=spec['dropout'],pooling=spec['pooling'],
                      channels='' if len(set(spec['channels']))==1 else ','.join(map(str,spec['channels'])),
                      dilations='' if spec['dilations']==[2**i for i in range(len(spec['channels']))] else ','.join(map(str,spec['dilations'])))
        blockers=[C.QSignalBlocker(widget) for widget in self.fields.values()]
        self.fields['tcn_stride'].setRange(0,8192 if config.tcn_mode=='prefix' else config.tcn_window or 900)
        for key,value in values.items():
            widget=self.fields[key]
            if isinstance(widget,W.QComboBox):widget.setCurrentIndex(widget.findData(value))
            elif isinstance(widget,W.QLineEdit):widget.setText(value)
            elif isinstance(widget,W.QCheckBox):widget.setChecked(value)
            else:widget.setValue(value)
        del blockers
        self._mode_visibility()

    @staticmethod
    def integer_list(text,fallback):
        text=text.strip()
        if not text:return fallback
        if not re.fullmatch(r'[0-9,，\s]+',text):raise ValueError('列表只接受逗号分隔的正整数')
        return [int(v) for v in re.split(r'[,，\s]+',text) if v]

    def value(self,base=None):
        base=base or self.base
        f=self.fields;mode=f['tcn_mode'].currentData();depth=f['depth'].value();architecture=f['architecture'].currentData()
        if mode in ('four_way','window_pool'):spec=default_spec(128 if mode=='window_pool' else 64);window=900
        else:
            spec=default_spec(f['width'].value(),depth)
            spec.update(channels=self.integer_list(f['channels'].text(),spec['channels']),
                        dilations=self.integer_list(f['dilations'].text(),spec['dilations']),kernel_size=f['kernel'].currentData(),
                        dropout=f['dropout'].value(),pooling=f['pooling'].currentData())
            if len(spec['channels'])!=depth or len(spec['dilations'])!=depth:raise ValueError('通道/膨胀列表项数必须等于残差块数')
            spec=normalize_spec(spec)
            if architecture=='cera':spec=normalize_model_spec(dict(spec,revision=PREFIX_REVISION if mode=='prefix' else CERA_REVISION,gru_hidden=f['gru_hidden'].value(),gru_layers=f['gru_layers'].value()))
            window=f['window_size'].value() if f['input_mode'].currentData() else 0
        values={key:(widget.isChecked() if isinstance(widget,W.QCheckBox) else widget.value())
                for key,widget in f.items() if key in ('lr','weight_decay','batch_size','micro_batch','epochs','early_stopping','scheduler','patience',
                'scheduler_patience','scheduler_factor','min_lr','grad_clip_norm','seed','split_seed','repetitions','workers','threads','cv_folds')}
        values['tcn_stride']=f['tcn_stride'].value() if mode in ('single','overlap_pair','prefix') and window else 0
        return replace(base,architectures=(architecture,),signals=('by',),readouts=('mean',),tcn_mode=mode,tcn_spec=spec,
                       tcn_window=window,dropout=spec['dropout'],**values).validate()


class TCNSettingsDialog(W.QDialog):
    def __init__(self,config,parent=None):
        super().__init__(parent);self.setWindowTitle('TCN参数');self.resize(540,700)
        layout=W.QVBoxLayout(self);scroll=W.QScrollArea();scroll.setWidgetResizable(True)
        self.editor=TCNParameterEditor(config);scroll.setWidget(self.editor);layout.addWidget(scroll)
        self.error=W.QLabel();self.error.setWordWrap(True);layout.addWidget(self.error)
        buttons=W.QDialogButtonBox(W.QDialogButtonBox.Ok|W.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
    def accept(self):
        try:self.result_config=self.editor.value()
        except ValueError as exc:self.error.setText(str(exc));return
        super().accept()
