"""Qt desktop workbench. UI submits intent; all device I/O stays off this thread."""
import queue
import os
from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import Qt, QEvent, QTimer, QRectF, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QFont
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,
    QLabel,QPushButton,QComboBox,QDoubleSpinBox,QCheckBox,QLineEdit,QTableWidget,QHeaderView,
    QFileDialog,QPlainTextEdit,QSplitter,QGroupBox,QAbstractSpinBox,QAbstractItemView,QTabWidget,QDialog,QTableWidgetItem,QSpinBox,QMessageBox)

from serial.tools import list_ports
from .backend import Worker
from .batch import validate_batch
from .catalog import Catalog, valid_label, DEFAULT_SETTINGS
from .preferences import load_settings, update_settings
from .model import AxisConfig,Recipe,Step,plan_steps,load_recipe,save_recipe,finite


STYLE='''
QMainWindow,QDialog,QWidget { background:#141a21; color:#dce4eb; font-size:13px; }
QGroupBox { background:#1a222c; border:1px solid #2b3744; border-radius:10px; margin-top:13px; padding:12px 12px 8px; }
QGroupBox::title { subcontrol-origin:margin; left:14px; padding:0 6px; color:#91a4b5; }
QTabWidget::pane { background:#1a222c; border:1px solid #2b3744; border-radius:8px; }
QTabBar::tab { background:transparent; color:#8295a6; padding:10px 22px; border-bottom:2px solid transparent; }
QTabBar::tab:selected { color:#67dbc6; border-bottom-color:#43c9af; }
QPushButton { background:#25313e; color:#dce4eb; border:1px solid #344353; border-radius:6px; padding:8px 12px; min-height:18px; }
QPushButton:hover { background:#2b3d4a; border-color:#54bba9; }
QPushButton:pressed { background:#214b46; }
QPushButton:disabled { color:#627181; background:#1d2731; border-color:#2b3744; }
QPushButton[primary="true"] { background:#45c8ad; color:#102a27; border-color:#45c8ad; font-weight:600; }
QPushButton[primary="true"]:hover { background:#6bdec7; }
QPushButton[primary="true"]:disabled { background:#294b48; color:#68928a; border-color:#294b48; }
QPushButton[stop="true"] { background:#963c46; color:#fff0f0; border-color:#b14b55; font-weight:600; }
QLineEdit,QDoubleSpinBox,QSpinBox,QComboBox { background:#141c25; color:#e2eaf1; border:1px solid #344353; border-radius:5px; padding:6px; min-height:20px; selection-background-color:#286f65; }
QLineEdit:disabled,QDoubleSpinBox:disabled,QSpinBox:disabled,QComboBox:disabled { color:#607182; background:#1a242f; border-color:#2b3744; }
QComboBox QAbstractItemView { background:#1c2834; color:#e2eaf1; selection-background-color:#28544f; }
QTableWidget { background:#1a222c; gridline-color:#2b3744; border:1px solid #2b3744; }
QTableWidget::item:selected { background:#25483f; color:#dce4eb; }
QHeaderView::section { background:#222d39; color:#9fafbe; padding:7px; border:0; border-right:1px solid #2b3744; }
QPlainTextEdit { background:#151e27; color:#aabcca; border:1px solid #2b3744; border-radius:5px; }
QCheckBox { background:transparent; spacing:8px; }
QCheckBox::indicator { width:15px; height:15px; }
QLabel { background:transparent; }
QScrollBar:vertical { background:#1a222c; width:10px; }
QScrollBar::handle:vertical { background:#3c4c5c; min-height:20px; border-radius:4px; }
'''


def spin(value=0,minimum=-1e8,maximum=1e8,decimals=3):
    widget=QDoubleSpinBox(); widget.setDecimals(decimals); widget.setRange(minimum,maximum); widget.setValue(value)
    widget.setKeyboardTracking(False); widget.setMinimumWidth(75)
    return widget


def button(text,callback,primary=False):
    widget=QPushButton(text); widget.clicked.connect(callback)
    if primary: widget.setProperty('primary',True)
    return widget


class MagneticPlot(QWidget):
    def __init__(self):
        super().__init__(); self.history=[]; self.baseline=None; self.setMinimumHeight(155)

    def paintEvent(self,event):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(),QColor('#1a222c'))
        box=QRectF(58,28,max(1,self.width()-76),max(1,self.height()-53))
        colors=['#57d6bd','#75adff','#e6ac71']
        title='ΔB / µT'
        p.setPen(QColor('#96a8b9')); p.drawText(8,18,title)
        for i,name in enumerate(('ΔBx','ΔBy','ΔBz')):
            p.setPen(QColor(colors[i])); p.drawText(110+i*70,18,name)
        for i in range(5):
            y=box.top()+box.height()*i/4
            p.setPen(QColor('#2b3744')); p.drawLine(QPointF(box.left(),y),QPointF(box.right(),y))
        if self.baseline is None or not self.history:
            text='待校准 · 请先采集空载基线' if self.baseline is None else '等待校准后的新样本'
            p.setPen(QColor('#8a99a2')); p.drawText(box,Qt.AlignCenter,text); return
        end=self.history[-1][0]; start=end-10
        rows=[r for r in self.history if r[0]>=start]
        base=self.baseline or (0,0,0)
        values=[v-base[i] for r in rows for i,v in enumerate(r[1:])]
        low,high=min(values),max(values); pad=max((high-low)*.12,.1); low-=pad; high+=pad
        p.setPen(QColor('#96a8b9')); p.drawText(1,39,f'{high:.1f}'); p.drawText(1,int(box.bottom()),f'{low:.1f}')
        p.drawText(int(box.left()),self.height()-5,'−10 s'); p.drawText(int(box.right())-35,self.height()-5,'现在')
        for i in range(3):
            p.setPen(QPen(QColor(colors[i]),1.6)); previous=None
            for row in rows:
                point=QPointF(box.left()+(row[0]-start)/10*box.width(),box.bottom()-(row[i+1]-base[i]-low)/(high-low)*box.height())
                if previous is not None: p.drawLine(previous,point)
                previous=point


class Window(QMainWindow):
    def __init__(self,worker=None):
        super().__init__(); self.setWindowTitle('XZ 工作台'); self.resize(1250,900)
        self.worker=worker or Worker(); self.state={}; self.held=None; self.recipe_config={}
        self.output_root=Path(__file__).resolve().parents[1]/'data'/'experiments_linux'
        self.last_saved=None; self.setStyleSheet(STYLE)
        self.settings_path=getattr(self.worker,'settings_path',DEFAULT_SETTINGS)
        self._saved_session=None; self._save_error=None; self.saved_start=None
        root=QWidget(); outer=QVBoxLayout(root); outer.setContentsMargins(24,18,24,14); outer.setSpacing(14)
        self.setCentralWidget(root)
        top=QHBoxLayout(); title=QLabel('XZ 工作台'); title.setStyleSheet('font-size:24px;font-weight:700;')
        top.addWidget(title); self.connection_status=QLabel('未连接'); top.addWidget(self.connection_status); top.addStretch()
        self.device_dialog=QDialog(self); self.device_dialog.setWindowTitle('设备连接'); self.device_dialog.resize(700,260)
        self.device_dialog.setModal(True); dl=QVBoxLayout(self.device_dialog)
        self.motor_ports=QComboBox(); self.sensor_ports=QComboBox(); self.motor_ports.setMinimumWidth(320); self.sensor_ports.setMinimumWidth(320)
        self.motor_connect=button('连接电机',self.toggle_motor); self.sensor_connect=button('连接传感器',self.toggle_sensor)
        self.sensor_baud=QComboBox()
        for baud in (115200,230400,460800,921600,9600): self.sensor_baud.addItem(str(baud),baud)
        row=QHBoxLayout(); row.addWidget(QLabel('电机')); row.addWidget(self.motor_ports,1); row.addWidget(self.motor_connect); dl.addLayout(row)
        row=QHBoxLayout(); row.addWidget(QLabel('磁传感器')); row.addWidget(self.sensor_ports,1); row.addWidget(self.sensor_baud); row.addWidget(self.sensor_connect); dl.addLayout(row)
        self.refresh_button=button('刷新串口',self.refresh_ports); dl.addWidget(self.refresh_button)
        info=QLabel('可输入 /dev/serial/by-id/… 路径。电机与传感器必须使用不同设备。\n模拟设备不连接硬件，模拟记录不用于实验。'); info.setWordWrap(True); dl.addWidget(info)
        dl.addWidget(button('完成',self.device_dialog.close))
        self.devices_button=button('设备连接',lambda:self.show_panel(self.device_dialog)); top.addWidget(self.devices_button)
        self.settings_dialog=QDialog(self); self.settings_dialog.setWindowTitle('运动参数'); self.settings_dialog.resize(500,470); self.settings_dialog.setModal(True)
        self.settings_button=button('运动参数',lambda:self.show_panel(self.settings_dialog)); top.addWidget(self.settings_button)
        self.stop_button=button('■  停止  [空格]',self.stop_all); self.stop_button.setProperty('stop',True); top.addWidget(self.stop_button)
        outer.addLayout(top)
        positions=QHBoxLayout(); self.position=QLabel('X  — mm       Z  — mm'); self.position.setStyleSheet('font-size:28px;font-weight:600;padding:6px 0;')
        positions.addWidget(self.position); positions.addStretch(); self.reference=QLabel('等待连接'); self.reference.setWordWrap(True); positions.addWidget(self.reference); outer.addLayout(positions)
        self.workspace_tabs=QTabWidget(); outer.addWidget(self.workspace_tabs,5)
        manual=QWidget(); manual_layout=QHBoxLayout(manual); manual_layout.setContentsMargins(28,18,28,18); manual_layout.setSpacing(45)
        self.workspace_tabs.addTab(manual,'  手动控制  ')
        left=QVBoxLayout(); left.setSpacing(10); self.speeds={}; self.accels={}; self.config_widgets=[]
        speed_grid=QGridLayout(); speed_grid.setHorizontalSpacing(12)
        for row,a in enumerate(('x','z')):
            speed_grid.addWidget(QLabel(a.upper()+' 速度'),row,0)
            self.speeds[a]=spin(1,.001,100000); self.speeds[a].setSuffix(' mm/s'); speed_grid.addWidget(self.speeds[a],row,1)
            self.accels[a]=spin(2,.001,1000000); self.accels[a].setSuffix(' mm/s²'); self.config_widgets.extend((self.speeds[a],self.accels[a]))
        left.addLayout(speed_grid)
        mode=QHBoxLayout(); self.jog_mode=QComboBox(); self.jog_mode.addItems(['按住移动','点击定距'])
        self.jog_mode.currentIndexChanged.connect(self.release_current); self.distance=spin(1,.001,1e8); self.distance.setSuffix(' mm')
        mode.addWidget(self.jog_mode); mode.addWidget(self.distance); left.addLayout(mode)
        self.keyboard_enabled=QCheckBox('启用方向键控制移动'); self.keyboard_enabled.setChecked(False)
        self.keyboard_enabled.setStyleSheet('QCheckBox::indicator { border:1px solid #7f96aa; border-radius:3px; background:#141c25; } QCheckBox::indicator:checked { background:#45c8ad; border-color:#45c8ad; }')
        self.keyboard_enabled.toggled.connect(self.release_current); left.addWidget(self.keyboard_enabled)
        self.workspace_tabs.currentChanged.connect(self.release_current)
        hint=QLabel('仅手动页主窗口：← → 移动 X，↑ ↓ 调整 Z\n按住移动 · 松开停止'); hint.setStyleSheet('color:#91a4b5;'); left.addWidget(hint)
        self.arm_button=button('使能运动',lambda:self.submit('arm'),True); left.addWidget(self.arm_button)
        self.origin_dialog=QDialog(self); self.origin_dialog.setWindowTitle('人工机械标定与固定采集起点'); self.origin_dialog.resize(650,520)
        ol=QVBoxLayout(self.origin_dialog)
        guide=QLabel('先把机构对准现实中的固定定位块/刻线，再确认机械原点。\n需要电动对位时，先建立临时对位坐标，再手动使能和点动；对位后必须重新确认原点。'); guide.setWordWrap(True); ol.addWidget(guide)
        origin_row=QHBoxLayout(); origin_row.addWidget(QLabel('固定基准名称')); self.origin_name=QLineEdit('人工机械基准'); origin_row.addWidget(self.origin_name); ol.addLayout(origin_row)
        origin_actions=QHBoxLayout(); self.align_button=button('开始临时对位',self.begin_alignment)
        self.confirm_origin_button=button('已对准基准 · 确认机械原点',self.confirm_origin,True)
        origin_actions.addWidget(self.align_button); origin_actions.addWidget(self.confirm_origin_button); ol.addLayout(origin_actions)
        point_grid=QGridLayout(); self.start_x=spin(); self.start_z=spin(); self.start_clearance=spin()
        for row,(text,widget) in enumerate([('采集起点 X / mm',self.start_x),('采集起点 Z / mm',self.start_z),('安全 Z 坐标 / mm（越小越高）',self.start_clearance)]):
            point_grid.addWidget(QLabel(text),row,0); point_grid.addWidget(widget,row,1)
        ol.addLayout(point_grid)
        path_hint=QLabel('返回顺序：先 Z 抬到指定安全高度 → X 到起点 → Z 到起点。\n当前 Z 已更高时保持该高度。安全高度由你核对，软件无法判断是否碰撞。'); path_hint.setWordWrap(True); ol.addWidget(path_hint)
        point_actions=QHBoxLayout(); self.save_start_current=button('将当前位置保存为起点',lambda:self.save_capture_start(True))
        self.save_start_input=button('保存输入的起点',lambda:self.save_capture_start(False))
        point_actions.addWidget(self.save_start_current); point_actions.addWidget(self.save_start_input); ol.addLayout(point_actions)
        self.origin_status=QLabel('机械原点待人工确认'); self.origin_status.setWordWrap(True); ol.addWidget(self.origin_status); ol.addWidget(button('完成',self.origin_dialog.close))
        self.zero_all=button('人工标定 / 固定采集起点…',lambda:self.show_panel(self.origin_dialog)); left.addWidget(self.zero_all)
        self.go_start_button=button('移动到固定采集起点',self.go_capture_start); left.addWidget(self.go_start_button)
        self.point_summary=QLabel('未保存固定采集起点'); self.point_summary.setWordWrap(True); left.addWidget(self.point_summary)
        self.coordinate_dialog=QDialog(self); self.coordinate_dialog.setWindowTitle('坐标操作'); self.coordinate_dialog.setModal(True); self.coordinate_dialog.resize(520,240); zl=QVBoxLayout(self.coordinate_dialog)
        zl.addWidget(QLabel('单轴设零会使固定机械标定失效；X/Z 回零返回机械坐标 0，不是采集起点。'))
        zr=QHBoxLayout(); self.zero_x=button('临时 X 设零',lambda:self.submit('zero',axis='X')); self.zero_z=button('临时 Z 设零',lambda:self.submit('zero',axis='Z'))
        self.return_x=button('X 回零',lambda:self.single_move('zero','X',0)); self.return_z=button('Z 回零',lambda:self.single_move('zero','Z',0))
        for w in (self.zero_x,self.zero_z,self.return_x,self.return_z): zr.addWidget(w)
        zl.addLayout(zr)
        absolute=QHBoxLayout(); self.absolute_axis=QComboBox(); self.absolute_axis.addItems(['X','Z']); self.absolute_target=spin()
        self.absolute_button=button('移动到坐标',lambda:self.single_move('absolute',self.absolute_axis.currentText(),self.absolute_target.value()))
        for w in (self.absolute_axis,self.absolute_target,self.absolute_button): absolute.addWidget(w)
        zl.addLayout(absolute); self.clear_button=button('复位故障',lambda:self.submit('clear')); zl.addWidget(self.clear_button); zl.addWidget(button('完成',self.coordinate_dialog.close))
        left.addWidget(button('更多坐标操作…',lambda:self.show_panel(self.coordinate_dialog)))
        left.addStretch(); manual_layout.addLayout(left,2)
        directions=QGridLayout(); directions.setSpacing(12); self.direction_buttons={}
        for text,direction,row,col in [('↑\nZ 上','up',0,1),('←\nX 左','left',1,0),('→\nX 右','right',1,2),('↓\nZ 下','down',2,1)]:
            b=QPushButton(text); b.setMinimumSize(105,72); b.setStyleSheet('font-size:17px;'); b.setFocusPolicy(Qt.NoFocus)
            b.pressed.connect(lambda d=direction:self.direction_pressed(d,'mouse')); b.released.connect(lambda d=direction:self.direction_released(d,'mouse'))
            self.direction_buttons[direction]=b; directions.addWidget(b,row,col)
        center=QLabel('X / Z'); center.setAlignment(Qt.AlignCenter); center.setStyleSheet('color:#687985;font-size:20px;'); directions.addWidget(center,1,1)
        manual_layout.addLayout(directions,3)
        sl=QVBoxLayout(self.settings_dialog); ag=QGridLayout(); self.ppmm={}; self.bounds={}
        for row,a in enumerate(('x','z')):
            ag.addWidget(QLabel(a.upper()+' 加速度'),row,0); ag.addWidget(self.accels[a],row,1)
            self.ppmm[a]=spin(250,.001,1e9); self.config_widgets.append(self.ppmm[a])
            ag.addWidget(QLabel(a.upper()+' 脉冲 / mm'),row+2,0); ag.addWidget(self.ppmm[a],row+2,1)
        self.right_sign=QComboBox(); self.right_sign.addItem('向右 = X 正向',1); self.right_sign.addItem('向右 = X 负向',-1); self.config_widgets.append(self.right_sign)
        ag.addWidget(QLabel('方向映射'),4,0); ag.addWidget(self.right_sign,4,1)
        self.bound_enabled=QCheckBox('启用软件边界'); self.config_widgets.append(self.bound_enabled); ag.addWidget(self.bound_enabled,5,0,1,2)
        for row,a in enumerate(('x','z'),6):
            ag.addWidget(QLabel(a.upper()+' 最小 / 最大 mm'),row,0); br=QHBoxLayout(); lo=spin(-100); hi=spin(100); self.bounds[a]=(lo,hi)
            br.addWidget(lo); br.addWidget(hi); ag.addLayout(br,row,1); self.config_widgets.extend((lo,hi))
        sl.addLayout(ag); sl.addWidget(QLabel('重新设零后软件边界清空。软件边界不替代硬件限位。'))
        self.apply_config=button('应用参数',self.sync_config,True); sl.addWidget(self.apply_config); sl.addWidget(button('完成',self.settings_dialog.close))
        recipes=QWidget(); rl=QVBoxLayout(recipes); rl.setContentsMargins(16,16,16,16); self.workspace_tabs.addTab(recipes,'  动作函数  ')
        names=QHBoxLayout(); self.recipe_name=QLineEdit('我的扫描流程'); names.addWidget(self.recipe_name,1)
        self.new_button=button('新建',self.new_recipe); self.load_button=button('打开',self.load_dialog); self.save_button=button('保存',self.save_dialog)
        for w in (self.new_button,self.load_button,self.save_button): names.addWidget(w)
        rl.addLayout(names)
        self.table=QTableWidget(0,9); self.table.setHorizontalHeaderLabels(['启用','动作','轴','方向','距离 / 坐标 mm','速度 mm/s','加速度 mm/s²','时间 / 超时 s','备注'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows); self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents); self.table.horizontalHeader().setStretchLastSection(True); self.table.verticalHeader().setDefaultSectionSize(40)
        self.table.setMinimumHeight(135); rl.addWidget(self.table,1)
        edit=QHBoxLayout(); self.edit_buttons=[]
        for text,callback in [('＋ 添加',lambda:self.add_step()),('复制',self.duplicate_row),('删除',self.delete_row),('↑',lambda:self.move_row(-1)),('↓',lambda:self.move_row(1))]:
            w=button(text,callback); self.edit_buttons.append(w); edit.addWidget(w)
        edit.addStretch(); self.check_button=button('检查路径',self.check_recipe); self.adopt_button=button('采用当前配置',self.adopt_config)
        edit.addWidget(self.check_button); edit.addWidget(self.adopt_button)
        self.run_button=button('运行',lambda:self.run_recipe(False),True); self.run_record_button=button('运行并采集',lambda:self.run_recipe(True),True); self.abort_button=button('中止',self.stop_all)
        for w in (self.run_button,self.run_record_button,self.abort_button): edit.addWidget(w)
        rl.addLayout(edit)
        batch_row=QHBoxLayout(); batch_row.addWidget(QLabel('批量次数'))
        self.batch_count=QSpinBox(); self.batch_count.setRange(1,10000); self.batch_count.setValue(1); self.batch_count.setKeyboardTracking(False)
        batch_row.addWidget(self.batch_count); batch_row.addWidget(QLabel('轮次间隔'))
        self.batch_interval=spin(0,0,86400); self.batch_interval.setSuffix(' s'); batch_row.addWidget(self.batch_interval)
        self.batch_button=button('批量运行并采集',self.run_batch,True)
        self.batch_stop_button=button('本轮结束后停止',lambda:self.submit('batch_stop_after'))
        batch_row.addWidget(self.batch_button); batch_row.addWidget(self.batch_stop_button); batch_row.addStretch(); rl.addLayout(batch_row)
        self.batch_progress=QLabel('每轮独立保存 · 从上一轮终点继续执行，请在函数中安排回程。'); self.batch_progress.setStyleSheet('color:#91a4b5;'); rl.addWidget(self.batch_progress)
        self.progress=QLabel('按顺序执行，到位后进入下一步。'); rl.addWidget(self.progress)
        self.preview=QPlainTextEdit(); self.preview.setReadOnly(True); self.preview.setMaximumHeight(65); self.preview.setPlaceholderText('检查后显示每步起终点与预计时间。'); rl.addWidget(self.preview)
        acquisition=QGroupBox('磁数据'); acquisition.setMaximumHeight(310); al=QVBoxLayout(acquisition)
        forward=QHBoxLayout()
        self.realtime_enabled=QCheckBox('允许实时转发（仅本机）')
        self.realtime_start=button('开始转发',lambda:self.submit('realtime_forward',enabled=True))
        self.realtime_stop=button('停止转发',lambda:self.submit('realtime_forward',enabled=False))
        self.realtime_start.setEnabled(False); self.realtime_stop.setEnabled(False)
        self.realtime_enabled.toggled.connect(self._toggle_realtime)
        self.realtime_status=QLabel('默认关闭 · 127.0.0.1:8766')
        for widget in (self.realtime_enabled,self.realtime_start,self.realtime_stop,self.realtime_status): forward.addWidget(widget)
        al.addLayout(forward)
        capture=QHBoxLayout(); self.values=QLabel('ΔBx —    ΔBy —    ΔBz — µT · 待校准'); self.values.setStyleSheet('font-size:17px;font-weight:500;'); capture.addWidget(self.values,1)
        self.capture_dialog=QDialog(self); self.capture_dialog.setWindowTitle('采集设置'); self.capture_dialog.setModal(True); self.capture_dialog.resize(620,230); cl=QVBoxLayout(self.capture_dialog)
        self.baseline_status=QLabel('未校准 · 保持静止空载后采集基线'); cl.addWidget(self.baseline_status)
        self.baseline_button=button('采集空载基线 · 2 秒',lambda:self.submit('baseline'),True); cl.addWidget(self.baseline_button)
        self.output_label=QLabel(str(self.output_root)); self.output_label.setWordWrap(True); self.output_label.setTextInteractionFlags(Qt.TextSelectableByMouse); cl.addWidget(self.output_label)
        self.output_button=button('选择保存目录',self.choose_output); cl.addWidget(self.output_button)
        self.manage_button=button('管理记录…',self.show_records); cl.addWidget(self.manage_button)
        cl.addWidget(button('完成',self.capture_dialog.close))
        self.capture_settings=button('采集设置',lambda:self.show_panel(self.capture_dialog)); capture.addWidget(self.capture_settings)
        self.record_start=button('开始采集',lambda:self.start_record(),True); self.record_stop=button('结束保存',lambda:self.submit('record_stop'))
        capture.addWidget(self.record_start); capture.addWidget(self.record_stop); al.addLayout(capture)
        naming=QHBoxLayout(); naming.addWidget(QLabel('布料名称'))
        self.record_label=QLineEdit(Catalog(self.output_root,getattr(self.worker,'settings_path',None)).last_label()); self.record_label.setMaximumWidth(260)
        self.record_label.setPlaceholderText('例如：棉布'); naming.addWidget(self.record_label)
        self.next_record=QLabel(); self.next_record.setStyleSheet('color:#91a4b5;'); naming.addWidget(self.next_record,1); al.addLayout(naming)
        self.record_label.textChanged.connect(self.update_next_name); self.update_next_name()
        self.records_dialog=QDialog(self); self.records_dialog.setWindowTitle('记录管理'); self.records_dialog.resize(900,440)
        management=QVBoxLayout(self.records_dialog)
        management.addWidget(QLabel('可多选删除至回收站；编号释放后可复用。重命名请选择一条记录，CSV 内容保持不变。'))
        self.record_tabs=QTabWidget(); management.addWidget(self.record_tabs)
        current_records=QWidget(); current_layout=QVBoxLayout(current_records); self.record_tabs.addTab(current_records,'当前记录')
        self.records_table=QTableWidget(0,5); self.records_table.setHorizontalHeaderLabels(['记录名','布料标签','采集时间','结果','样本数'])
        self.records_table.setEditTriggers(QAbstractItemView.NoEditTriggers); self.records_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.records_table.setSelectionMode(QAbstractItemView.ExtendedSelection); self.records_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.records_table.horizontalHeader().setStretchLastSection(True); current_layout.addWidget(self.records_table)
        rename_row=QHBoxLayout(); self.rename_label=QLineEdit(); self.rename_label.setPlaceholderText('新的布料名称')
        rename_row.addWidget(self.rename_label,1); self.rename_button=button('重命名所选记录',self.rename_record)
        self.delete_button=button('删除所选记录',self.delete_records)
        rename_row.addWidget(self.rename_button); rename_row.addWidget(self.delete_button); rename_row.addWidget(button('刷新',self.refresh_records)); current_layout.addLayout(rename_row)
        recycled=QWidget(); recycled_layout=QVBoxLayout(recycled); self.record_tabs.addTab(recycled,'回收站')
        self.trash_table=QTableWidget(0,2); self.trash_table.setHorizontalHeaderLabels(['原记录名','删除时间'])
        self.trash_table.setEditTriggers(QAbstractItemView.NoEditTriggers); self.trash_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.trash_table.setSelectionMode(QAbstractItemView.SingleSelection); self.trash_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        recycled_layout.addWidget(self.trash_table)
        self.restore_button=button('恢复所选记录',self.restore_record); recycled_layout.addWidget(self.restore_button)
        recycled_layout.addWidget(QLabel('恢复时原名称如已被使用，将分配新的可用编号。'))
        self.records_table.itemSelectionChanged.connect(self.selected_record_changed)
        self.records_status=QLabel(''); self.records_status.setWordWrap(True); management.addWidget(self.records_status)
        management.addWidget(button('完成',self.records_dialog.close))
        self.plot=MagneticPlot(); self.plot.setMinimumHeight(115); al.addWidget(self.plot,1)
        self.record_status=QLabel('未录制'); al.addWidget(self.record_status); outer.addWidget(acquisition,2)
        self.notice=QLabel('连接设备 → 核对参数 → 设零并使能，即可开始操作。'); self.notice.setWordWrap(True); outer.addWidget(self.notice)
        logrow=QHBoxLayout(); self.log_toggle=QCheckBox('显示操作日志'); logrow.addWidget(self.log_toggle); logrow.addStretch()
        footer=QLabel('软件坐标 · 松开减速停止 · 无硬件限位保护'); footer.setStyleSheet('color:#73818c;font-size:11px;'); logrow.addWidget(footer); outer.addLayout(logrow)
        self.log=QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(300); self.log.setMaximumHeight(85); self.log.hide(); outer.addWidget(self.log)
        self.log_toggle.toggled.connect(self.log.setVisible)
        self.refresh_ports(); QApplication.instance().installEventFilter(self)
        self.timer=QTimer(self); self.timer.timeout.connect(self.poll); self.timer.start(50)
        self.hold_timer=QTimer(self); self.hold_timer.timeout.connect(self.renew_hold); self.hold_timer.start(80)
        if worker is None: self.worker.start()
        self.poll()
        self.restore_session()
        self.session_timer=QTimer(self); self.session_timer.timeout.connect(self.save_session); self.session_timer.start(500)

    def session_data(self):
        # Keep the draft, including currently inactive bounds, separate from live device state.
        cfg=self.config().to_dict()
        recipe=self.read_recipe()
        data=dict(version=1,output_root=str(self.output_root.absolute()),label=self.record_label.text(),
                  config=cfg,bounds={a:[w.value() for w in self.bounds[a]] for a in ('x','z')},
                  distance=self.distance.value(),jog_mode=self.jog_mode.currentIndex(),
                  recipe=dict(name=recipe.name,steps=[asdict(s) for s in recipe.steps],config=recipe.config),
                  sensor_baud=self.sensor_baud.currentData(),
                  manual_origin=dict(name=self.origin_name.text(),point=self.saved_start))
        if hasattr(self,'batch_count'): data['batch']=dict(count=self.batch_count.value(),interval=self.batch_interval.value())
        return data

    def save_session(self):
        try:
            data=self.session_data()
            if data!=self._saved_session:
                changes={'session':data}
                try: changes['last_label']=valid_label(data['label'])
                except ValueError: pass
                update_settings(self.settings_path,changes); self._saved_session=data; self._save_error=None
        except (ValueError,OSError) as exc:
            if self._save_error!=str(exc): self.message('工作台设置保存失败：'+str(exc),True); self._save_error=str(exc)

    def restore_session(self):
        try:
            data=load_settings(self.settings_path).get('session')
            if data is None: return
            if not isinstance(data,dict) or data.get('version')!=1: raise ValueError('设置版本无效')
            cfg=AxisConfig(**data['config']).validate(); draft=data['recipe']
            bounds=data.get('bounds',{}); batch=data.get('batch',{})
            if not isinstance(bounds,dict) or not isinstance(batch,dict) or not isinstance(draft,dict):
                raise ValueError('动作、边界或批量设置结构无效')
            for a in ('x','z'):
                pair=bounds.get(a,[-100,100])
                if not isinstance(pair,list) or len(pair)!=2: raise ValueError('边界设置无效')
                for value in pair: finite(value,'边界')
            if not isinstance(draft.get('name'),str) or not isinstance(draft.get('steps'),list) or not isinstance(draft.get('config',{}),dict):
                raise ValueError('动作函数草稿结构无效')
            steps=[Step(**raw) for raw in draft['steps']]
            for step in steps:
                if type(step.enabled) is not bool or not isinstance(step.label,str): raise ValueError('动作字段无效')
                if step.kind not in ('relative','absolute','zero','wait','restart_sensor','baseline','capture_start') or step.axis not in ('X','Z'): raise ValueError('动作类型无效')
                for value in (step.value,step.duration): finite(value,'动作数值')
                for value in (step.speed,step.accel):
                    if value is not None: finite(value,'动作速度/加速度')
            if type(batch.get('count',1)) is not int or not 1<=batch.get('count',1)<=10000: raise ValueError('批量次数无效')
            if not 0<=finite(batch.get('interval',0),'批量间隔')<=86400: raise ValueError('批量间隔无效')
            manual=data.get('manual_origin',{})
            if not isinstance(manual,dict) or not isinstance(manual.get('name','人工机械基准'),str): raise ValueError('人工基准设置无效')
            point=manual.get('point')
            if point is not None:
                if not isinstance(point,dict) or not isinstance(point.get('origin_name'),str): raise ValueError('采集起点设置无效')
                for key in ('x','z','clearance_z','x_ppmm','z_ppmm'): finite(point[key],'采集起点')
                AxisConfig(start_x=point['x'],start_z=point['z'],clearance_z=point['clearance_z']).validate()
            path=Path(data['output_root'])
            if not path.is_absolute(): raise ValueError('保存目录不是绝对路径')
            self.output_root=path; self.output_label.setText(str(path))
            for a in ('x','z'):
                self.speeds[a].setValue(getattr(cfg,a+'_speed')); self.accels[a].setValue(getattr(cfg,a+'_accel'))
                self.ppmm[a].setValue(getattr(cfg,a+'_ppmm'))
                for widget,value in zip(self.bounds[a],data.get('bounds',{}).get(a,[-100,100])): widget.setValue(value)
            self.right_sign.setCurrentIndex(self.right_sign.findData(cfg.x_right))
            self.bound_enabled.setChecked(False); self.keyboard_enabled.setChecked(False)
            self.distance.setValue(data.get('distance',1)); self.jog_mode.setCurrentIndex(data.get('jog_mode',0))
            self.origin_name.setText(manual.get('name','人工机械基准')); self.saved_start=point
            if point:
                self.start_x.setValue(point['x']); self.start_z.setValue(point['z']); self.start_clearance.setValue(point['clearance_z'])
            self.recipe_name.setText(draft['name']); self.recipe_config=draft.get('config',{}); self.replace_rows(steps)
            self.record_label.setText(data.get('label',self.catalog().last_label()))
            index=self.sensor_baud.findData(data.get('sensor_baud',115200))
            if index>=0: self.sensor_baud.setCurrentIndex(index)
            if hasattr(self,'batch_count'):
                self.batch_count.setValue(data.get('batch',{}).get('count',1)); self.batch_interval.setValue(data.get('batch',{}).get('interval',0))
            self._saved_session=self.session_data()
            self.update_next_name()
            note='已恢复上次工作台设置；请重新确认零点，软件边界需手动启用。'
            if not path.is_dir() or not os.access(path,os.W_OK): note+=' 保存目录不可用，请重新选择。'
            self.message(note,not path.is_dir()); self.poll()
        except (ValueError,TypeError,KeyError,OSError) as exc: self.message('工作台设置恢复失败：'+str(exc),True)

    def require_output(self):
        if not self.output_root.is_dir() or not os.access(self.output_root,os.W_OK):
            raise ValueError('保存目录不存在或不可写，请在采集设置中重新选择')

    def show_panel(self,panel):
        self.release_current(); panel.setModal(False); panel.show(); panel.raise_(); panel.activateWindow()

    def message(self,text,error=False):
        self.notice.setText(text); self.notice.setStyleSheet('color:#ff9098;' if error else 'color:#97b3c4;')
        self.log.appendPlainText(text)

    def submit(self,name,**payload): self.worker.submit(name,**payload)

    def refresh_ports(self):
        for combo in (self.motor_ports,self.sensor_ports):
            old=combo.currentData(); combo.clear(); combo.addItem('模拟设备 · 不连接硬件','DEMO')
            combo.setEditable(True)
            for p in list_ports.comports():
                if p.vid is not None or p.device.startswith(('/dev/ttyACM','/dev/ttyUSB','/dev/serial/')):
                    combo.addItem(f'{p.device} · {p.description}',p.device)
            if old is not None:
                index=combo.findData(old)
                if index>=0: combo.setCurrentIndex(index)
        self.message('可选择模拟设备，或输入 /dev/serial/by-id/… 完整路径。')

    @staticmethod
    def port_value(combo):
        return combo.currentData() if combo.currentIndex()>=0 and combo.currentText()==combo.itemText(combo.currentIndex()) else combo.currentText().strip()

    def toggle_motor(self):
        self.release_current()
        if self.state.get('motor_port'): self.submit('disconnect_motor')
        else: self.submit('connect_motor',port=self.port_value(self.motor_ports))

    def toggle_sensor(self):
        port=None if self.state.get('sensor_port') else self.port_value(self.sensor_ports)
        self.submit('connect_sensor',port=port,baud=self.sensor_baud.currentData())

    def config(self):
        values={}
        for a in ('x','z'):
            values[a+'_ppmm']=self.ppmm[a].value(); values[a+'_speed']=self.speeds[a].value(); values[a+'_accel']=self.accels[a].value()
            values[a+'_min']=self.bounds[a][0].value() if self.bound_enabled.isChecked() else None
            values[a+'_max']=self.bounds[a][1].value() if self.bound_enabled.isChecked() else None
        values['x_right']=self.right_sign.currentData()
        if self.start_matches():
            values.update(start_x=self.saved_start['x'],start_z=self.saved_start['z'],clearance_z=self.saved_start['clearance_z'])
        return AxisConfig(**values).validate()

    def sync_config(self):
        try:
            cfg=self.config()
            if self.state.get('config')!=cfg.to_dict(): self.submit('configure',config=cfg.to_dict())
            return cfg
        except ValueError as exc: self.message(str(exc),True); return None

    def start_matches(self):
        point=self.saved_start
        return bool(point and point['origin_name']==self.origin_name.text().strip()
                    and point['x_ppmm']==self.ppmm['x'].value() and point['z_ppmm']==self.ppmm['z'].value())

    def reference_matches(self):
        ref=self.state.get('manual_reference',{})
        cfg=self.state.get('config',{})
        return bool(ref.get('valid') and ref.get('name')==self.origin_name.text().strip()
                    and ref.get('x_ppmm',cfg.get('x_ppmm'))==self.ppmm['x'].value()
                    and ref.get('z_ppmm',cfg.get('z_ppmm'))==self.ppmm['z'].value())

    def begin_alignment(self):
        answer=QMessageBox.question(self,'临时对位',
            '此操作建立仅供手动对位的临时坐标，使原机械标定失效，不会自动移动。\n手动使能后可点动；自动流程需等重新对准固定基准并确认原点。是否继续？',QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if answer==QMessageBox.Yes and self.sync_config() is not None: self.submit('begin_alignment')

    def confirm_origin(self):
        name=self.origin_name.text().strip()
        if not name or len(name)>80: self.message('请输入 1–80 字的固定基准名称',True); return
        answer=QMessageBox.question(self,'确认人工机械原点',
            f'请确认机构已静止，并且已对准实体固定基准“{name}”。\n当前实体位置将定义为机械 X=0、Z=0。此操作不移动电机，程序无法检测你是否实际对准。\n是否确认？',QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if answer==QMessageBox.Yes and self.sync_config() is not None:
            self.submit('calibrate_origin',origin_name=name)

    def save_capture_start(self,use_current=False):
        try:
            if not self.reference_matches() or not self.state.get('idle') or self.state.get('recording'):
                raise ValueError('请先完成固定机械原点的人工标定，并结束运动及采集')
            cfg=self.config()
            x=self.state['status']['x']/cfg.x_ppmm if use_current else self.start_x.value()
            z=self.state['status']['z']/cfg.z_ppmm if use_current else self.start_z.value()
            safe=self.start_clearance.value()
            AxisConfig(start_x=x,start_z=z,clearance_z=safe).validate()
            cfg.check_position(cfg.pulses('x',x),cfg.pulses('z',z))
            cfg.check_position(cfg.pulses('x',x),cfg.pulses('z',safe))
            self.saved_start=dict(x=x,z=z,clearance_z=safe,x_ppmm=cfg.x_ppmm,z_ppmm=cfg.z_ppmm,origin_name=self.origin_name.text().strip())
            self.start_x.setValue(x); self.start_z.setValue(z)
            self.sync_config(); self.save_session(); self.poll()
            self.message(f'固定采集起点已保存：X={x:g}、Z={z:g} mm；安全 Z={safe:g} mm。')
        except (ValueError,KeyError) as exc: self.message(str(exc),True)

    def go_capture_start(self):
        if not self.reference_matches() or not self.start_matches():
            self.message('请先确认固定机械原点并保存匹配的采集起点',True); return
        if self.sync_config() is not None: self.submit('move',step=Step('capture_start'))

    def direction_pressed(self,direction,source):
        if self.held is not None: return
        cfg=self.sync_config()
        if cfg is None: return
        axis,sign=cfg.direction(direction)
        if source=='mouse' and self.jog_mode.currentIndex()==1:
            self.submit('move',step=Step('relative',axis,sign*self.distance.value())); return
        self.held=(direction,source)
        self.worker.lease(); self.submit('jog',axis=axis,direction=sign)

    def direction_released(self,direction,source):
        if self.held==(direction,source): self.release_current()

    def release_current(self,*args):
        if self.held is not None:
            self.held=None; self.submit('release')

    def owns_widget(self,widget):
        while isinstance(widget,QWidget):
            if widget is self: return True
            widget=widget.parentWidget()
        return False

    def keyboard_scope(self):
        return bool(self.keyboard_enabled.isChecked() and self.workspace_tabs.currentIndex()==0
                    and QApplication.activeWindow() is self)

    def renew_hold(self):
        if self.held is None: return
        direction,source=self.held
        if ((source=='key' and not self.keyboard_scope()) or not self.owns_widget(QApplication.activeWindow())
            or (source=='mouse' and not self.direction_buttons[direction].isDown())):
            self.release_current(); return
        self.worker.lease()

    def eventFilter(self,obj,event):
        kind=event.type()
        if kind==QEvent.ApplicationDeactivate or (kind==QEvent.WindowDeactivate and self.owns_widget(obj)):
            self.release_current()
        if kind in (QEvent.KeyPress,QEvent.KeyRelease):
            key=event.key(); mapping={Qt.Key_Left:'left',Qt.Key_Right:'right',Qt.Key_Up:'up',Qt.Key_Down:'down'}
            owned=self.owns_widget(obj) and self.owns_widget(QApplication.activeWindow())
            if key in mapping:
                direction=mapping[key]
                scope=owned and self.keyboard_scope()
                if event.isAutoRepeat(): return scope or bool(self.held and self.held[1]=='key')
                if kind==QEvent.KeyRelease and self.held==(direction,'key'):
                    self.release_current(); return True
                if scope:
                    if kind==QEvent.KeyPress and self.can_move(): self.direction_pressed(direction,'key')
                    # In enabled manual control, arrows belong to jogging even if the motor is not ready.
                    return True
            elif key==Qt.Key_Space and kind==QEvent.KeyPress and not event.isAutoRepeat() and owned:
                self.stop_all(); return True
        return super().eventFilter(obj,event)

    def can_move(self):
        s=self.state.get('status') or {}
        return bool(self.state.get('idle') and s.get('armed') and s.get('refx') and s.get('refz') and s.get('fault')=='NONE' and not self.state.get('calibrating'))

    def stop_all(self): self.held=None; self.submit('stop')

    def single_move(self,kind,axis,value):
        if self.sync_config() is not None: self.submit('move',step=Step(kind,axis,value))

    def add_step(self,step=None):
        if isinstance(step,bool) or step is None: step=Step()
        row=self.table.rowCount(); self.table.insertRow(row)
        enabled=QCheckBox(); enabled.setChecked(step.enabled); self.table.setCellWidget(row,0,enabled)
        kind=QComboBox()
        for text,code in [('相对移动','relative'),('绝对定位','absolute'),('回零','zero'),('到固定采集起点','capture_start'),('等待','wait'),('重启采集板','restart_sensor'),('采集空载基线','baseline')]: kind.addItem(text,code)
        kind.setCurrentIndex(max(0,kind.findData(step.kind))); self.table.setCellWidget(row,1,kind)
        axis=QComboBox(); axis.addItems(['X','Z']); axis.setCurrentText(step.axis); self.table.setCellWidget(row,2,axis)
        direction=QComboBox(); direction.addItem('正向',1); direction.addItem('负向',-1); direction.setCurrentIndex(0 if step.value>=0 else 1); self.table.setCellWidget(row,3,direction)
        value=spin(abs(step.value) if step.kind=='relative' else step.value); self.table.setCellWidget(row,4,value)
        speed=spin(step.speed or 0,0,100000); speed.setSpecialValueText('全局'); self.table.setCellWidget(row,5,speed)
        accel=spin(step.accel or 0,0,1000000); accel.setSpecialValueText('全局'); self.table.setCellWidget(row,6,accel)
        duration=spin(step.duration,0,86400); self.table.setCellWidget(row,7,duration)
        note=QLineEdit(step.label); self.table.setCellWidget(row,8,note)
        def update_fields():
            code=kind.currentData(); relative=code=='relative'; moving=code in ('relative','absolute','zero','capture_start')
            axis.setEnabled(moving and code!='capture_start'); direction.setEnabled(relative)
            value.setEnabled(code in ('relative','absolute')); value.setMinimum(0 if relative else -1e8)
            speed.setEnabled(moving); accel.setEnabled(moving); duration.setEnabled(not moving)
            duration.setSpecialValueText('默认 10 s' if code=='restart_sensor' else '默认 2 s' if code=='baseline' else '0')
            duration.setToolTip('重启：等待启动和新样本的超时；基线：静止空载采样时间；0 使用默认值。')
            right=self.right_sign.currentData()
            if axis.currentText()=='X': labels=('右 (+)','左 (−)') if right==1 else ('左 (+)','右 (−)')
            else: labels=('下 (+)','上 (−)')
            direction.setItemText(0,labels[0]); direction.setItemText(1,labels[1])
        kind.currentIndexChanged.connect(update_fields); axis.currentIndexChanged.connect(update_fields)
        self.right_sign.currentIndexChanged.connect(update_fields)
        update_fields(); self.table.selectRow(row)

    def read_recipe(self):
        steps=[]
        for row in range(self.table.rowCount()):
            w=lambda col:self.table.cellWidget(row,col)
            kind=w(1).currentData(); val=w(4).value()
            if kind=='relative': val=abs(val)*w(3).currentData()
            steps.append(Step(kind,w(2).currentText(),val,w(5).value() or None,w(6).value() or None,
                              w(7).value(),w(0).isChecked(),w(8).text()))
        return Recipe(self.recipe_name.text().strip() or '未命名流程',steps,self.recipe_config or self.config().to_dict())

    def replace_rows(self,steps):
        # Disconnect row-specific callbacks before Qt destroys their controls.
        if self.table.rowCount():
            try: self.right_sign.currentIndexChanged.disconnect()
            except RuntimeError: pass
        self.table.setRowCount(0)
        for step in steps: self.add_step(step)

    def new_recipe(self):
        self.recipe_name.setText('新动作函数'); self.recipe_config={}; self.replace_rows([]); self.preview.clear()

    def duplicate_row(self):
        row=self.table.currentRow()
        if row<0: return
        steps=self.read_recipe().steps; from copy import deepcopy
        steps.insert(row+1,deepcopy(steps[row])); self.replace_rows(steps); self.table.selectRow(row+1)

    def delete_row(self):
        row=self.table.currentRow()
        if row>=0:
            steps=self.read_recipe().steps; del steps[row]; self.replace_rows(steps)

    def move_row(self,delta):
        row=self.table.currentRow(); target=row+delta
        if row<0 or target<0 or target>=self.table.rowCount(): return
        steps=self.read_recipe().steps; steps[row],steps[target]=steps[target],steps[row]
        self.replace_rows(steps); self.table.selectRow(target)

    def adopt_config(self):
        try:
            self.recipe_config=self.config().to_dict(); self.message('流程已采用当前参数；请重新检查路径后运行。'); self.preview.clear()
        except ValueError as exc: self.message(str(exc),True)

    def check_recipe(self):
        try:
            recipe=self.read_recipe(); cfg=self.config()
            if recipe.config_changed(cfg): raise ValueError('加载流程的配置不同：核对后点击“采用当前配置”')
            status=self.state.get('status')
            if not self.reference_matches(): raise ValueError('请先人工标定固定机械原点')
            if status is None or not status.get('refx') or not status.get('refz'): raise ValueError('请先连接设备并设置两轴零点')
            steps=plan_steps(recipe,status,cfg); x,z=status['x'],status['z']; lines=[]
            for step in steps:
                if step.kind in ('restart_sensor','baseline'):
                    action='重启磁采集板，等待启动与新数据（超时）' if step.kind=='restart_sensor' else '静止空载基线采样'
                    lines.append(f'{step.index:02d}  {action} · {step.duration:g} s'); continue
                lines.append(f'{step.index:02d}  ({x/cfg.x_ppmm:g}, {z/cfg.z_ppmm:g}) → ({step.x/cfg.x_ppmm:g}, {step.z/cfg.z_ppmm:g}) mm  · {step.duration:.2f} s  · 目标脉冲 ({step.x}, {step.z})')
                x,z=step.x,step.z
            self.preview.setPlainText('\n'.join(lines)); self.message('路径参数检查通过；机械行程与防碰撞仍需现场核对。')
            return recipe
        except (ValueError,OverflowError) as exc: self.message(str(exc),True); return None

    def run_recipe(self,record):
        recipe=self.check_recipe()
        if recipe is None or self.sync_config() is None: return
        if record:
            try: self.require_output()
            except ValueError as exc: self.message(str(exc),True); return
        payload=dict(recipe=recipe,root=str(self.output_root))
        if record:
            try: payload['label']=valid_label(self.record_label.text())
            except ValueError as exc: self.message(str(exc),True); return
        self.submit('run_record' if record else 'run',**payload)

    def run_batch(self):
        recipe=self.check_recipe()
        if recipe is None: return
        try:
            self.require_output(); label=valid_label(self.record_label.text()); cfg=self.config()
            validate_batch(recipe,self.state.get('status') or {},cfg,self.batch_count.value(),self.batch_interval.value(),self.state.get('baseline') is not None)
            if self.sync_config() is None: return
            self.save_session()
            self.submit('batch_record',root=str(self.output_root),label=label,recipe=recipe,
                        count=self.batch_count.value(),interval=self.batch_interval.value())
        except (ValueError,KeyError) as exc: self.message(str(exc),True)

    def save_dialog(self):
        try:
            recipe=self.read_recipe(); recipe.config=self.config().to_dict()
            path,_=QFileDialog.getSaveFileName(self,'保存动作函数',str(Path.home()/(recipe.name+'.json')),'动作函数 (*.json)')
            if path:
                save_recipe(path,recipe); self.recipe_config=recipe.config; self.message('已保存：'+path)
        except (ValueError,OSError) as exc: self.message(str(exc),True)

    def load_dialog(self):
        path,_=QFileDialog.getOpenFileName(self,'加载动作函数',str(Path.home()),'动作函数 (*.json)')
        if not path: return
        try:
            recipe=load_recipe(path); self.recipe_name.setText(recipe.name); self.recipe_config=recipe.config
            self.replace_rows(recipe.steps); self.preview.clear()
            self.message('已加载；配置不同，请核对后采用当前配置。' if recipe.config_changed(self.config()) else '已加载；执行前请检查当前位置和路径。')
        except (ValueError,OSError) as exc: self.message(str(exc),True)

    def choose_output(self):
        path=QFileDialog.getExistingDirectory(self,'选择磁数据保存目录',str(self.output_root.parent))
        if path:
            self.output_root=Path(path); self.output_label.setText(path); self.update_next_name(); self.refresh_records()

    def start_record(self):
        try:
            self.require_output(); label=valid_label(self.record_label.text())
        except ValueError as exc: self.message(str(exc),True); return
        if self.sync_config() is not None: self.submit('record_start',root=str(self.output_root),label=label)

    def catalog(self): return Catalog(self.output_root,getattr(self.worker,'settings_path',None))

    def update_next_name(self,*args):
        try: self.next_record.setText('下次保存：'+self.catalog().next_name(self.record_label.text())+'.csv')
        except (ValueError,OSError) as exc: self.next_record.setText(str(exc))

    def show_records(self):
        self.refresh_records(); self.show_panel(self.records_dialog)

    def refresh_records(self):
        rows=self.catalog().records(); self.records_table.setRowCount(0)
        outcomes={'completed':'完成','aborted':'已中止','interrupted':'未正常结束 / 保存中'}
        for row,record in enumerate(rows):
            self.records_table.insertRow(row)
            for col,value in enumerate((record['name'],record['label'],record['created'],outcomes.get(record['outcome'],record['outcome']),record['samples'])):
                item=QTableWidgetItem(str(value)); item.setData(Qt.UserRole,record['path']); self.records_table.setItem(row,col,item)
        self.rename_button.setEnabled(False); self.delete_button.setEnabled(False); self.update_next_name()
        self.trash_table.setRowCount(0)
        for row,record in enumerate(self.catalog().trash_records()):
            self.trash_table.insertRow(row)
            for col,value in enumerate((record['name'],record['deleted'])):
                item=QTableWidgetItem(value); item.setData(Qt.UserRole,record['path']); self.trash_table.setItem(row,col,item)

    def selected_record_changed(self):
        row=self.records_table.currentRow(); item=self.records_table.item(row,1) if row>=0 else None
        count=len(self.records_table.selectionModel().selectedRows())
        self.rename_button.setEnabled(item is not None and count==1); self.delete_button.setEnabled(count>0)
        if item: self.rename_label.setText('' if item.text()=='未标注' else item.text())

    def rename_record(self):
        row=self.records_table.currentRow(); item=self.records_table.item(row,0) if row>=0 else None
        if not item: return
        try: label=valid_label(self.rename_label.text())
        except ValueError as exc: self.message(str(exc),True); return
        self.records_status.setText('正在重命名…')
        self.submit('rename_record',root=str(self.output_root),path=item.data(Qt.UserRole),label=label)

    def delete_records(self):
        rows=self.records_table.selectionModel().selectedRows()
        items=[self.records_table.item(row.row(),0) for row in rows]
        if not items: return
        names='\n'.join(item.text() for item in items)
        answer=QMessageBox.question(self,'删除记录',f'将以下 {len(items)} 条记录移入回收站并释放编号：\n\n{names}\n\n可在回收站恢复。',QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if answer!=QMessageBox.Yes: return
        self.records_status.setText('正在移入回收站…')
        self.submit('trash_records',root=str(self.output_root),paths=[item.data(Qt.UserRole) for item in items])

    def restore_record(self):
        row=self.trash_table.currentRow(); item=self.trash_table.item(row,0) if row>=0 else None
        if not item: return
        self.submit('restore_record',root=str(self.output_root),path=item.data(Qt.UserRole))

    def _toggle_realtime(self,checked):
        enabled=self.state.get('realtime_forward',{}).get('enabled',False)
        self.realtime_start.setEnabled(bool(checked and not enabled))
        if not checked: self.submit('realtime_forward',enabled=False)

    def poll(self):
        for _ in range(100):
            try: event=self.worker.events.get_nowait()
            except queue.Empty: break
            kind=event['type']
            if kind=='error':
                self.release_current(); self.message(event['message'],True)
                if self.records_dialog.isVisible(): self.records_status.setText(event['message'])
            elif kind=='log': self.message(event['message'])
            elif kind=='saved':
                self.last_saved=event['path']; self.message('采集已保存：'+event['path']); self.update_next_name()
                if self.records_dialog.isVisible(): self.refresh_records()
            elif kind=='label':
                self.record_label.setText(event['label']); self.update_next_name()
            elif kind=='renamed':
                self.last_saved=event['path']; self.record_label.setText(event['label']); self.refresh_records(); self.message('记录已重命名：'+event['path']); self.records_status.setText('已重命名：'+event['path'])
            elif kind in ('trashed','restored'):
                self.refresh_records(); self.last_saved=None
                text='已移入回收站，原编号现可复用。' if kind=='trashed' else '已恢复：'+event['path']
                self.records_status.setText(text); self.message(text)
            elif kind=='batch_done':
                self.message(f'批量结束：已完成 {event["completed"]}/{event["count"]} 轮。'+event.get('reason',''),event['phase']=='aborted')
            elif kind=='stopped': self.held=None; self.message(event.get('reason','已停止'),True)
            elif kind=='step': self.progress.setText(f'正在执行第 {event["number"]}/{event["count"]} 个启用动作 · 表格第 {event["index"]} 行')
            elif kind=='complete': self.progress.setText('执行完成 · 保持最后一步的位置')
            elif kind=='start_cleared':
                self.saved_start=None; self.preview.clear()
            elif kind=='origin_changed':
                self.bound_enabled.setChecked(False); self.preview.clear(); self.message('软件零点已请求更新，原软件边界已清空，请重新检查流程。')
        self.state=self.worker.snapshot()
        if not self.state: return
        state=self.state; s=state.get('status') or {}; idle=state.get('idle',False); rec=state.get('recording',False)
        batch=state.get('batch',{}); batching=batch.get('active',False)
        owner=state.get('owner'); calibrating=state.get('calibrating',False); ready=state.get('ready',False)
        fixed=self.reference_matches()
        free=idle and not calibrating; can=self.can_move(); editing=not owner and not state.get('pending') and not calibrating
        cfg=state.get('config',{})
        if s:
            self.position.setText(f'X  {s["x"]/cfg.get("x_ppmm",250):.3f} mm       Z  {s["z"]/cfg.get("z_ppmm",250):.3f} mm')
            self.reference.setText(f'零点 X {"有效" if s.get("refx") else "未建立/失效"} · Z {"有效" if s.get("refz") else "未建立/失效"}  |  {"已使能" if s.get("armed") else "未使能"}  |  故障 {s.get("fault")}')
        else: self.position.setText('X  — mm       Z  — mm'); self.reference.setText('控制器未连接或位置已失效')
        self.connection_status.setText(('模拟演示 · 非实测' if state.get('demo') else '真实设备')+' · '+('控制器已连接' if ready else '控制器未就绪'))
        self.motor_connect.setText('断开电机' if state.get('motor_port') else '连接电机')
        self.sensor_connect.setText('断开传感器' if state.get('sensor_port') else '连接传感器')
        self.motor_connect.setEnabled(not owner and not rec and not calibrating)
        self.sensor_connect.setEnabled(not owner and not rec and not calibrating)
        self.motor_ports.setEnabled(not state.get('motor_port')); self.sensor_ports.setEnabled(not state.get('sensor_port'))
        self.sensor_baud.setEnabled(not state.get('sensor_port'))
        self.arm_button.setEnabled(bool(free and s.get('refx') and s.get('refz') and not s.get('armed') and s.get('fault')=='NONE'))
        self.clear_button.setEnabled(bool(idle and not rec and s.get('fault')!='NONE'))
        for w in (self.zero_x,self.zero_z,self.zero_all): w.setEnabled(bool(free and not rec and s.get('fault')=='NONE'))
        for d,w in self.direction_buttons.items(): w.setEnabled(can or self.held==(d,'mouse'))
        self.reference.setText(self.reference.text()+(' · 人工机械基准已确认' if fixed else ' · 固定机械基准待人工确认'))
        self.origin_status.setText(('已确认：'+self.origin_name.text() if fixed else '机械标定无效：请人工对准固定基准后确认。')+'\n临时对位只允许手动移动；重连或脉冲换算改变后需要重新标定。')
        self.point_summary.setText(f'起点 X={self.saved_start["x"]:g}、Z={self.saved_start["z"]:g} mm · 安全Z={self.saved_start["clearance_z"]:g}' if self.start_matches() else '未保存匹配的固定采集起点（检查基准名与脉冲换算）')
        for w in (self.align_button,self.confirm_origin_button): w.setEnabled(bool(free and not rec))
        for w in (self.save_start_current,self.save_start_input): w.setEnabled(bool(free and not rec and fixed))
        for w in (self.origin_name,self.start_x,self.start_z,self.start_clearance): w.setEnabled(editing and not rec)
        self.go_start_button.setEnabled(bool(can and fixed and self.start_matches()))
        for w in (self.return_x,self.return_z,self.absolute_button): w.setEnabled(can and fixed)
        self.jog_mode.setEnabled(not owner); self.distance.setEnabled(not owner)
        for w in self.config_widgets+[self.apply_config]: w.setEnabled(editing and not rec)
        for w in self.edit_buttons+[self.new_button,self.load_button,self.save_button,self.adopt_button,self.recipe_name,self.table]: w.setEnabled(editing)
        self.check_button.setEnabled(editing); self.run_button.setEnabled(can and fixed)
        self.run_record_button.setEnabled(bool(can and fixed and not rec and (state.get('sensor_port') or state.get('sensor_alive'))))
        self.batch_button.setEnabled(bool(can and fixed and not rec and not batching and state.get('sensor_alive')))
        self.batch_count.setEnabled(not batching); self.batch_interval.setEnabled(not batching)
        self.batch_stop_button.setEnabled(batching and not batch.get('stop_after',False))
        if batch:
            phases={'running':'运行采集中','saving':'等待保存完成','interval':'等待下一轮','completed':'已结束','aborted':'已中止'}
            self.batch_progress.setText(f'第 {batch.get("round",0)} / {batch.get("count",0)} 轮 · 已完成 {batch.get("completed",0)} 轮 · '+phases.get(batch.get('phase'),'准备中')+(' · 本轮结束后停止' if batch.get('stop_after') else ''))
        self.abort_button.setEnabled(bool(owner or rec or batching)); self.stop_button.setEnabled(True)
        self.baseline_button.setEnabled(bool(free and not rec and state.get('sensor_alive')))
        self.record_start.setEnabled(bool(free and not rec and state.get('baseline') is not None and state.get('sensor_alive')))
        self.record_stop.setEnabled(bool(rec and not state.get('record_owned'))); self.output_button.setEnabled(not rec and not batching); self.record_label.setEnabled(not rec and not batching)
        values=state.get('sensor_values')
        if values is not None and state.get('baseline') is not None: self.values.setText('   '.join(f'{a} {v:.3f}' for a,v in zip(('ΔBx','ΔBy','ΔBz'),values))+' µT')
        else: self.values.setText('ΔBx —    ΔBy —    ΔBz — µT · 待校准')
        forwarding=state.get('realtime_forward',{})
        forwarding_on=bool(forwarding.get('enabled'))
        self.realtime_start.setEnabled(self.realtime_enabled.isChecked() and not forwarding_on)
        self.realtime_stop.setEnabled(forwarding_on)
        self.realtime_status.setText(('转发中 · 客户端 '+str(forwarding.get('clients',0))+' · 丢弃 '+str(forwarding.get('dropped',0)))
            if forwarding_on else ('转发失败：'+forwarding['error'] if forwarding.get('error') else '默认关闭 · 127.0.0.1:8766'))
        baseline=state.get('baseline')
        self.baseline_status.setText('正在校准，请保持空载静止…' if calibrating else
            ('基线 XYZ / µT：'+', '.join(f'{v:.3f}' for v in baseline) if baseline is not None else '未校准 · 空载静止后采集 2 秒基线'))
        self.plot.history=state.get('history',[]); self.plot.baseline=baseline; self.plot.update()
        self.record_status.setText(('等待采集板重启 / 新基线 · ' if state.get('restarting') or (rec and calibrating) else '')+f'正在采集 · {state.get("stage")} · 已写入 {state.get("samples",0)} 样本 · 解析异常 {state.get("bad_lines",0)}' if rec else ('最近保存：'+self.last_saved if self.last_saved else '未录制'))

    def closeEvent(self,event):
        self.save_session(); self.session_timer.stop()
        self.release_current(); self.timer.stop(); self.hold_timer.stop()
        QApplication.instance().removeEventFilter(self)
        self.worker.shutdown(); event.accept()


def main(demo=False):
    app=QApplication.instance() or QApplication([]); app.setStyle('Fusion')
    app.setFont(QFont('Noto Sans CJK SC',10))
    window=Window(); window.show()
    if demo:
        window.submit('connect_motor',port='DEMO'); window.submit('connect_sensor',port='DEMO',baud=115200)
    return app.exec()
