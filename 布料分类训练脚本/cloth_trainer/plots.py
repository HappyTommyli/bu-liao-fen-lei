"""Small scientific plots shared by data, evaluation and replay pages."""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('QtAgg')
from matplotlib import font_manager,rcParams
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PyQt5 import QtWidgets as W
from .data import prepare_trials
from .config import input_label

font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font));rcParams['font.family']=[font_manager.FontProperties(fname=str(font)).get_name(),'DejaVu Sans']
rcParams['axes.unicode_minus']=False
BG='#141d2b';FG='#c5d1e3';COLORS=['#82afff','#66d9c4','#e7b777','#c19bf5']

def deferred_plot(method):
    def render(self,*args,**kwargs):
        if getattr(self,'defer_when_hidden',False) and not self.isVisible():
            self._pending_render=(method,args,kwargs);return
        self._pending_render=None
        return method(self,*args,**kwargs)
    return render


class Plot(W.QWidget):
    def __init__(self,parent=None):
        super().__init__(parent);self.figure=Figure(figsize=(6,3),facecolor=BG,layout='constrained');self.canvas=FigureCanvasQTAgg(self.figure)
        layout=W.QVBoxLayout(self);layout.setContentsMargins(0,0,0,0);layout.addWidget(self.canvas);self.setMinimumHeight(170);self.empty('等待数据')
    def flush_pending(self):
        pending=getattr(self,'_pending_render',None);self._pending_render=None
        if pending:
            method,args,kwargs=pending;method(self,*args,**kwargs)
    def showEvent(self,event):
        super().showEvent(event);self.flush_pending()
    def axes(self,n=1):
        self.figure.clear();axes=np.atleast_1d(self.figure.subplots(n,1))
        for ax in axes:
            ax.set_facecolor(BG);ax.tick_params(colors=FG,labelsize=9);ax.xaxis.label.set_color(FG);ax.yaxis.label.set_color(FG);ax.title.set_color(FG)
            for spine in ax.spines.values():spine.set_color('#33445b')
            ax.grid(alpha=.12,color=FG)
        return axes
    def finish(self):self.canvas.draw_idle()
    @deferred_plot
    def empty(self,message):
        if hasattr(self,'training_history'):self.training_history=[]
        ax=self.axes()[0];ax.text(.5,.5,message,ha='center',va='center',transform=ax.transAxes,color=FG);ax.set_axis_off();self.finish()
    def curve(self,record,signal='by',processed=False):
        axes=self.axes(2);trials=prepare_trials([record],signal)
        for ax,raw,t in zip(axes,record.trials,trials):
            if processed:
                x=t.times-t.times[0];values=t.values;xlabel='接触段时间 / s'
            else:
                x=raw.raw_x;values=raw.raw_values[:,record.axes.index('y'):record.axes.index('y')+1] if signal=='by' else raw.raw_values[:,record.axes.index('x'):record.axes.index('x')+1] if signal=='bx' else raw.raw_values
                if signal=='xy':values=np.linalg.norm(raw.raw_values,axis=1,keepdims=True)
                ax.axvspan(*raw.bounds,color=COLORS[1],alpha=.12);xlabel='X位置 / mm'
            stride=max(1,len(x)//2500)
            for j in range(values.shape[1]):ax.plot(x[::stride],values[::stride,j],color=COLORS[j],lw=.8,label=(input_label(signal,record.diagnostics.get('preprocess_version','legacy-v2')) if processed and signal in ('by','bx','xy') else ('|ΔBx|' if j==0 else '|ΔBy|') if processed and record.diagnostics.get('preprocess_version')=='abs-v3' else 'ΔBy' if signal=='by' else 'ΔBx' if signal=='bx' else 'XY模长' if signal=='xy' else ['ΔBx','ΔBy'][j]))
            ax.set_title('去程' if t.direction=='out' else '回程',loc='left',fontsize=10,color=FG);ax.set_xlabel(xlabel);ax.set_ylabel('磁响应 / µT')
            ax.legend(facecolor=BG,labelcolor=FG,fontsize=8)
        self.finish()
    def set_training_axis(self,axis):
        if axis not in ('epoch','updates'):raise ValueError('训练曲线横轴无效')
        self.training_x_axis=axis
        if hasattr(self,'training_history'):self.training(self.training_history,axis)

    @deferred_plot
    def training(self,history,x_axis=None):
        self.training_history=history
        self.training_x_axis=x_axis or getattr(self,'training_x_axis','epoch')
        use_updates=self.training_x_axis=='updates' and bool(history) and all('optimizer_updates' in row for row in history)
        xlabel='累计优化器更新次数' if use_updates else '训练轮数（旧记录无更新数）' if self.training_x_axis=='updates' and history else '训练轮数'
        axes=self.axes(2)
        if history:
            x=[r['optimizer_updates'] if use_updates else r['epoch'] for r in history]
            axes[0].plot(x,[r['loss'] for r in history],color=COLORS[0],marker='.',label='训练窗口CE' if 'loss_kind' in history[0] else '训练总损失');axes[0].plot(x,[r['val_loss'] for r in history],color=COLORS[2],marker='.',label='验证单程NLL' if 'loss_kind' in history[0] else '验证总损失')
            axes[1].plot(x,[r.get('val_f1',r.get('val_prefix_f1')) for r in history],color=COLORS[1],marker='.',label='验证整段F1' if 'val_f1' in history[0] else '旧版验证前缀F1');axes[1].set_ylim(0,1)
            for ax in axes:ax.legend(facecolor=BG,labelcolor=FG,fontsize=8);ax.set_xlabel(xlabel)
        self.finish()
    @deferred_plot
    def prefix(self,result):
        ax=self.axes()[0];r=result['prefix_metrics'];x=[v['checkpoint']*100 for v in r]
        ax.plot(x,[v['accuracy'] for v in r],marker='o',color=COLORS[0],label='完整测试前缀准确率')
        ax.plot(x,[v['macro_f1'] for v in r],marker='o',color=COLORS[1],label='Macro-F1');ax.set_ylim(0,1.05);ax.set_xlabel('扫描进度 / %');ax.legend(facecolor=BG,labelcolor=FG,fontsize=9);self.finish()
    @deferred_plot
    def confusion(self,matrix,labels):
        self.setMinimumHeight(max(300,100+24*len(labels)))
        ax=self.axes()[0];a=np.array(matrix);ax.grid(False);ax.imshow(a,cmap='Blues')
        ax.set_xticks(range(len(labels)),labels,rotation=55,ha='right',fontsize=8);ax.set_yticks(range(len(labels)),labels)
        ax.set_xlabel('预测布料');ax.set_ylabel('真实布料')
        for (i,j),v in np.ndenumerate(a):ax.text(j,i,str(v),ha='center',va='center',fontsize=9,color='white' if v>a.max()/2 else '#17283e')
        self.finish()
    @deferred_plot
    def classes(self,metrics):
        self.setMinimumHeight(max(300,100+24*len(metrics['per_class'])))
        ax=self.axes()[0];rows=metrics['per_class'];ax.barh([r['label'] for r in rows],[r['recall'] for r in rows],color=COLORS[0]);ax.set_xlim(0,1.05);ax.set_xlabel('召回率');self.finish()
    @deferred_plot
    def trial(self,row,labels):
        if 'checkpoints' not in row:
            ax=self.axes()[0];ax.barh(labels,row['scores'],color=COLORS[0]);ax.set_xlim(0,1.05);ax.set_xlabel('完整接触段类别得分');self.finish();return
        axes=self.axes(2);cps=row['checkpoints'];x=[p['checkpoint']*100 for p in cps]
        for k,c in enumerate(labels):axes[0].plot(x,[p['scores'][k] for p in cps],label=c,lw=1)
        axes[0].set_ylim(0,1.05);axes[0].set_ylabel('类别得分');axes[0].legend(facecolor=BG,labelcolor=FG,fontsize=7,ncol=min(4,len(labels)))
        axes[1].plot(x,[p['risk'] for p in cps],color=COLORS[2],label='原始风险分数')
        axes[1].plot(x,[float(np.mean(p['evidence'])) for p in cps],color=COLORS[1],label='证据状态均值')
        axes[1].set_xlabel('扫描进度 / %');axes[1].legend(facecolor=BG,labelcolor=FG,fontsize=8);self.finish()
