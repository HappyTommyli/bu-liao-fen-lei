import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from matplotlib.transforms import Bbox
D=Path(__file__).parent
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font);plt.rcParams['font.family']=[font_manager.FontProperties(fname=font).get_name()];plt.rcParams['axes.unicode_minus']=False
records=json.loads((D/'新边界未覆盖_原始坐标核对.json').read_text());selected=[next(r for r in records if r['label']==name) for name in ['亮丝粗花呢','复合灯芯绒','覆膜帆布']]
fig,axes=plt.subplots(3,2,figsize=(14,11));evidence=[];waves=[]
for i,r in enumerate(selected):
 p=Path(r['path']);m=json.loads((p.parent/'metadata.json').read_text());raw=[]
 with p.open(encoding='utf-8-sig',newline='') as f:
  for row in csv.DictReader(f):
   if row['stage']=='step_003':raw.append([float(row['sensor_t_s']),float(row['motor_x_pulses']),float(row['dBy_uT'])])
 a=np.array(raw);t=a[:,0]-a[0,0];x=a[:,1]/r['ppmm'];by=np.abs(a[:,2]);waves.append((r,t,x,by));upper=float(x.max());origin=float(x[0]);end=float(x[-1]);n_in=int(((x>=51)&(x<=140)).sum());n_tail=int(((x>upper)&(x<=140)).sum())
 ev=dict(record=r['record'],path=r['path'],raw_outgoing_points=len(a),pulse_min=float(a[:,1].min()),pulse_max=float(a[:,1].max()),ppmm=r['ppmm'],x_start=origin,x_end=end,x_max=upper,travel=end-origin,
 required_absolute=[51,140],right_missing_mm=140-upper,points_above_130_652=int((x>130.652+1e-7).sum()),metadata_start=m['context'].get('capture_start_mm'),relative_51_140_absolute=[origin+51,origin+140])
 evidence.append(ev)
 ax=axes[i,0];ax.axhspan(upper,140,color='#d64a45',alpha=.13);ax.plot(t,x,color='#2464a1',lw=1.6);ax.axhline(51,color='#2f8b66',ls='--',lw=1.2);ax.axhline(140,color='#bc332f',ls='--',lw=1.4);ax.plot(t[-1],end,'o',color='#2464a1',ms=5)
 ax.annotate(f'终点 {end:.3f} mm\n最大脉冲数 {a[:,1].max():.0f}',xy=(t[-1],end),xytext=(t[-1]*.45,75),arrowprops={'arrowstyle':'->','color':'#315574'},fontsize=10,color='#27496b');ax.text(t[-1]*.02,144,'要求上界：140 mm',color='#bc332f',fontsize=10);ax.text(t[-1]*.02,54,'要求下界：51 mm',color='#2f8b66',fontsize=9)
 ax.set_title(r['record']+'｜去程原始电机轨迹',loc='left',fontsize=12);ax.set_xlabel('去程开始后的传感器时间 / s');ax.set_ylabel('CSV中的绝对X / mm');ax.set_ylim(-32,155);ax.grid(alpha=.2)
 ax=axes[i,1];ax.axvspan(51,140,color='#349674',alpha=.07);ax.axvspan(upper,140,color='#df4f46',alpha=.2);ax.axvline(51,color='#2f8b66',ls='--',lw=1.1);ax.axvline(140,color='#bc332f',ls='--',lw=1.1);ax.axvline(upper,color='#315b91',ls=':',lw=1.2);ax.plot(x,by,color='#2464a1',lw=.6)
 ax.set_xlim(-32,150);ylim=max(np.quantile(by,.999)*1.12,1);ax.set_ylim(0,max(ylim,float(by.max())*1.03));ax.text(52,ax.get_ylim()[1]*.94,'要求的接触区间',color='#2f8b66',fontsize=10);ax.annotate('这9.348 mm内\n没有任何去程采样',xy=((upper+140)/2,ax.get_ylim()[1]*.45),xytext=(70,ax.get_ylim()[1]*.65),arrowprops={'arrowstyle':'->','color':'#ad332b'},color='#ad332b',fontsize=10)
 ax.set_title('去程全部原始 |dBy|，包含标记无效点',loc='left',fontsize=12);ax.set_xlabel('CSV中的绝对X / mm');ax.set_ylabel('|dBy| / µT');ax.grid(alpha=.2)
fig.suptitle('核实新边界51～140 mm：三个不同布料均止于绝对X=130.652 mm',fontsize=16,y=.995)
fig.tight_layout(rect=[0,.015,1,.975],h_pad=2.3,w_pad=3)
fig.savefig(D/'51至140毫米_三个样本原始证据.png',dpi=155)
fig.canvas.draw();renderer=fig.canvas.get_renderer()
for i,ev in enumerate(evidence):
 box=Bbox.union([ax.get_tightbbox(renderer) for ax in axes[i]]).transformed(fig.dpi_scale_trans.inverted()).expanded(1.02,1.08)
 fig.savefig(D/('51至140_证据_'+ev['record']+'.png'),dpi=155,bbox_inches=box)
plt.close(fig)
# Same signal in the two coordinate systems. No waveform or position correction.
r,t,x,by=waves[0];fig,axs=plt.subplots(1,2,figsize=(12,4))
for ax,xx,title in zip(axs,[x,x-x[0]],['绝对X：固定X模式当前实际使用','从本次起点移动的距离：另一种含义']):
 ax.plot(xx,by,color='#2464a1',lw=.6);ax.axvspan(51,140,color='#349674',alpha=.14)
 for b in (51,140):ax.axvline(b,color='#2f8b66',ls='--')
 ax.set_xlim(-30,160);ax.set_xlabel('距离 / mm');ax.set_ylabel('|dBy| / µT');ax.set_title(title,fontsize=12);ax.grid(alpha=.2)
axs[0].axvspan(x.max(),140,color='#df4f46',alpha=.25);axs[0].text(0,.94,'坐标范围 −24.348～130.652 mm',transform=axs[0].transAxes,fontsize=10)
axs[1].text(0,.94,'移动范围 0～155 mm，覆盖51～140 mm',transform=axs[1].transAxes,fontsize=10)
fig.suptitle(r['record']+'：是否“未覆盖”取决于51～140指哪一种坐标',fontsize=14);fig.tight_layout(rect=[0,0,1,.9]);fig.savefig(D/'绝对坐标与相对行程_对照.png',dpi=155);plt.close(fig)
(D/'三个样本坐标证据.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2));print(json.dumps(evidence,ensure_ascii=False,indent=2))
