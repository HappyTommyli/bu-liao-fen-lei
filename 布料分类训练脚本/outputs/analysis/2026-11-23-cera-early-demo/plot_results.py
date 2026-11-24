import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
font_name=font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc").get_name()
from cloth_trainer.online import csv_packets
out=Path(__file__).resolve().parent;data=json.loads((out/'replay.json').read_text())
plt.rcParams.update({'font.family':font_name,'axes.unicode_minus':False,'font.size':11,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
items=list(data['models'].items());colors=['#2764aa','#d27920']
fig,axs=plt.subplots(1,2,figsize=(12,4.4),gridspec_kw={'width_ratios':[1,1.3]})
for i,(key,m) in enumerate(items):
 s=m['summary'];n=len(m['records']);f=s['first_confirmation_counts'];left=0
 for name,color in [('correct','#38956c'),('error','#d95d57'),('abstain','#b1bac6')]:
  v=f[name];axs[0].barh(i,v,left=left,color=color,height=.5,label={'correct':'首次确认正确','error':'首次确认错误','abstain':'未确认'}[name] if i==0 else None)
  if v:axs[0].text(left+v/2,i,str(v),ha='center',va='center',color='white' if name!='abstain' else '#263647',weight='bold')
  left+=v
 t=[r['first_from_crop_s'] for r in m['records'] if r['first_confirmation']]
 axs[1].hist(t,bins=np.arange(0,18,1.35),histtype='step',linewidth=2.3,color=colors[i],label=('A 不重叠' if i==0 else 'B 33%重叠')+f" · 中位数{s['median_first_from_crop_s']:.2f}秒")
axs[0].set_yticks(range(len(items)),['A 不重叠训练','B 33%重叠训练']);axs[0].invert_yaxis();axs[0].set_xlim(0,154);axs[0].set_xlabel('去程数量（每组154次）');axs[0].set_title('首次确认结果，含未确认样本');axs[0].legend(loc='lower left',bbox_to_anchor=(-.05,-.35),ncol=3,fontsize=9)
axs[1].axvline(items[0][1]['summary']['median_crop_duration_s'],color='#7c8792',ls='--',label='有效扫描时长中位数');axs[1].set_xlabel('进入有效范围后的首次确认时间 / 秒');axs[1].set_ylabel('次数');axs[1].set_title('确认时间分布（只统计已确认）');axs[1].legend(fontsize=9)
fig.suptitle('最新CERA：默认在线门槛的逐帧回放',fontsize=16,weight='bold');fig.text(.5,.015,'验证子集；非独立测试成绩。窗口900点，预测步长900点，概率≥0.95，前两类差≥0.20，稳定2.7秒。',ha='center',fontsize=9,color='#526171');fig.subplots_adjust(bottom=.27,top=.82,wspace=.35);fig.savefig(out/'overview.png',dpi=170);plt.close(fig)
b=items[-1][1];good=sorted([r for r in b['records'] if r['early_confirmed'] and r['first_correct'] and r['status']=='correct'],key=lambda r:r['first_from_crop_s'])
chosen=[good[len(good)//2]] if good else []
other=[r for r in b['records'] if r['first_confirmation'] and not r['first_correct']] or [r for r in b['records'] if not r['first_confirmation']]
if other:chosen.append(other[0])
elif good:chosen.append(good[-1])
fig,axs=plt.subplots(len(chosen),2,figsize=(12,4*len(chosen)),squeeze=False)
example_rows=[]
for ri,example in enumerate(chosen):
 path=example['path'];packets=[p for p in csv_packets(path) if p.get('type')=='sample' and p.get('stage')=='step_003' and 51<=p['x_mm']<=140];start=packets[0]['t_s']
 for ci,(key,m) in enumerate(items):
  ax=axs[ri,ci];record=next(r for r in m['records'] if r['path']==path);events=[e for e in m['events'] if e.get('type')=='prediction' and e['path']==path]
  t=[e['t_s']-start for e in events];p=[e['probability'] for e in events]
  ax.plot(t,p,'o-',color=colors[ci],lw=2,label='当前最高类别概率');ax.axhline(.95,color='#8b929a',ls=':',label='概率门槛0.95');ax.axvline(record['crop_duration_s'],color='#616c77',ls='--',label='有效扫描结束')
  first=record['first_confirmation']
  if first:
   fc=record['first_from_crop_s'];color='#23895c' if record['first_correct'] else '#cc3a36';ax.axvline(fc,color=color,lw=2);ax.axvspan(fc,record['crop_duration_s'],color=color,alpha=.07)
   text=f"首次确认：{first['label']}\n{fc:.2f}秒 / X={first['x_mm']:.1f}mm\n比有效扫描结束提前{record['lead_before_crop_end_s']:.2f}秒"
  else:text='本次始终未确认'
  text+='\n结束状态：'+{'correct':'正确确认','error':'错误确认','abstain':'未确认'}[record['status']]
  ax.text(.04,.08,text,transform=ax.transAxes,fontsize=10,bbox=dict(facecolor='white',alpha=.9,edgecolor='#dce2e8',boxstyle='round,pad=.5'))
  ax.set_ylim(0,1.04);ax.set_xlim(0,record['crop_duration_s']+.4);ax.set_xlabel('进入51mm有效范围后的时间 / 秒');ax.set_ylabel('最高类别概率');ax.set_title(('A 不重叠' if ci==0 else 'B 33%重叠')+f" ｜真实：{record['true_label']} · {Path(path).stem.rsplit('_',1)[-1]}次")
  if ri==0 and ci==0:ax.legend(loc='lower right',fontsize=8)
  example_rows.append(dict(model=key,**record))
fig.suptitle('扫描尚未结束，是否已经确认类别？',fontsize=16,weight='bold');fig.text(.5,.015,'连线仅帮助阅读离散窗口输出；确认同时要求概率差与持续稳定，单次高概率不等于已经确认。',ha='center',fontsize=9,color='#526171');fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(out/'examples.png',dpi=170);plt.close(fig)
(out/'examples.json').write_text(json.dumps(example_rows,ensure_ascii=False,indent=2))
print(json.dumps([dict(model=r['model'],label=r['true_label'],first=r['first_confirmation'],first_from_crop_s=r['first_from_crop_s'],lead_s=r['lead_before_crop_end_s'],status=r['status'],path=r['path']) for r in example_rows],ensure_ascii=False))
