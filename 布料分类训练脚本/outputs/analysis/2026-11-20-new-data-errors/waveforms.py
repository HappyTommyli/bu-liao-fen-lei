import csv,json,sys,collections,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from cloth_trainer.data import file_hash
DEST=Path(__file__).parent
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font);plt.rcParams['font.family']=[font_manager.FontProperties(fname=font).get_name(),'DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False

def compact(a,count=170):
 if len(a)<=count*2:ix=np.arange(len(a))
 else:
  ix={0,len(a)-1}
  for part in np.array_split(np.arange(len(a)),count):
   by=a[part,2];ix.update([int(part[np.nanargmin(by)]),int(part[np.nanargmax(by)])])
  ix=np.array(sorted(ix))
 return np.round(a[ix],5).tolist()

def parse(row):
 p=Path(row['path']);meta=json.loads((p.parent/'metadata.json').read_text());ppmm=meta.get('context',{}).get('config',{}).get('x_ppmm',250);base=meta.get('baseline_B0_uT',{}).get('By_uT',0)
 stages={'step_003':[],'step_004':[]}
 with p.open(encoding='utf-8-sig',newline='') as f:
  for raw in csv.DictReader(f):
   if raw['stage'] not in stages:continue
   try:
    t=float(raw.get('sensor_t_s',float(raw['device_time_unwrapped_us'])/1e6));x=float(raw['motor_x_pulses'])/ppmm;by=float(raw['dBy_uT']) if raw.get('dBy_uT','') else float(raw['By_uT'])-base
    valid=raw.get('sample_valid','1').strip().lower() not in ('0','false','no') and np.isfinite([t,x,by]).all()
    if np.isfinite([t,x,by]).all():stages[raw['stage']].append([t,x,by,int(valid)])
   except (ValueError,KeyError):pass
 aa=[np.array(stages[s],float) for s in stages]
 origin=aa[0][0,1];seq=meta.get('sequence',0);rules={};details={}
 for rule in ('fixed-x-v1','relative-035-v1'):
  windows=[(45.,142.),(16.,114.)] if rule=='fixed-x-v1' or seq<35 else [(origin+44,origin+144)]*2
  ss=[]
  for a,bounds in zip(aa,windows):
   lo,hi=bounds;use=(a[:,1]>=lo)&(a[:,1]<=hi)&(a[:,3]>0);good=a[use];dt=np.diff(good[:,0]);i=int(np.argmax(dt)) if len(dt) else 0
   gap=float(dt[i]*1000) if len(dt) else None;center=float((good[i,0]+good[i+1,0])/2) if len(dt) else float(a[0,0]);zoom=a[np.abs(a[:,0]-center)<max(.11,(gap or 0)/1000*.8)]
   ss.append(dict(bounds=list(bounds),full_x=[float(a[:,1].min()),float(a[:,1].max())],valid_contact_x=[float(good[:,1].min()),float(good[:,1].max())] if len(good) else None,
     maximum_valid_gap_ms=gap,invalid_contact=int(((a[:,1]>=lo)&(a[:,1]<=hi)&(a[:,3]<1)).sum()),gap_times=[float(good[i,0]),float(good[i+1,0])] if len(dt) else [],zoom=np.round(zoom,5).tolist()))
  rules[rule]=ss
 if file_hash(p)!=row['sha256']:raise RuntimeError('来源在检查期间改变：'+str(p))
 return dict(record=row['record'],label=row['label'],path=str(p),sequence=seq,origin=origin,checks={rule:row[rule] for rule in rules},rules=rules,waves=[compact(a) for a in aa]),aa

if __name__=='__main__':
 scan=json.loads((DEST/'scan.json').read_text());rr=[r for r in scan['records'] if any(r[rule]['status']=='data_error' for rule in scan['rules'])];dataset=[];raw_selected={}
 selected=[]
 for reason in ('step_003未覆盖完整裁剪范围','step_004未覆盖完整裁剪范围','step_003缺口超过20ms','step_004缺口超过20ms','step_004不足预先固定的完整扫描时长，网格超过最后有效观测'):
  match=[r for r in rr if r['fixed-x-v1']['reason']==reason]
  if not match:match=[r for r in rr if r['relative-035-v1']['reason']==reason]
  if match:selected.append(match[0]['record'])
 for row in rr:
  item,aa=parse(row);dataset.append(item)
  if row['record'] in selected:raw_selected[row['record']]=(item,aa)
 print('waveform records',len(dataset),flush=True)
 (DEST/'waveform-details.json').write_text(json.dumps(dataset,ensure_ascii=False,separators=(',',':'),allow_nan=False))
 # Static representative waveforms. Full raw data used, never interpolated across gaps.
 examples=[raw_selected[name] for name in selected if name in raw_selected]
 fig,axes=plt.subplots(len(examples),2,figsize=(14,4*len(examples)),squeeze=False)
 for j,(item,aa) in enumerate(examples):
  rule='fixed-x-v1';reason=item['checks'][rule]['reason'];specs=item['rules'][rule]
  ax=axes[j,0]
  for a,s,color,direction in zip(aa,specs,['#2563a6','#d17b22'],['去程','回程']):
   ax.plot(a[:,1],np.abs(a[:,2]),lw=.55,color=color,alpha=.75,label=direction+' |dBy|')
   ax.axvspan(*s['bounds'],color=color,alpha=.065)
   for x in s['bounds']:ax.axvline(x,color=color,ls='--',lw=.9)
  ax.set_title(item['record']+'：原始幅值与固定X接触窗',loc='left',fontsize=11);ax.set_xlabel('原始电机X坐标 / mm');ax.set_ylabel('|dBy| / µT');ax.legend(fontsize=9);ax.grid(alpha=.2)
  ax=axes[j,1]
  if '缺口' in reason:
   k=1 if '004' in reason else 0;s=specs[k];a=np.array(s['zoom']);valid=a[:,3]>0;good=a[valid];ax.plot(good[:,0],good[:,2],'.-',color='#2563a6',ms=3,lw=.7);bad=a[~valid]
   if len(bad):ax.scatter(bad[:,0],bad[:,2],marker='x',color='#c72e36',s=25,label='采集标记无效')
   ax.axvspan(*s['gap_times'],color='#e95757',alpha=.18,label=f"有效采样间隔 {s['maximum_valid_gap_ms']:.2f} ms")
   ax.set_xlabel('传感器时间 / s');ax.set_ylabel('带符号 dBy / µT');ax.set_title('缺口放大：阈值20 ms，红区为相邻有效点间隔',loc='left',fontsize=11);ax.legend(fontsize=9)
  else:
   for a,s,color,direction in zip(aa,specs,['#2563a6','#d17b22'],['去程','回程']):
    ax.plot(a[:,0],a[:,1],color=color,lw=1,label=direction+f"实际X {a[:,1].min():.2f}–{a[:,1].max():.2f} mm")
    ax.hlines(s['bounds'],a[0,0],a[-1,0],color=color,ls='--',lw=1)
   ax.set_title('轨迹与要求范围：虚线为裁剪边界',loc='left',fontsize=11);ax.set_xlabel('传感器时间 / s');ax.set_ylabel('电机X坐标 / mm');ax.legend(fontsize=9)
  ax.grid(alpha=.2)
  axes[j,0].text(0,-.27,'判定：'+reason+'；相对起点规则：'+item['checks']['relative-035-v1']['status']+' '+item['checks']['relative-035-v1']['reason'],transform=axes[j,0].transAxes,fontsize=9,color='#9b2525',wrap=True)
 fig.suptitle('数据错误样本：波形存在，不代表满足接触范围与有效采样间隔要求',fontsize=16,y=.995)
 fig.tight_layout(rect=[0,.02,1,.98],h_pad=4);fig.savefig(DEST/'错误样本波形总览.png',dpi=145)
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 from matplotlib.transforms import Bbox
 for j,(item,_) in enumerate(examples):
  box=Bbox.union([ax.get_tightbbox(renderer) for ax in axes[j]]).transformed(fig.dpi_scale_trans.inverted()).expanded(1.02,1.08)
  fig.savefig(DEST/('波形_'+item['record']+'.png'),bbox_inches=box,dpi=145)
 plt.close(fig)
 template=(DEST/'gallery-template.html').read_text()
 (DEST/'错误样本波形浏览.html').write_text(template.replace('__DATA__',json.dumps(dataset,ensure_ascii=False,separators=(',',':'),allow_nan=False)).replace('__COUNTS__',json.dumps(scan['rules'],ensure_ascii=False)))
