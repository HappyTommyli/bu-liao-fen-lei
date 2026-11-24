from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
DEST=Path(__file__).parent
d=json.loads((DEST/'重合度分析.json').read_text());source=json.loads((DEST/'原始数据清单.json').read_text());cache=np.load(DEST/'原始波形统计缓存.npz');profiles=cache['profiles']
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
name=font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[name,'DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
labels=[s['label'] for s in sorted(d['classes'],key=lambda r:r['amplitude'])]
matrix=np.eye(len(labels))
for pair in d['pairs']:
    i,j=labels.index(pair['a']),labels.index(pair['b']);matrix[i,j]=matrix[j,i]=pair['overlap']
fig,ax=plt.subplots(figsize=(12,10));im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlOrRd')
for i in range(len(labels)):
 for j in range(len(labels)):
    ax.text(j,i,'—' if i==j else f'{100*matrix[i,j]:.0f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>.65 else '#333333')
ax.set_xticks(range(len(labels)),labels,rotation=65,ha='right');ax.set_yticks(range(len(labels)),labels)
ax.set_title('原始 |ΔBy| 的90%波动带重合率（%）\n逐方向按2mm位置箱对齐，保留幅值；这是区间面积重合，不是错分率',pad=15)
fig.colorbar(im,ax=ax,fraction=.035,pad=.03,label='交集面积 / 并集面积');fig.tight_layout();fig.savefig(DEST/'原始波形重合矩阵.png',dpi=160);plt.close(fig)
chosen=d['pairs'][:3]
fig,axes=plt.subplots(3,2,figsize=(14,10))
for row,pair in enumerate(chosen):
 for col,(start,end,x,di) in enumerate([(0,48,np.arange(46,142,2),'去程'),(48,96,np.arange(18,114,2),'回程')]):
    ax=axes[row,col]
    for label,color in [(pair['a'],'#246DA5'),(pair['b'],'#D27720')]:
        ix=[r['feature_index'] for r in source['records'] if r['eligible'] and r['label']==label]
        qq=np.nanquantile(profiles[ix,start:end],[.05,.5,.95],axis=0)
        ax.fill_between(x,qq[0],qq[2],alpha=.2,color=color)
        ax.plot(x,qq[1],lw=1.8,color=color,label=label)
    ax.set_title(f"{di}：{pair['a']} 与 {pair['b']}（往返合计重合{pair['overlap']:.1%}）",fontsize=10)
    ax.set_xlabel('软件记录的X位置 / mm');ax.set_ylabel('|ΔBy| / μT');ax.legend(fontsize=9,loc='best');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.suptitle('重合最高的三对布料：实线为类中位数，色带为重复采集的第5～95百分位\n每个CSV是一个独立统计记录；不改变基线、不做逐条标准化、不使用模型结果',fontsize=12)
fig.tight_layout(rect=(0,0,1,.94));fig.savefig(DEST/'相似布料原始波形对比.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(12,8));rng=np.random.default_rng(42)
colors=['#7498C3','#D99944','#499E87','#B077AE','#777777']
for i,label in enumerate(labels):
 rr=[r for r in source['records'] if r['eligible'] and r['label']==label]
 for r in rr:
    bucket=min((r['sequence']-1)//10,4)
    ax.scatter(r['amplitude'],i+rng.uniform(-.12,.12),s=18,c=colors[bucket],alpha=.8)
ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_xlabel('去、回接触窗幅值中位数的平均 / μT')
ax.set_title('全部可用原始采集的幅值分布\n蓝=001～010；橙=011～020；绿=021～030；紫=031～040；灰=041以后')
ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(DEST/'各布料原始幅值分布.png',dpi=160);plt.close(fig)
print('PLOTS EXPORTED')
