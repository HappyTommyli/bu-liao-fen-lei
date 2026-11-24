"""Bounded CUDA acceptance; creates no formal model-library experiment."""
from pathlib import Path
from dataclasses import replace
import json,sys,time
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from test_modern_training import trials
from cloth_trainer.modern_config import ModernConfig
from cloth_trainer.modern_training import fit
from cloth_trainer.devices import diagnostics
from cloth_trainer.engine import Cancelled
from cloth_trainer.dataset import load_dataset
from cloth_trainer.data import prepare_trials


def main():
    assert torch.cuda.is_available(),'GPU未生效，拒绝以CPU替代'
    dest=Path(__file__).parent;out=dict(device=diagnostics(),checks={},formal_training=False)
    tt=trials();cfg=ModernConfig(device='cuda',workers=2,epochs=2,batch_size=2,hidden_size=32)
    for precision in ('fp32','amp_fp16','amp_bf16'):
        run=replace(cfg,precision=precision)
        direct=fit(tt[:6],tt[6:],['A','B'],run,'gru','mean',42,dest/('resume-direct-'+precision))
        def stop(e):
            if e['type']=='epoch' and e['epoch']==1:raise Cancelled('验收：主动模拟中断')
        try:fit(tt[:6],tt[6:],['A','B'],run,'gru','mean',42,dest/('resume-retry-'+precision),emit=stop)
        except Cancelled:pass
        resumed=fit(tt[:6],tt[6:],['A','B'],run,'gru','mean',42,dest/('resume-retry-'+precision))
        equal=all(torch.equal(v,resumed['model'].state_dict()[k]) for k,v in direct['model'].state_dict().items())
        assert equal,precision+' 恢复结果不一致'
        out['checks'][precision]=dict(exact_resume=True,device=str(next(resumed['model'].parameters()).device),workers=2)
        print(precision,'GPU训练和恢复完全一致',flush=True)
        del direct,resumed;torch.cuda.empty_cache()
    records,_=load_dataset(ROOT/'outputs/models/CLOTH5_20261117_142528_39d7d3/dataset')
    labels=sorted({r.label for r in records})[:2];rr=[[r for r in records if r.label==label][:2] for label in labels]
    train=prepare_trials([r[0] for r in rr],'by');val=prepare_trials([r[1] for r in rr],'by')
    realcfg=ModernConfig(device='cuda',workers=2,epochs=1)
    out['real_lengths']=[len(t.values) for t in train+val];out['real_smoke']={}
    for arch in ('gru','tcn','cera'):
        torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        print('真实完整接触段 GPU 单轮验证：',arch,'长度',out['real_lengths'],flush=True)
        r=fit(train,val,labels,realcfg,arch,'mean',42,dest/('real-smoke-'+arch),emit=lambda e:print(e.get('message',''),flush=True))
        out['real_smoke'][arch]=dict(runtime=r['runtime'],epochs=len(r['history']),peak_allocated_MB=torch.cuda.max_memory_allocated()/1e6,seconds=time.monotonic()-start)
        assert next(r['model'].parameters()).is_cuda
        del r;torch.cuda.empty_cache()
    (dest/'actual-gpu-acceptance.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print('GPU验收通过；只运行了有限验证，未启动正式训练',flush=True)

if __name__=='__main__':main()
