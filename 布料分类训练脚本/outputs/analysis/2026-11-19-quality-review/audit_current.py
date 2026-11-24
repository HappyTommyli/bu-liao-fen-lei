"""Fresh read-only review of all current fabric sources; no training/model access."""
import sys,json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from cloth_trainer.data import discover_report,import_dataset
from cloth_trainer.config import DataConfig
DEST=Path(__file__).parent;SOURCE=ROOT.parent/'xz-control/data/布料采集数据'
paths,ignored,discovery_errors=discover_report([SOURCE]);fabric=[];other=[]
for p in paths:
    try:m=json.loads((p.parent/'metadata.json').read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):m={}
    (other if m.get('label') in ('三角形间隔','矩形间隔') else fabric).append(p)
started=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def progress(i,n):
    if i%50==0 or i==n:print('AUDIT',i,n,flush=True)
records,report=import_dataset(fabric,DataConfig(),signals=('by',),progress=progress)
out=dict(scope=dict(started_at=started,finished_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),source=str(SOURCE),fabric_count=len(fabric),
    trajectory_count=len(other),trajectory_paths=[str(p)for p in other],ignored_count=len(ignored),discovery_errors=discovery_errors,
    signal='|ΔBy|',model_results_used=False,raw_sources_modified=False),report=report)
(DEST/'当前完整审查批次.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print('FRESH AUDIT SAVED',len(fabric),'records',len(records),'errors',len(report['errors']),'eligible',report.get('eligible_count'),flush=True)
