"""Create an explicitly nonformal continuation; leave the source task intact."""
import copy
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from .data import file_hash
from .storage import new_task,write_json


def fork_runtime_validation(task,meta,target_key,runtime):
    task=Path(task).resolve()
    # list_tasks scans one level only; incomplete copies stay unpublished.
    staging=new_task(task.parent/'.runtime-validation-staging',prefix='CLOTHFIX');derived=task.parent/staging.name
    try:
        checkpoint=task/target_key/'recovery.ckpt';source_sha=file_hash(checkpoint) if checkpoint.is_file() else None
        shutil.copytree(task,staging,dirs_exist_ok=True)
        provenance=dict(source_task=str(task),from_runtime=meta.get('training_runtime') or 'unversioned',to_runtime=runtime,
                        target_key=target_key,checkpoint_sha256=source_sha)
        provenance['from']=provenance['from_runtime'];provenance['to']=runtime
        if source_sha:
            copied=staging/target_key/'recovery.ckpt'
            if file_hash(copied)!=source_sha:raise ValueError('源恢复点在建立验证副本期间发生改变')
            shutil.copyfile(copied,staging/target_key/'recovery-v1-original.ckpt')
        result=copy.deepcopy(meta)
        result.update(id=derived.name,experiment_id=uuid.uuid4().hex,name=meta['name']+' · 修复续训验证',created_at=datetime.now().isoformat(),
                      status='running',purpose='quick',validation_kind='runtime_upgrade',parent_task=str(task),validation_target_key=target_key,
                      runtime_upgrade=provenance,training_runtime=runtime)
        write_json(staging/'metadata.json',result)
        staging.rename(derived)
        return derived,result
    except Exception:
        shutil.rmtree(staging)
        raise
