import pytest
import torch


def test_recovery_roundtrip_and_identity(tmp_path):
    from cloth_trainer.recovery import save_recovery,load_recovery
    assert load_recovery(tmp_path,{'fold':0}) is None
    save_recovery(tmp_path,{'epoch':2,'weights':{'w':torch.tensor([1.,2.])}}, {'fold':0})
    saved=load_recovery(tmp_path,{'fold':0})
    assert saved['epoch']==2 and torch.equal(saved['weights']['w'],torch.tensor([1.,2.]))
    with pytest.raises(ValueError,match='关联|匹配'):load_recovery(tmp_path,{'fold':1})
    p=tmp_path/'recovery.ckpt';p.write_bytes(p.read_bytes()+b'altered')
    with pytest.raises(ValueError,match='校验|损坏'):load_recovery(tmp_path,{'fold':0})


def test_interrupted_replace_preserves_previous_epoch(tmp_path,monkeypatch):
    from cloth_trainer.recovery import save_recovery,load_recovery
    save_recovery(tmp_path,{'epoch':1,'w':torch.ones(2)}, {})
    def interrupted(*args):raise OSError('disk unavailable')
    with monkeypatch.context() as fault:
        fault.setattr('cloth_trainer.recovery.os.replace',interrupted)
        with pytest.raises(OSError):save_recovery(tmp_path,{'epoch':2,'w':torch.zeros(2)}, {})
    assert load_recovery(tmp_path,{})['epoch']==1


def test_invalid_state_cannot_overwrite_valid_checkpoint(tmp_path):
    from cloth_trainer.recovery import save_recovery,load_recovery
    save_recovery(tmp_path,{'epoch':1,'optimizer':{'v':torch.ones(1)}}, {})
    with pytest.raises(ValueError,match='非有限'):save_recovery(tmp_path,{'epoch':2,'optimizer':{'v':torch.tensor([float('nan')])}}, {})
    assert load_recovery(tmp_path,{})['epoch']==1


def test_runtime_upgrade_is_explicit_and_never_bypasses_other_binding(tmp_path):
    from cloth_trainer.recovery import save_recovery,load_recovery
    old=dict(runtime='numerical-safety-v1',job={'fold':0},config={'lr':.001},data='abc')
    new=dict(old,runtime='numerical-safety-v2')
    save_recovery(tmp_path,{'epoch':21,'w':torch.ones(1)},old)
    original=(tmp_path/'recovery.ckpt').read_bytes()
    with pytest.raises(ValueError,match='关联|匹配'):load_recovery(tmp_path,new)
    state=load_recovery(tmp_path,new,allow_runtime_upgrade=True)
    assert state['epoch']==21 and state['_runtime_upgrade']['from']=='numerical-safety-v1'
    assert (tmp_path/'recovery.ckpt').read_bytes()==original
    for mismatch in [dict(new,config={'lr':.0003}),dict(new,data='changed'),dict(new,job={'fold':1}),dict(new,runtime='unknown')]:
        with pytest.raises(ValueError,match='关联|匹配'):load_recovery(tmp_path,mismatch,allow_runtime_upgrade=True)
