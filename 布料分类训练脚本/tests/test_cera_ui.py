from dataclasses import replace
from test_ui import app,wait,dispose_test_windows
from test_data import fixture_csv
from cloth_trainer.data import load_record
from cloth_trainer.tcn_ui import TCNTrainerWindow


def test_cera_editor_submit_restore_and_paired_count(app,tmp_path):
    w=TCNTrainerWindow(tmp_path/'models');w.navigation.setCurrentRow(1);w.show();e=w.editor
    assert 'architecture' in e.fields
    combo=e.fields['architecture'];combo.setCurrentIndex(combo.findData('cera'))
    cfg=w.controls_config()
    assert cfg.architectures==('cera',) and cfg.tcn_spec['gru_hidden']==128 and cfg.tcn_spec['gru_layers']==1
    assert cfg.tcn_spec['pooling']=='mean' and e.gru_group.isVisible()
    e.fields['gru_hidden'].setValue(96);e.fields['gru_layers'].setValue(2)
    e.fields['tcn_mode'].setCurrentIndex(e.fields['tcn_mode'].findData('overlap_pair'))
    assert '2项' in w.task_count.text() and 'GRU96' in w.candidate_preview.toPlainText()
    rr=[load_record(fixture_csv(tmp_path/'raw',c,i,speed=100),signals=('by',)) for c in ('布A','布B') for i in range(3)]
    w.add_records(rr,[]);request=w.make_training_request()
    assert request['config']['architectures']==('cera',) and request['config']['tcn_spec']['gru_hidden']==96
    target=tmp_path/'project.json';w.write_project(target)
    b=TCNTrainerWindow(tmp_path/'other');b.read_project(target);wait(app,lambda:not b.import_jobs)
    assert not b.project_read_only and b.controls_config().architectures==('cera',)
    assert b.controls_config().tcn_spec['gru_layers']==2
    w.close();b.close()


def test_old_whole_record_cera_project_can_create_fresh_window_experiment(app,tmp_path):
    from cloth_trainer.modern_config import ModernConfig
    from cloth_trainer.storage import write_json,read_json
    cfg=ModernConfig(architectures=('cera',),signals=('by',),tcn_spec=None)
    value=dict(format='cloth-project-v3',config=cfg.to_dict(),store=str(tmp_path/'models'),sources=[])
    path=tmp_path/'old.json';write_json(path,value)
    w=TCNTrainerWindow(tmp_path/'models');w.read_project(path)
    assert w.project_read_only
    w.new_tcn_from_project()
    assert not w.project_read_only and w.controls_config().tcn_spec is not None
    assert read_json(path)['config']['architectures']==['cera'] and 'tcn_spec' not in read_json(path)['config']
    w.close()
