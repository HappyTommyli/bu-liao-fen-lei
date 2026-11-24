import time
from dataclasses import replace
import pytest
from PyQt5 import QtWidgets as W,QtCore as C
from cloth_trainer.ui import TrainerWindow
from cloth_trainer.data import load_record
from test_data import fixture_csv

@pytest.fixture(scope='session')
def app():return W.QApplication.instance() or W.QApplication([])

@pytest.fixture(autouse=True)
def dispose_test_windows(app):
    yield
    # Drain deferred Matplotlib paints while their Qt parents still exist;
    # close() hides widgets, whereas deleteLater() gives Qt a stable teardown.
    windows=list(app.topLevelWidgets())
    for w in windows:
        if isinstance(w,TrainerWindow):
            w.close();wait(app,lambda:not w.import_jobs and (not getattr(w,'desktop_jobs',None) or not w.desktop_jobs.running()) and (not getattr(w,'online_worker',None) or not w.online_worker.isRunning()))
        w.close()
    app.processEvents()
    for w in windows:w.deleteLater()
    app.sendPostedEvents(None,C.QEvent.DeferredDelete)
    app.processEvents()

def wait(app,predicate):
    end=time.monotonic()+40
    while time.monotonic()<end and not predicate():app.processEvents();time.sleep(.01)
    assert predicate()

def test_import_labels_and_project_recovery(app,tmp_path):
    fixture_csv(tmp_path/'data',speed=100)
    w=TrainerWindow(tmp_path/'store');w.import_paths([str(tmp_path/'data')]);wait(app,lambda:not w.import_jobs)
    assert len(w.records)==1 and w.navigation.count()==4
    assert not w.start_button.isEnabled()
    w.assign_labels([0],'布料新名');w.data_table.item(0,0).setCheckState(C.Qt.Unchecked)
    project=tmp_path/'project.json';w.write_project(project)
    b=TrainerWindow(tmp_path/'another');b.read_project(project);wait(app,lambda:not b.import_jobs)
    assert b.records[0].label=='布料新名' and len(b.selected_records())==0
    w.close();b.close()

def test_training_snapshot_and_input_controls(app,tmp_path):
    records=[load_record(fixture_csv(tmp_path,c,i,speed=100)) for c in ['布A','布B'] for i in range(10)]
    w=TrainerWindow(tmp_path/'store');w.add_records(records,[])
    assert w.start_button.isEnabled()
    assert w.signal_checks['by'].isChecked() and not w.signal_checks['xy'].isChecked()
    w.signal_checks['xy'].setChecked(True)
    req=w.make_training_request()
    assert len(req['sources'])==20 and req['config']['signals']==('by','xy')
    assert req['config']['architectures']==('gru','tcn','cera')
    assert req['config']['readouts']==('mean',)
    assert '30' in w.task_count.text()
    w.data_table.item(0,0).setCheckState(C.Qt.Unchecked)
    assert w.start_button.isEnabled() and len(req['paths'])==20
    w.close()

def test_gui_background_training_history_and_prediction(app,tmp_path,monkeypatch):
    records=[load_record(fixture_csv(tmp_path,c,i,speed=100)) for c in ['布A','布B'] for i in range(10)]
    w=TrainerWindow(tmp_path/'store');w.config=replace(w.config,device="cpu",workers=0,epochs=1,batch_size=4);w.sync_controls();w.add_records(records,[])
    real_config=w.controls_config
    monkeypatch.setattr(w,"controls_config",lambda:replace(real_config(),device="cpu",workers=0))
    w.start_training();wait(app,lambda:not w.busy())
    assert w.result_task and w.result_task['status']=='completed'
    assert w.model_selector.count()==0
    assert w.final_button.isEnabled()
    assert w.comparison.rowCount()==3
    assert all(r['test_metrics'] is None for r in w.result_task['models'].values())
    assert not w.stage2_button.isEnabled()
    w.run_final()
    wait(app,lambda:not w.busy())
    assert w.result_task['final_model'] and w.model_selector.count()==1
    w.run_test()
    wait(app,lambda:not w.busy())
    assert w.result_task['holdout']['metrics']['total']==8
    assert all(r['test_metrics'] is None for r in w.result_task['models'].values())
    w.add_recognition([records[0].path]);w.start_prediction();wait(app,lambda:not w.busy())
    assert len(w.prediction_rows)==2
    assert all(r['training_overlap'] for result,r in w.prediction_rows)
    assert '整段分类' in w.prediction_table.item(0,5).text()
    w.close()

def test_reimport_replaces_changed_path_and_invalidates_qc_failure(app,tmp_path):
    p=fixture_csv(tmp_path,speed=100);old=load_record(p)
    w=TrainerWindow(tmp_path/'store');w.add_records([old],[]);w.assign_labels([0],'手动标签');w.data_table.item(0,0).setCheckState(C.Qt.Unchecked)
    p.write_text(p.read_text(encoding='utf-8-sig')+'\n',encoding='utf-8-sig')
    new=load_record(p);w.add_records([new],[])
    assert len(w.records)==1 and w.records[0].sha256==new.sha256
    assert not w.selected_records() and w.records[0].label=='手动标签'
    w.add_records([],[dict(path=str(p.resolve()),message='回程缺口')])
    assert len(w.records)==0 and len(w.errors)==1
    w.close()


def test_presets_readout_and_repeated_library(app,tmp_path):
    from cloth_trainer.storage import write_json
    w=TrainerWindow(tmp_path/'store')
    w.preset_selector.setCurrentIndex(w.preset_selector.findData('repeat'))
    assert w.config.repetitions==3 and '45' in w.task_count.text()
    w.last_readout.setChecked(True)
    assert '90' in w.task_count.text()
    task=w.store/'grouped'
    models={f'run{i}':dict(configuration_id='gru_by_mean',architecture='gru',signal='by',readout='mean',seed=42+i,
                          model_id=str(i),val_metrics=dict(macro_f1=.5+i*.1,accuracy=.5,total=2,correct=1,confusion=[[1,0],[1,0]],per_class=[dict(label='A',recall=1),dict(label='B',recall=0)]),test_metrics=None,status='completed',history=[],
                          val_predictions=[],best_epoch=1) for i in range(3)}
    write_json(task/'metadata.json',dict(format='cloth-classification-v2',name='复验',id='x',status='completed',purpose='formal',labels=['A','B'],models=models))
    w.refresh_history();w.show_results(task)
    assert w.comparison.rowCount()==1 and w.result_selector.count()==3
    assert w.comparison.item(0,3).text()=='未测试'
    assert '后续开放' in w.comparison.item(0,4).text()
    w.close()


def test_start_after_import_respects_invalid_selection(app,tmp_path):
    w=TrainerWindow(tmp_path/'store')
    assert w.start_imported_training() is False
    assert not w.busy()
    assert '未启动' in w.status.text()
    w.close()


def test_library_refresh_switch_and_table_remain_usable(app,tmp_path):
    from cloth_trainer.storage import write_json
    w=TrainerWindow(tmp_path/'store');w.show()
    for name in ('first','second'):
        models={f'run{i}':dict(configuration_id=f'{a}_by_mean',architecture=a,signal='by',readout='mean',seed=42,
                              model_id=name+str(i),val_metrics=dict(macro_f1=.5,accuracy=.5,total=2,correct=1,confusion=[[1,0],[1,0]],per_class=[dict(label='A',recall=1),dict(label='B',recall=0)]),test_metrics=None,status='completed',history=[],
                              val_predictions=[],best_epoch=1) for i,a in enumerate(('gru','tcn','cera'))}
        write_json(w.store/name/'metadata.json',dict(format='cloth-classification-v2',name=name,id=name,status='completed',purpose='formal',labels=['A','B'],models=models))
    w.show_results(w.store/'first');w.navigation.setCurrentRow(2);app.processEvents()
    assert w.comparison.viewport().height()>=3*w.comparison.rowHeight(0)
    w.show_results(w.store/'first')
    assert w.result_selector.count()==1 and w.result_model['model_id']=='first0'
    w.show_results(w.store/'second')
    assert w.result_selector.count()==1 and w.result_model['model_id']=='second0'
    w.close()


def test_quick_settings_remain_trial_and_readable(app,tmp_path):
    from cloth_trainer.settings_ui import SettingsDialog
    from cloth_trainer.config import TrainConfig
    cfg=TrainConfig().with_preset('quick')
    for changed in (False,True):
        d=SettingsDialog(cfg);d.show();app.processEvents()
        if changed:d.fields['threads'].setValue(3)
        d.accept()
        assert d.result_config.preset=='quick'
        assert d.result_config.threads==(3 if changed else 2)
        d.close()


def test_library_table_has_space_for_multiple_models(app,tmp_path):
    w=TrainerWindow(tmp_path/'store');w.show();w.navigation.setCurrentRow(2);app.processEvents()
    assert w.comparison.viewport().height()>=3*w.comparison.rowHeight(0)
    w.close()


def test_settings_background_contrast(app):
    from cloth_trainer.settings_ui import SettingsDialog
    from cloth_trainer.config import TrainConfig
    d=SettingsDialog(TrainConfig());d.show();app.processEvents()
    body=d.findChild(W.QScrollArea).widget()
    assert body.grab().toImage().pixelColor(2,2).lightnessF()<.2
    d.close()


def test_recognition_matrix_filters_and_export(app,tmp_path):
    from cloth_trainer.storage import write_json,read_json
    w=TrainerWindow(tmp_path/'store')
    rows=[dict(id='1',path='A.csv',direction='out',label='A',true=0,predicted=0,scores=[.9,.1],training_overlap=False),
          dict(id='2',path='B.csv',direction='back',label='B',true=1,predicted=0,scores=[.8,.2],training_overlap=False),
          dict(id='3',path='old.csv',direction='out',label='B',true=1,predicted=1,scores=[.1,.9],training_overlap=True),
          dict(id='4',path='unknown.csv',direction='out',label='',true=-1,predicted=1,scores=[.1,.9],training_overlap=False)]
    p=tmp_path/'prediction.json';write_json(p,dict(results=[dict(model='one',signal='by',labels=['A','B'],errors=[],predictions=rows)]))
    w.show_prediction(p)
    assert w.recognition_metrics['confusion']==[[1,0],[1,0]] and w.recognition_metrics['total']==2
    w.matrix_direction.setCurrentIndex(w.matrix_direction.findData('back'))
    assert w.recognition_metrics['accuracy']==0 and w.recognition_metrics['total']==1
    w.matrix_scope.setCurrentIndex(w.matrix_scope.findData('replay'));w.matrix_direction.setCurrentIndex(0)
    assert w.recognition_metrics['total']==1 and w.recognition_metrics['accuracy']==1
    w.export_recognition_matrix(tmp_path/'matrix')
    assert (tmp_path/'matrix.png').exists() and (tmp_path/'matrix.csv').exists()
    assert read_json(tmp_path/'matrix.json')['scope']=='replay'
    w.clear_recognition();assert w.recognition_metrics['accuracy'] is None
    w.close()


def test_ui_dataset_snapshot_and_required_channels(app,tmp_path):
    from cloth_trainer.dataset import save_dataset
    records=[load_record(fixture_csv(tmp_path/'raw',c,i,speed=100),signals=('by',)) for c in ('布A','布B') for i in range(6)]
    w=TrainerWindow(tmp_path/'store');w.add_records(records,[])
    assert '15' in w.task_count.text() and w.start_button.isEnabled()
    snap=save_dataset(tmp_path/'snap',records,w.config.data,screening=w.screening_value())
    w.load_dataset_version(snap)
    assert w.make_training_request()['snapshot']==str(snap.resolve())
    w.signal_checks['bxby'].setChecked(True)
    assert not w.start_button.isEnabled()
    assert '通道' in w.partition_summary.text()
    w.close()


def test_aggregate_fold_view_is_not_single_fold_score(app,tmp_path):
    from cloth_trainer.storage import write_json
    from cloth_trainer.evaluation import classification_metrics,summarize_configs
    w=TrainerWindow(tmp_path/'store');models={}
    for fold in range(5):
        row=dict(id=str(fold),true=0,predicted=fold%2,path='A.csv',direction='out',label='A',scores=[.5,.5],quality='normal')
        metrics=classification_metrics([row],['A','B'])
        models[str(fold)]=dict(configuration_id='tcn_by_mean',architecture='tcn',signal='by',readout='mean',seed=42,model_id=str(fold),fold=fold,
                                train_metrics=metrics,val_metrics=metrics,test_metrics=metrics,history=[],val_predictions=[row],test_predictions=[row],best_epoch=1)
    meta=dict(format='cloth-classification-v3',name='5折',id='x',status='completed',purpose='formal',labels=['A','B'],models=models)
    meta['summaries']=summarize_configs(meta);task=w.store/'five';write_json(task/'metadata.json',meta);w.show_results(task)
    w.result_partition.setCurrentIndex(1);w.result_aggregate.setCurrentIndex(1)
    assert '3/5' in w.result_notice.text() and len(w.result_rows)==5
    w.close()


def test_duplicate_label_conflict_across_import_batches(app,tmp_path):
    import shutil,json
    p=fixture_csv(tmp_path/'one',speed=100);a=load_record(p)
    folder=tmp_path/'two';folder.mkdir();q=folder/'copy.csv';shutil.copy(p,q)
    md=json.loads((p.parent/'metadata.json').read_text());md['label']='布B';(folder/'metadata.json').write_text(json.dumps(md));b=load_record(q)
    w=TrainerWindow(tmp_path/'store');w.add_records([a],[]);w.add_records([b],[])
    assert not w.records and any('冲突' in e['message'] for e in w.errors)
    w.close()


def test_narrowing_inputs_reprocesses_by_and_snapshot_matches_fresh_import(app,tmp_path):
    import csv,numpy as np
    from cloth_trainer.dataset import load_dataset
    records=[fixture_csv(tmp_path/'raw',c,i,speed=100) for c in ('布A','布B') for i in range(5)]
    p=records[0]
    with p.open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
    for row in rows:
        if row['stage']=='step_003' and .8<float(row['sensor_t_s'])<.804:row['dBx_uT']=row['Bx_uT']='nan'
    with p.open('w',encoding='utf-8-sig',newline='') as f:wcsv=csv.DictWriter(f,fields);wcsv.writeheader();wcsv.writerows(rows)
    w=TrainerWindow(tmp_path/'store');w.config=replace(w.config,signals=('by','bxby'));w.sync_controls();w.import_paths([tmp_path/'raw']);wait(app,lambda:not w.import_jobs)
    w.signal_checks['bxby'].setChecked(False);app.processEvents()
    assert w.import_jobs or all(r.axes==('y',) for r in w.records)
    wait(app,lambda:not w.import_jobs)
    w.save_dataset_version(tmp_path/'by_snapshot');restored,_=load_dataset(tmp_path/'by_snapshot')
    for r in restored:assert np.array_equal(r.trials[0].values,load_record(r.path,signals=('by',)).trials[0].values)
    w.close()


def test_new_absolute_input_labels_and_old_model_labels(app,tmp_path):
    w=TrainerWindow(tmp_path/'store')
    assert '|ΔBy|' in w.signal_checks['by'].text() and '|ΔBx|' in w.signal_checks['bxby'].text()
    w.close()


def test_switching_input_through_empty_choice_does_not_import_empty_channels(app,tmp_path):
    for c in ('布A','布B'):
        for i in range(5):fixture_csv(tmp_path/'raw',c,i,speed=100)
    w=TrainerWindow(tmp_path/'store');w.import_paths([tmp_path/'raw']);wait(app,lambda:not w.import_jobs)
    w.signal_checks['by'].setChecked(False);app.processEvents()
    assert not w.import_jobs
    w.signal_checks['bxby'].setChecked(True);app.processEvents();wait(app,lambda:not w.import_jobs)
    assert len(w.records)==10 and not w.errors and all(r.axes==('x','y') for r in w.records)
    w.close()


def test_recognition_report_identity_distinguishes_same_keys_across_experiments(app,tmp_path):
    from cloth_trainer.storage import write_json,read_json
    a=dict(id='a',path='A.csv',direction='out',label='A',true=0,predicted=0,scores=[.8,.2],training_overlap=False)
    b=dict(a,id='b',predicted=1,scores=[.2,.8])
    results=[dict(model=f'/experiments/{experiment}/shared/model.pt',model_id=experiment,weight_sha256=experiment+'hash',signal='by',labels=['A','B'],errors=[],predictions=[row]) for experiment,row in [('one',a),('two',b)]]
    p=tmp_path/'p.json';write_json(p,dict(results=results));w=TrainerWindow(tmp_path/'store');w.show_prediction(p)
    assert w.matrix_model.itemText(0)!=w.matrix_model.itemText(1)
    assert w.recognition_metrics['total']==1 and w.recognition_metrics['accuracy']==1
    w.matrix_model.setCurrentIndex(1);assert w.recognition_metrics['total']==1 and w.recognition_metrics['accuracy']==0
    w.export_recognition_matrix(tmp_path/'matrix')
    export=read_json(tmp_path/'matrix.json');assert export['model_id']=='two' and export['weight_sha256']=='twohash'
    assert 'two' in w.matrix_notice.text()
    w.close()


def test_admission_rows_remain_visible_and_cannot_be_selected(app,tmp_path):
    from test_r3_data import rewrite
    from cloth_trainer.data import import_dataset
    bad=fixture_csv(tmp_path/'raw','布A',0,speed=100)
    rewrite(bad,modify=lambda row:row.update(dBy_uT=str(float(row['dBy_uT'])*2),By_uT=str(float(row['By_uT'])*2)) if row['stage']=='step_004' else None)
    fixture_csv(tmp_path/'raw','布A',1,speed=100)
    w=TrainerWindow(tmp_path/'store');w.import_paths([str(tmp_path/'raw')]);wait(app,lambda:not w.import_jobs)
    assert len(w.records)==2 and w.data_table.rowCount()==2 and len(w.selected_records())==1
    i=next(i for i,r in enumerate(w.records) if not r.audit['eligible'])
    assert w.data_table.item(i,4).text()=='本实验排除·建议复采'
    assert not w.data_table.item(i,0).flags() & C.Qt.ItemIsUserCheckable
    w.data_table.item(i,0).setCheckState(C.Qt.Checked)
    assert len(w.selected_records())==1
    w.data_table.selectRow(i);app.processEvents();assert '差异率' in w.data_details.text()
    assert '扫描2' in w.data_summary.text() and '排除1' in w.data_summary.text()
    frozen=w.screening_value()['review_batches'][0]['sha256']
    for j in range(len(w.records)):w.data_table.item(j,0).setCheckState(C.Qt.Unchecked)
    assert w.screening_value()['review_batches'][0]['sha256']==frozen
    w.close()


def test_invalid_and_undetermined_do_not_hide_or_block_other_valid_records(app,tmp_path):
    import json
    for c in ('布A','布B'):
        for i in range(6):fixture_csv(tmp_path/'raw',c,i,speed=100)
    missing=fixture_csv(tmp_path/'raw','布A',9,speed=100);mp=missing.parent/'metadata.json';m=json.loads(mp.read_text());m['context']['baseline'].pop('sigma');mp.write_text(json.dumps(m))
    bad=tmp_path/'raw'/'broken.csv';bad.write_text('foo,bar\n1,2\n')
    w=TrainerWindow(tmp_path/'store');w.import_paths([str(tmp_path/'raw')]);wait(app,lambda:not w.import_jobs)
    assert w.data_table.rowCount()==14 and len(w.selected_records())==12 and w.start_button.isEnabled()
    req=w.make_training_request();assert len(req['paths'])==12 and len(req['review_batches'][0]['entries'])==14
    assert '无法判断1' in w.data_summary.text() and '数据错误1' in w.data_summary.text()
    w.close()


def test_snapshot_retains_excluded_audit_rows_and_old_snapshot_is_readonly(app,tmp_path):
    from test_r3_data import rewrite
    from cloth_trainer.storage import read_json,write_json
    from cloth_trainer.data import file_hash
    for c in ('布A','布B'):
        for i in range(6 if c=='布A' else 5):fixture_csv(tmp_path/'raw',c,i,speed=100)
    bad=tmp_path/'raw'/'布A_000'/'布A_000.csv'
    rewrite(bad,modify=lambda row:row.update(dBy_uT=str(float(row['dBy_uT'])*2),By_uT=str(float(row['By_uT'])*2)) if row['stage']=='step_004' else None)
    w=TrainerWindow(tmp_path/'store');w.import_paths([str(tmp_path/'raw')]);wait(app,lambda:not w.import_jobs)
    w.save_dataset_version(tmp_path/'snap')
    assert w.data_table.rowCount()==11 and len(w.records)==10 and len(w.selected_records())==10
    assert any(w.data_table.item(i,4).text()=='本实验排除·建议复采' for i in range(w.data_table.rowCount()))
    p=tmp_path/'snap'/'dataset.json';m=read_json(p);m.pop('admission');write_json(p,m);p.with_name('dataset.sha256').write_text(file_hash(p))
    w.load_dataset_version(tmp_path/'snap');assert not w.selected_records() and not w.start_button.isEnabled()
    assert '旧快照' in w.partition_summary.text();w.close()


def test_channel_refresh_keeps_removed_candidates_in_fixed_batch(app,tmp_path):
    from test_r3_data import rewrite
    paths=[fixture_csv(tmp_path/'raw',c,i,speed=100) for c in ('布A','布B') for i in (range(25) if c=='布A' else range(100,105))]
    for p in paths[:20]:rewrite(p,modify=lambda row:row.update(dBy_uT=str(float(row['dBy_uT'])*2),By_uT=str(float(row['By_uT'])*2)) if row['stage']=='step_004' else None)
    w=TrainerWindow(tmp_path/'store');w.import_paths([str(tmp_path/'raw')]);wait(app,lambda:not w.import_jobs)
    assert len(w.records)==30,w.errors
    assert all(not r.audit['eligible'] for r in w.records if r.label=='布A')
    w.import_paths([str(p) for p in paths[20:25]]);wait(app,lambda:not w.import_jobs)
    assert all(not r.audit['eligible'] for r in w.records if r.label=='布A')
    w.data_table.clearSelection();w.data_table.setRangeSelected(W.QTableWidgetSelectionRange(0,0,19,4),True);w.remove_selected()
    assert len(w.records)==10
    w.signal_checks['bxby'].setChecked(True);app.processEvents();wait(app,lambda:not w.import_jobs)
    assert all(not r.audit['eligible'] for r in w.records if r.label=='布A')
    assert any(b['classes']['布A']['assessable_groups']==25 for b in w.batches)
    w.close()


def test_label_refresh_and_project_preserve_incremental_batch_boundaries(app,tmp_path):
    from test_r3_data import rewrite
    first=[fixture_csv(tmp_path/'one',c,i,speed=100) for c in ('布A','布B') for i in range(10 if c=='布A' else 5)]
    for p in first[:8]:rewrite(p,modify=lambda row:row.update(dBy_uT=str(float(row['dBy_uT'])*2),By_uT=str(float(row['By_uT'])*2)) if row['stage']=='step_004' else None)
    second=[fixture_csv(tmp_path/'two','布A',i,speed=100) for i in range(10,20)]
    w=TrainerWindow(tmp_path/'store');w.import_paths([str(tmp_path/'one')]);wait(app,lambda:not w.import_jobs)
    w.import_paths([str(tmp_path/'two')]);wait(app,lambda:not w.import_jobs)
    w.assign_labels([i for i,r in enumerate(w.records) if r.label=='布B'],'布C');wait(app,lambda:not w.import_jobs)
    assert sorted(b['classes']['布A']['assessable_groups'] for b in w.batches)==[10,10]
    assert sum(r.audit['eligible'] for r in w.records if r.label=='布A')==10
    w.write_project(tmp_path/'project.json');w.read_project(tmp_path/'project.json');wait(app,lambda:not w.import_jobs)
    assert sorted(b['classes']['布A']['assessable_groups'] for b in w.batches)==[10,10]
    assert sum(r.audit['eligible'] for r in w.records if r.label=='布A')==10
    w.close()


def test_epoch_restore_event_restores_curve_and_progress(app,tmp_path):
    w=TrainerWindow(tmp_path/'store')
    history=[dict(epoch=1,loss=1.,val_loss=1.1,train_accuracy=.5,val_accuracy=.4,val_f1=.4,lr=.001,seconds=1.)]
    w.on_event(dict(type='restored',epoch=1,history=history,job=2,jobs=5,epochs=4,message='恢复第1轮'))
    assert w.live_history==history and w.progress.value()==25
    assert '恢复第1轮' in w.training_log.toPlainText()


def test_runtime_validation_is_separate_and_not_offered_for_formal_prediction(app,tmp_path):
    from cloth_trainer.storage import write_json
    w=TrainerWindow(tmp_path/'store');task=w.store/'validation'
    write_json(task/'metadata.json',dict(format='cloth-classification-v3',name='修复续训',id='validation',status='completed',purpose='quick',validation_kind='runtime_upgrade',labels=['A','B'],models={},records=10))
    w.refresh_history();w.show_results(task)
    assert w.task_history.count()==1 and w.history.count()==0 and w.prediction_experiment.count()==0
    assert '修复验证（非正式）' in w.experiment_title.text() and not w.test_button.isEnabled()
    assert '修复验证（非正式）' in w.task_history.item(0).text()


def test_modern_gpu_defaults_without_device_choices(app,tmp_path):
    from cloth_trainer.modern_config import ModernConfig
    w=TrainerWindow(tmp_path/'store')
    assert isinstance(w.config,ModernConfig) and w.controls_config().device=='cuda'
    assert not hasattr(w,'device_selector') and not hasattr(w,'precision_selector')
    w.config=replace(w.config,device='cpu')
    assert w.controls_config().device=='cuda'
    from cloth_trainer.settings_ui import SettingsDialog
    d=SettingsDialog(w.controls_config());d.restore();d.accept()
    assert d.result_config.lr==.0003 and d.result_config.early_stopping
    assert d.result_config.workers==2 and d.result_config.micro_batch==0
    d.close();w.close()
