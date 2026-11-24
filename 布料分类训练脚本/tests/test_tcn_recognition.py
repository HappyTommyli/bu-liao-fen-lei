from dataclasses import replace
from pathlib import Path
import pytest
from PyQt5 import QtWidgets as W, QtCore as C
from test_ui import app, wait, dispose_test_windows
from test_data import fixture_csv
from cloth_trainer.tcn_ui import TCNTrainerWindow
from cloth_trainer.storage import read_json, write_json


def test_independent_inputs_labels_and_removal_do_not_enter_training(app,tmp_path):
    w=TCNTrainerWindow(tmp_path/'models')
    paths=[fixture_csv(tmp_path/'new',c,0,speed=100) for c in ('布A','布B')]
    w.add_recognition([str(tmp_path/'new'),str(paths[0])])
    assert w.navigation.count()==4 and w.recognition_queue.rowCount()==2
    assert len(w.recognition_paths)==2 and not w.records and not w.overrides
    w.set_recognition_labels([0],'布A')
    key=w.recognition_paths[0]
    assert w.recognition_overrides[key]=='布A' and not w.overrides
    w.remove_recognition([0])
    assert key not in w.recognition_paths and key not in w.recognition_overrides
    w.clear_recognition()
    assert w.recognition_queue.rowCount()==0 and not w.recognition_overrides
    w.close()


def test_independent_page_submits_saved_final_model_and_separate_labels(app,tmp_path,monkeypatch):
    w=TCNTrainerWindow(tmp_path/'models')
    raw=fixture_csv(tmp_path/'new','布A',0,speed=100)
    w.add_recognition([str(raw)]);w.set_recognition_labels([0],'布A')
    directory=tmp_path/'chosen';directory.mkdir()
    write_json(directory/'metadata.json',dict(name='已选模型',final_model={},labels=[],config={}))
    w.model_selector.addItem('最终模型',(str(directory),'final'))
    requests=[]
    monkeypatch.setattr(w,'launch',lambda request,kind:requests.append((request,kind)))
    w.start_prediction()
    request,kind=requests[0]
    assert kind=='predict' and request['independent'] is True
    assert request['models']==[str(directory/'final/model.pt')]
    assert request['overrides']=={str(Path(raw).resolve()):'布A'}
    assert 'recognition' in Path(request['destination']).parts and not w.records
    w.close()


@pytest.mark.parametrize('size',[(1024,640),(1366,768)])
def test_independent_page_scrolls_without_squeezing_controls(app,tmp_path,size):
    w=TCNTrainerWindow(tmp_path/'models');w.resize(*size);w.show();w.navigation.setCurrentRow(3)
    app.processEvents()
    assert w.pages.currentIndex()==3 and '在线' in w.page_title.text()
    assert not w.predict_button.isVisible() and not w.recognition_queue.isVisible()
    assert w.recognition_scroll.widget().minimumSizeHint().height()<=w.recognition_scroll.widget().height()
    assert w.recognition_scroll.horizontalScrollBar().maximum()==0
    w.close()


def test_independent_model_choices_exclude_non_tcn_and_unfinished(app,tmp_path):
    store=tmp_path/'models'
    for name,arch,signal,final,status in [('valid','tcn','by',True,'completed'),('gru','gru','by',True,'completed'),('xy','tcn','xy',True,'completed'),('nofinal','tcn','by',False,'completed')]:
        d=store/name;d.mkdir(parents=True)
        write_json(d/'metadata.json',dict(format='cloth-classification-v4',name=name,status=status,purpose='formal',records=0,models={},config={'architectures':[arch],'signals':[signal]},final_model=dict(architecture=arch,signal=signal,readout='mean',seed=1,model_id='test') if final else None))
    w=TCNTrainerWindow(store)
    wait(app,lambda:not w.desktop_jobs.running())
    assert w.prediction_experiment.count()==1 and w.prediction_experiment.currentText()=='valid'
    assert w.model_selector.currentData()==(str(store/'valid'),'final')
    w.close()


def test_independent_final_inference_evaluates_new_only_and_exports(app,tmp_path):
    from cloth_trainer.engine import train_request, final_request
    from cloth_trainer.data import load_record
    from cloth_trainer.modern_config import ModernConfig
    from cloth_trainer.tcn_models import default_spec
    rr=[load_record(fixture_csv(tmp_path/'raw',c,i,speed=100),signals=('by',)) for c in ('布A','布B') for i in range(6)]
    cfg=ModernConfig(device='cpu',workers=0,architectures=('tcn',),tcn_spec=default_spec(16,1),tcn_window=128,epochs=1,batch_size=4)
    task=train_request(dict(paths=[r.path for r in rr],sources=[r.manifest() for r in rr],config=cfg.to_dict(),output=str(tmp_path/'models')))
    final_request(dict(task=str(task)))
    before=(task/'metadata.json').read_bytes()
    # Fixed boundaries now apply to every sequence; a different scan speed
    # still violates the saved model protocol and must be rejected.
    fresh=fixture_csv(tmp_path/'fresh','布A',10,speed=100)
    unknown=fixture_csv(tmp_path/'fresh','布B',11,speed=100)
    outside=fixture_csv(tmp_path/'fresh','布C',12,speed=100)
    mismatched=fixture_csv(tmp_path/'fresh','布A',42,speed=90)
    w=TCNTrainerWindow(tmp_path/'models')
    w.add_recognition([str(fresh),str(unknown),str(outside),str(mismatched),rr[0].path,str(tmp_path/'missing.csv')])
    w.set_recognition_labels([w.recognition_paths.index(str(unknown))],'')
    wait(app,lambda:bool(w.model_selector.currentData()));w.start_prediction();wait(app,lambda:w.prediction_data is not None and not w.busy())
    assert w.prediction_data is not None
    assert w.prediction_data['evaluation_kind']=='independent'
    result=w.prediction_data['results'][0];rows=result['predictions']
    assert len(rows)==8 and all('windows' in r for r in rows)
    assert result['evaluation']['new']['total']==2
    assert result['evaluation']['replay']['total']==2
    assert result['evaluation']['unknown_labels']==4
    assert sum(r['training_overlap'] for r in rows)==2
    assert len(result['errors'])==1 and '协议不匹配' in result['errors'][0]['message']
    assert 'missing.csv' in w.recognition_errors.toPlainText()
    assert w.recognition_metrics['total']==2 and w.recognition_class_table.rowCount()==2
    w.matrix_scope.setCurrentIndex(w.matrix_scope.findData('replay'))
    assert w.recognition_metrics['total']==2
    w.matrix_direction.setCurrentIndex(w.matrix_direction.findData('out'))
    assert w.recognition_metrics['total']==1
    assert (task/'metadata.json').read_bytes()==before
    assert not w.records and not w.overrides and w.navigation.currentRow()==3
    dest=tmp_path/'detail.csv';w.export_recognition_csv(dest)
    import csv
    with dest.open(encoding='utf-8-sig') as f:assert len(list(csv.reader(f)))==9
    w.close()


def test_busy_recognition_inputs_and_invalid_paths(app,tmp_path,monkeypatch):
    w=TCNTrainerWindow(tmp_path/'models')
    w.add_recognition([str(tmp_path/'missing.csv')])
    assert not w.recognition_paths and '不存在' in w.recognition_errors.toPlainText()
    path=fixture_csv(tmp_path/'new');w.add_recognition([str(path)])
    monkeypatch.setattr(w,'busy',lambda:True)
    w.set_recognition_labels([0],'不应写入');w.remove_recognition([0]);w.clear_recognition()
    assert w.recognition_paths==[str(path)] and not w.recognition_overrides
    monkeypatch.setattr(w,'busy',lambda:False);w.close()
