from dataclasses import replace
import json
import pytest
from PyQt5 import QtWidgets as W,QtCore as C
from test_ui import app,wait,dispose_test_windows
from test_data import fixture_csv
from cloth_trainer.data import load_record
from cloth_trainer.storage import write_json,read_json


def test_old_project_can_upgrade_contact_rule_from_data_menu(app,tmp_path):
    from test_tcn_helpers import TCNTrainerWindow
    from cloth_trainer.config import DataConfig
    w=TCNTrainerWindow(tmp_path/'models')
    w.config=replace(w.config,data=DataConfig.from_dict({}))
    actions=[a for menu in w.findChildren(W.QMenu) for a in menu.actions() if a.text()=='采集协议…']
    assert len(actions)==1
    def accept_latest():
        dialog=app.activeModalWidget()
        dialog.restore();dialog.accept()
    C.QTimer.singleShot(0,accept_latest)
    actions[0].trigger()
    assert w.config.data.contact_rule=='fixed-x-v1'
    assert w.config.data.sample_quality=='selected-axis-v1'
    assert w.config.data.out_low==51 and w.config.data.back_low==30
    w.close()


def test_focused_window_only_submits_tcn_by_and_counts_modes(app,tmp_path):
    from test_tcn_helpers import TCNTrainerWindow
    w=TCNTrainerWindow(tmp_path/'models');w.show()
    assert w.navigation.count()==4
    assert w.controls_config().architectures==('tcn',) and w.selected_signals()==('by',)
    assert w.controls_config().tcn_window==900 and '1项' in w.task_count.text()
    w.editor.fields['tcn_mode'].setCurrentIndex(w.editor.fields['tcn_mode'].findData('four_way'))
    assert '4项' in w.task_count.text()
    assert w.controls_config().tcn_spec['channels']==[64]*6
    assert w.preview_signal.count()==1
    assert not any(x.isVisible() and x.text()=='第二阶段后续开放' for x in w.findChildren(W.QPushButton))
    rr=[load_record(fixture_csv(tmp_path/'raw',c,i,speed=100),signals=('by',)) for c in ('布A','布B') for i in range(6)]
    w.add_records(rr,[]);request=w.make_training_request()
    assert request['config']['architectures']==('tcn',) and request['config']['signals']==('by',)
    assert request['config']['tcn_mode']=='four_way'
    w.close()


def test_parameter_editor_changes_real_spec_and_restores_custom_mode(app,tmp_path):
    from test_tcn_helpers import TCNTrainerWindow
    w=TCNTrainerWindow(tmp_path/'models');e=w.editor
    e.fields['width'].setValue(96);e.fields['depth'].setValue(3);e.fields['kernel'].setCurrentIndex(e.fields['kernel'].findData(3))
    e.fields['dilations'].setText('1,3,9');e.fields['pooling'].setCurrentIndex(e.fields['pooling'].findData('mean_std'))
    e.fields['grad_clip_norm'].setValue(.5);e.fields['scheduler_factor'].setValue(.3)
    cfg=w.controls_config()
    assert cfg.tcn_spec['channels']==[96]*3 and cfg.tcn_spec['dilations']==[1,3,9]
    assert cfg.tcn_spec['pooling']=='mean_std' and cfg.grad_clip_norm==.5 and cfg.scheduler_factor==.3
    e.fields['tcn_mode'].setCurrentIndex(1);assert w.controls_config().tcn_spec['channels']==[64]*6
    e.fields['tcn_mode'].setCurrentIndex(0);assert w.controls_config().tcn_spec['channels']==[96]*3
    e.fields['channels'].setText('16,32')
    with pytest.raises(ValueError):w.controls_config()
    w.close()


@pytest.mark.parametrize('size',[(1024,640),(1280,720),(1366,768),(1920,1080)])
def test_training_controls_are_not_squeezed_at_supported_sizes(app,tmp_path,size):
    from test_tcn_helpers import TCNTrainerWindow
    w=TCNTrainerWindow(tmp_path/'models');w.resize(*size);w.navigation.setCurrentRow(1);w.show();app.processEvents()
    page=w.pages.currentWidget()
    assert w.start_button.isVisible() and w.cancel_button.isVisible()
    assert page.rect().contains(w.start_button.mapTo(page,C.QPoint(0,0)))
    for text in page.findChildren(W.QLabel):
        if text.isVisible() and text.text():
            need=text.heightForWidth(text.width()) if text.hasHeightForWidth() else text.sizeHint().height()
            assert text.height()+2>=need,(text.text(),text.height(),need)
    assert w.parameter_scroll.verticalScrollBar().maximum()>0
    w.close()


def test_incompatible_old_project_cannot_submit_hidden_models(app,tmp_path):
    from test_tcn_helpers import TCNTrainerWindow
    from cloth_trainer.modern_config import ModernConfig
    path=tmp_path/'old.json';value=dict(format='cloth-project-v3',config=ModernConfig().to_dict(),store=str(tmp_path/'models'),sources=[])
    write_json(path,value)
    w=TCNTrainerWindow(tmp_path/'models');w.read_project(path)
    assert w.project_read_only
    with pytest.raises(ValueError,match='只读|TCN'):w.make_training_request()
    assert read_json(path)==json.loads(json.dumps(value))
    w.close()


def test_focused_background_training_and_final_test(app,tmp_path,monkeypatch):
    from test_tcn_helpers import TCNTrainerWindow
    w=TCNTrainerWindow(tmp_path/'models')
    w.editor.fields['epochs'].setValue(1);w.editor.fields['width'].setValue(32);w.editor.fields['depth'].setValue(2)
    rr=[load_record(fixture_csv(tmp_path/'raw',c,i,speed=100),signals=('by',)) for c in ('布A','布B') for i in range(6)]
    w.add_records(rr,[])
    original=w.controls_config
    monkeypatch.setattr(w,'controls_config',lambda:replace(original(),device='cpu',workers=0))
    wait(app,lambda:w.start_button.isEnabled());w.start_training();wait(app,lambda:w.result_task is not None and not w.busy())
    assert w.result_task['status']=='completed' and len(w.result_task['models'])==1
    assert w.comparison.rowCount()==1 and '32' in w.comparison.item(0,0).text()
    assert w.window_table.rowCount()==sum(len(r['windows']) for r in w.result_model['val_predictions'])
    w.run_final();wait(app,lambda:bool(w.result_task.get('final_model')) and not w.busy());assert w.result_task['final_model']
    w.run_test();wait(app,lambda:not w.busy());assert w.result_task['holdout']['metrics']['total']==4
    w.close()


def test_many_class_plots_have_room_for_readable_labels(app):
    import numpy as np
    from cloth_trainer.plots import Plot
    plot=Plot();labels=[f'布料名称{i}' for i in range(15)]
    plot.confusion(np.eye(15,dtype=int),labels)
    assert plot.minimumHeight()>=400
    plot.classes(dict(per_class=[dict(label=name,recall=.5) for name in labels]))
    assert plot.minimumHeight()>=400
    plot.close()


def test_initial_show_uses_actual_content_width_for_splitter(app,tmp_path):
    from test_tcn_helpers import TCNTrainerWindow
    w=TCNTrainerWindow(tmp_path/'models');w.navigation.setCurrentRow(1);w.show();app.processEvents()
    assert w.pages.width()>=740
    assert w.training_splitter.orientation()==C.Qt.Horizontal
    w.close()


def test_update_count_axis_and_export_use_recorded_optimizer_steps(app,tmp_path):
    from cloth_trainer.plots import Plot
    plot=Plot()
    history=[dict(epoch=i,optimizer_updates=i*7,loss=1/i,val_loss=1.1/i,val_f1=.1*i,loss_kind='window CE') for i in (1,2)]
    plot.training(history,x_axis='updates')
    assert list(plot.figure.axes[0].lines[0].get_xdata())==[7,14]
    assert '更新' in plot.figure.axes[0].get_xlabel()
    target=tmp_path/'updates.png';plot.figure.savefig(target);assert target.stat().st_size>0
    plot.training([{k:v for k,v in row.items() if k!='optimizer_updates'} for row in history],x_axis='updates')
    assert list(plot.figure.axes[0].lines[0].get_xdata())==[1,2]
    assert '轮' in plot.figure.axes[0].get_xlabel()
    plot.empty('尚未训练最终模型');plot.set_training_axis('updates')
    assert not plot.figure.axes[0].lines
    plot.close()


def test_preview_counts_actual_parent_trips_and_windows(app,tmp_path):
    from test_tcn_helpers import TCNTrainerWindow
    w=TCNTrainerWindow(tmp_path/'models')
    rr=[load_record(fixture_csv(tmp_path/'raw',c,i,speed=100),signals=('by',)) for c in ('布A','布B') for i in range(6)]
    w.add_records(rr,[]);w.editor.fields['window_size'].setValue(128)
    expected=sum((len(t.values)+127)//128 for r in rr for t in r.trials)
    preview=w.candidate_preview.toPlainText()
    assert '24条原始单程' in preview and f'{expected}个窗口' in preview
    assert '0.38秒' in preview
    w.navigation.setCurrentRow(2);w.show();app.processEvents()
    assert any(b.isVisible() and b.text()=='导出图表' for b in w.pages.currentWidget().findChildren(W.QPushButton))
    w.close()
