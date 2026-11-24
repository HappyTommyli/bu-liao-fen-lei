from dataclasses import replace
from test_ui import app,wait,dispose_test_windows
from test_tcn_helpers import TCNTrainerWindow,legacy_focused_config as focused_config


def test_focused_defaults_are_one_run_and_overlap_is_independent(app,tmp_path):
    cfg=focused_config()
    assert cfg.cv_folds==1 and cfg.tcn_spec['channels']==[128]*6
    assert cfg.tcn_spec['pooling']=='mean_std'
    w=TCNTrainerWindow(tmp_path/'models');e=w.editor
    assert '1项' in w.task_count.text()
    e.fields['tcn_stride'].setValue(450)
    assert w.controls_config().tcn_stride==450
    assert '450' in w.candidate_preview.toPlainText()
    e.fields['cv_folds'].setValue(3)
    assert w.controls_config().cv_folds==3 and '3项' in w.task_count.text()
    assert '重叠50%' in w.candidate_preview.toPlainText()
    w.close()


def test_online_controls_fit_narrow_window_without_horizontal_scrolling(app,tmp_path):
    w=TCNTrainerWindow(tmp_path/'models');w.resize(1024,640);w.navigation.setCurrentRow(3);w.show();app.processEvents()
    assert w.online_scroll.horizontalScrollBar().maximum()==0
    assert w.online_scroll.widget().objectName()=='tcnOnline'
    assert not w.online_worker
    w.close()


def test_overlap_pair_is_selectable_and_submits_two_shared_groups(app,tmp_path):
    from cloth_trainer.tcn_study import candidates
    w=TCNTrainerWindow(tmp_path/'models');w.navigation.setCurrentRow(1);w.show();e=w.editor;mode=e.fields['tcn_mode']
    index=mode.findData('overlap_pair')
    assert index>=0
    mode.setCurrentIndex(index)
    cfg=w.controls_config();a,b=candidates(cfg)
    assert cfg.tcn_stride==450 and '2项' in w.task_count.text()
    assert e.input_group.isVisible() and e.model_group.isVisible()
    assert not e.fields['input_mode'].isEnabled()
    e.fields['width'].setValue(64);e.fields['tcn_stride'].setValue(225)
    cfg=w.controls_config();a,b=candidates(cfg)
    assert a['spec']['channels']==b['spec']['channels']==[64]*6 and b['window_stride']==225
    assert '重叠75%' in w.candidate_preview.toPlainText()
    e.fields['tcn_stride'].setValue(1);e.fields['lr'].setValue(.0004)
    assert w.controls_config().tcn_stride==1
    e.fields['window_size'].setValue(128)
    e.set_config(replace(cfg,tcn_stride=450))
    assert w.controls_config().tcn_stride==450 and w.controls_config().tcn_window==900
    e.fields['cv_folds'].setValue(3);assert '6项' in w.task_count.text()
    w.close()
