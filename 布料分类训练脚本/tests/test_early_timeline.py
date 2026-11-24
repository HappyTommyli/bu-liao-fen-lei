from PyQt5 import QtCore as C
from test_ui import app,wait,dispose_test_windows
from cloth_trainer.tcn_ui import TCNTrainerWindow


def test_four_pages_default_prefix_and_uncrowded_process_view(app,tmp_path):
    w=TCNTrainerWindow(tmp_path/'models');w.resize(1024,640);w.navigation.setCurrentRow(3);w.show();app.processEvents()
    assert w.navigation.count()==4 and w.pages.count()==4
    cfg=w.controls_config();assert cfg.tcn_mode=='prefix' and cfg.architectures==('cera',) and cfg.cv_folds==1
    assert hasattr(w,'online_timeline') and w.online_stack.currentIndex()==0
    assert w.online_consecutive.value()==2 and w.online_lock.isChecked()
    assert not w.online_advanced.isVisible() and not w.online_formal.isVisible()
    assert w.online_scroll.horizontalScrollBar().maximum()==0
    w.online_receive_event(dict(type='trace',path='x',stage='step_003',points=[[0,60,1],[1,66,2]],labels=['a','b']))
    w.online_receive_event(dict(type='prediction',path='x',stage='step_003',t_s=10,x_mm=80,prediction='a',probability=.99,margin=.98,status='locked',confirmed_label='a',probabilities=[.99,.01],raw_probabilities=[.99,.01],stable_elapsed_s=2.7,sample_count=1800,consecutive_count=2,flip_count=0,completion=dict(t_s=10,x_mm=80,label='a',effective_elapsed_s=5.4)))
    app.processEvents();assert '锁定' in w.online_current.text()
    assert w.online_timeline.completion is not None
    w.close();wait(app,lambda:not w.desktop_jobs.running())


def test_focused_result_plots_defer_hidden_render_until_opened(app,tmp_path):
    w=TCNTrainerWindow(tmp_path/'models');p=w.matrix_plot
    old=p.figure.axes[0]
    p.confusion([[1,0],[0,1]],['a','b'])
    assert p.figure.axes[0] is old and p._pending_render
    w.navigation.setCurrentRow(2);w.result_tabs.setCurrentWidget(p);w.show();app.processEvents()
    assert not p._pending_render and p.figure.axes[0] is not old
    w.close();wait(app,lambda:not w.desktop_jobs.running())


def test_prefix_editor_preserves_growth_interval_larger_than_minimum(app,tmp_path):
    from dataclasses import replace
    w=TCNTrainerWindow(tmp_path/'models');cfg=replace(w.config,tcn_window=128,tcn_stride=512).validate()
    w.config=cfg;w.sync_controls()
    assert w.controls_config().tcn_stride==512
    w.close();wait(app,lambda:not w.desktop_jobs.running())


def test_live_waiting_state_explains_need_to_run_scan(app,tmp_path):
    w=TCNTrainerWindow(tmp_path/'models')
    w.online_receive_event(dict(type='waiting',status='waiting',stage='manual_idle',reason='stage_not_classifiable'))
    assert '等待扫描' in w.online_state.text() and '运行并采集' in w.online_current.text()
    assert '数据错误' not in w.online_state.text()
    w.close();wait(app,lambda:not w.desktop_jobs.running())
