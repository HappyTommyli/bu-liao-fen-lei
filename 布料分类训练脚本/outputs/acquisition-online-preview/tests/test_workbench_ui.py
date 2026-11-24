import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication
from xz_control.protocol import Simulator
from xz_workbench.model import AxisConfig
from xz_workbench.ui import Window
import queue
import tempfile
from pathlib import Path


class DeviceBoundary:
    def __init__(self):
        self.requests=[]; self.events=queue.Queue(); self.lease_count=0
        status=Simulator().s.copy(); status.update(armed=1,refx=1,refz=1)
        self.state=dict(ready=True,status=status,idle=True,pending=None,owner=None,
                        recording=False,record_owned=False,config=AxisConfig().to_dict(),calibrating=False,
                        manual_reference={'valid':True,'name':'人工机械基准'})
    def submit(self,name,**payload): self.requests.append((name,payload))
    def lease(self): self.lease_count+=1
    def snapshot(self): return self.state.copy()
    def shutdown(self): pass


class UITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.worker=DeviceBoundary(); self.worker.settings_path=Path(self.tmp.name)/'settings.json'; self.window=Window(self.worker)
        self.window.show(); self.window.activateWindow(); self.window.setFocus()
        self.app.processEvents(); self.window.poll()
    def tearDown(self): self.window.close(); self.app.processEvents()
    def key(self,kind,key,repeat=False):
        event=QKeyEvent(kind,key,Qt.NoModifier,'',repeat,1)
        self.app.sendEvent(self.window,event)

    def enable_keyboard(self):
        self.assertTrue(hasattr(self.window,'keyboard_enabled'),'manual keyboard switch is missing')
        self.window.keyboard_enabled.setChecked(True)

    def test_keyboard_requires_opt_in_on_manual_page(self):
        self.assertTrue(hasattr(self.window,'keyboard_enabled'))
        self.assertFalse(self.window.keyboard_enabled.isChecked())
        self.key(QEvent.KeyPress,Qt.Key_Right)
        self.assertNotIn('jog',[n for n,_ in self.worker.requests])
        self.enable_keyboard(); self.key(QEvent.KeyPress,Qt.Key_Right)
        self.assertIn('jog',[n for n,_ in self.worker.requests])

    def test_switching_to_function_page_releases_jog(self):
        self.enable_keyboard(); self.key(QEvent.KeyPress,Qt.Key_Right)
        self.window.workspace_tabs.setCurrentIndex(1); self.app.processEvents()
        self.assertIsNone(self.window.held)
        self.assertIn('release',[n for n,_ in self.worker.requests])
        before=[n for n,_ in self.worker.requests].count('jog')
        self.key(QEvent.KeyPress,Qt.Key_Down); self.window.renew_hold()
        self.assertEqual([n for n,_ in self.worker.requests].count('jog'),before)

    def test_disabling_keyboard_releases_jog(self):
        self.enable_keyboard(); self.key(QEvent.KeyPress,Qt.Key_Right)
        self.window.keyboard_enabled.setChecked(False); self.window.renew_hold()
        self.assertIsNone(self.window.held); self.assertEqual(self.worker.requests[-1][0],'release')

    def test_disabled_keyboard_keeps_spinbox_arrow_editing(self):
        self.assertTrue(hasattr(self.window,'keyboard_enabled'))
        spin=self.window.speeds['x']; spin.setFocus(); self.app.processEvents(); before=spin.value()
        self.app.sendEvent(spin,QKeyEvent(QEvent.KeyPress,Qt.Key_Up,Qt.NoModifier))
        self.assertGreater(spin.value(),before)
        self.assertNotIn('jog',[n for n,_ in self.worker.requests])

    def test_arrows_move_while_spinbox_is_focused_without_changing_value(self):
        self.enable_keyboard()
        spin=self.window.speeds['x']; spin.setFocus(); self.app.processEvents()
        value=spin.value()
        self.app.sendEvent(spin,QKeyEvent(QEvent.KeyPress,Qt.Key_Up,Qt.NoModifier))
        self.assertEqual(spin.value(),value)
        self.assertEqual([p for n,p in self.worker.requests if n=='jog'][-1],{'axis':'Z','direction':-1})
        self.app.sendEvent(spin,QKeyEvent(QEvent.KeyRelease,Qt.Key_Up,Qt.NoModifier))
        self.assertEqual(self.worker.requests[-1][0],'release')

    def test_arrows_in_parameter_panel_never_jog(self):
        self.enable_keyboard()
        self.window.show_panel(self.window.settings_dialog); self.app.processEvents()
        widget=self.window.accels['x']; widget.setFocus()
        self.app.sendEvent(widget,QKeyEvent(QEvent.KeyPress,Qt.Key_Right,Qt.NoModifier))
        self.window.renew_hold()
        self.assertNotIn('jog',[n for n,_ in self.worker.requests])
        self.assertEqual(self.worker.lease_count,0)

    def test_no_baseline_numeric_display_never_shows_raw_as_delta(self):
        self.worker.state.update(sensor_values=None,baseline=None); self.window.poll()
        self.assertIn('待校准',self.window.values.text())
        self.worker.state.update(sensor_values=(1,2,3),baseline=(100,200,300)); self.window.poll()
        self.assertIn('ΔBx 1.000',self.window.values.text())

    def test_new_sensor_actions_can_be_edited_in_table(self):
        from xz_workbench.model import Step
        self.window.replace_rows([Step('restart_sensor',duration=10),Step('baseline',duration=2)])
        self.assertEqual([s.kind for s in self.window.read_recipe().steps],['restart_sensor','baseline'])
        self.assertFalse(self.window.table.cellWidget(0,2).isEnabled())
        self.assertTrue(self.window.table.cellWidget(1,7).isEnabled())

    def test_function_page_arrows_edit_controls_without_jog(self):
        self.enable_keyboard(); self.window.workspace_tabs.setCurrentIndex(1)
        self.window.add_step(); self.window.add_step()
        combo=self.window.table.cellWidget(0,1); combo.setFocus(); self.app.processEvents()
        index=combo.currentIndex()
        self.app.sendEvent(combo,QKeyEvent(QEvent.KeyPress,Qt.Key_Down,Qt.NoModifier))
        self.assertNotEqual(combo.currentIndex(),index)
        self.assertNotIn('jog',[n for n,_ in self.worker.requests])

    def test_stale_sensor_values_without_baseline_are_hidden(self):
        self.worker.state.update(sensor_values=(100,200,300),baseline=None); self.window.poll()
        self.assertNotIn('100.000',self.window.values.text()); self.assertIn('待校准',self.window.values.text())

    def test_ordinary_run_does_not_require_a_fabric_label(self):
        self.window.add_step(); self.window.record_label.clear(); self.window.run_recipe(False)
        self.assertIn('run',[n for n,_ in self.worker.requests])

    def test_keyboard_holds_and_ignores_auto_repeat_release(self):
        self.enable_keyboard()
        self.key(QEvent.KeyPress,Qt.Key_Right)
        self.key(QEvent.KeyPress,Qt.Key_Right,True)
        self.key(QEvent.KeyRelease,Qt.Key_Right,True)
        self.assertEqual([n for n,_ in self.worker.requests].count('jog'),1)
        self.assertNotIn('release',[n for n,_ in self.worker.requests])
        self.key(QEvent.KeyRelease,Qt.Key_Right)
        self.assertEqual(self.worker.requests[-1][0],'release')

    def test_focus_loss_releases_jog(self):
        self.enable_keyboard()
        self.key(QEvent.KeyPress,Qt.Key_Up)
        self.app.sendEvent(self.window,QEvent(QEvent.WindowDeactivate))
        self.assertEqual(self.worker.requests[-1][0],'release')

    def test_mouse_fixed_mode_submits_one_distance_move(self):
        self.window.jog_mode.setCurrentIndex(1)
        self.window.distance.setValue(3)
        self.window.direction_pressed('up','mouse')
        self.window.direction_released('up','mouse')
        moves=[p for n,p in self.worker.requests if n=='move']
        self.assertEqual(len(moves),1)
        self.assertEqual((moves[0]['step'].axis,moves[0]['step'].value),('Z',-3))

    def test_recipe_rows_can_be_reordered_and_disabled(self):
        self.window.add_step(); self.window.add_step()
        self.window.table.cellWidget(0,4).setValue(5)
        self.window.table.cellWidget(1,4).setValue(9)
        self.window.table.selectRow(1); self.window.move_row(-1)
        recipe=self.window.read_recipe()
        self.assertEqual([s.value for s in recipe.steps],[9,5])

    def test_running_locks_zero_and_editor_but_not_stop(self):
        self.worker.state.update(owner='recipe',idle=False)
        self.window.poll()
        self.assertFalse(self.window.zero_x.isEnabled())
        self.assertFalse(self.window.table.isEnabled())
        self.assertTrue(self.window.stop_button.isEnabled())

    def test_connection_settings_open_on_demand(self):
        self.assertFalse(self.window.motor_ports.isVisible())
        self.window.devices_button.click(); self.app.processEvents()
        self.assertTrue(self.window.motor_ports.isVisible())
        self.window.device_dialog.close()

    def test_manual_and_function_pages_are_separate(self):
        self.assertFalse(self.window.table.isVisible())
        self.window.workspace_tabs.setCurrentIndex(1); self.app.processEvents()
        self.assertTrue(self.window.table.isVisible())

    def test_space_stops_while_parameter_panel_is_active(self):
        self.window.show_panel(self.window.settings_dialog)
        self.window.settings_dialog.activateWindow(); self.app.processEvents()
        event=QKeyEvent(QEvent.KeyPress,Qt.Key_Space,Qt.NoModifier)
        self.app.sendEvent(self.window.settings_dialog,event)
        self.assertIn('stop',[name for name,_ in self.worker.requests])


if __name__=='__main__': unittest.main()

class SessionUITests(unittest.TestCase):
    setUpClass=classmethod(UITests.setUpClass.__func__)
    setUp=UITests.setUp
    tearDown=UITests.tearDown
    def test_close_reopen_restores_directory_speeds_and_unsaved_recipe(self):
        from xz_workbench.model import Step
        self.window.output_root=Path(self.tmp.name)/'captures'
        self.window.speeds['x'].setValue(3.25); self.window.accels['z'].setValue(4.75)
        self.window.distance.setValue(2.5); self.window.recipe_name.setText('棉布扫描')
        self.window.replace_rows([Step('relative','X',-2,speed=1.2,label='返回'),Step('wait',duration=.5)])
        self.window.bounds['x'][0].setValue(-20); self.window.bound_enabled.setChecked(True)
        self.window.keyboard_enabled.setChecked(True)
        self.window.close()
        self.window=Window(self.worker)
        self.assertEqual(self.window.output_root,Path(self.tmp.name)/'captures')
        self.assertEqual(self.window.speeds['x'].value(),3.25)
        self.assertEqual(self.window.accels['z'].value(),4.75)
        self.assertEqual(self.window.distance.value(),2.5)
        self.assertEqual(self.window.recipe_name.text(),'棉布扫描')
        self.assertEqual(self.window.read_recipe().steps[0].value,-2)
        self.assertEqual(self.window.read_recipe().steps[0].label,'返回')
        self.assertFalse(self.window.keyboard_enabled.isChecked()); self.assertFalse(self.window.bound_enabled.isChecked())
        self.assertEqual(self.window.bounds['x'][0].value(),-20)

    def test_invalid_settings_start_with_visible_warning(self):
        self.window.close(); self.worker.settings_path.write_text('{bad')
        self.window=Window(self.worker)
        self.assertIn('设置',self.window.notice.text())

    def test_malformed_nested_settings_fall_back_without_crashing(self):
        import json
        self.window.save_session()
        good=json.loads(self.worker.settings_path.read_text())
        self.window.close()
        from copy import deepcopy
        for key in ('bounds','batch'):
            broken=deepcopy(good); broken['session'][key]=[]
            self.worker.settings_path.write_text(json.dumps(broken))
            self.window=Window(self.worker)
            self.assertIn('设置恢复失败',self.window.notice.text())
            self.window.close()

class BatchAndRecycleUITests(unittest.TestCase):
    setUpClass=classmethod(UITests.setUpClass.__func__)
    setUp=UITests.setUp
    tearDown=UITests.tearDown
    def test_batch_controls_submit_and_restore_count_interval(self):
        from xz_workbench.model import Step
        self.window.output_root=Path(self.tmp.name)
        self.window.replace_rows([Step('relative','X',.1)])
        self.worker.state.update(sensor_port='DEMO',sensor_alive=True,baseline=(0,0,0))
        self.window.poll(); self.window.batch_count.setValue(5); self.window.batch_interval.setValue(1.5)
        self.window.run_batch()
        name,payload=self.worker.requests[-1]
        self.assertEqual(name,'batch_record'); self.assertEqual(payload['count'],5); self.assertEqual(payload['interval'],1.5)
        self.window.close(); self.window=Window(self.worker)
        self.assertEqual(self.window.batch_count.value(),5); self.assertEqual(self.window.batch_interval.value(),1.5)
    def test_batch_save_wait_locks_manual_settings_and_shows_progress(self):
        self.worker.state.update(owner='batch_wait',idle=False,batch=dict(active=True,round=2,count=5,completed=1,phase='saving'))
        self.window.poll()
        self.assertFalse(self.window.batch_button.isEnabled()); self.assertFalse(self.window.output_button.isEnabled())
        self.assertFalse(self.window.record_label.isEnabled()); self.assertTrue(self.window.batch_stop_button.isEnabled())
        self.assertIn('2 / 5',self.window.batch_progress.text())
    def test_delete_requires_confirmation_and_submits_selected_records(self):
        from unittest.mock import patch
        from PySide6.QtWidgets import QMessageBox
        import json
        root=Path(self.tmp.name)/'data'; root.mkdir(); self.window.output_root=root
        for name in ('A_001','A_002'):
            p=root/name; p.mkdir(); (p/'A.csv').write_text('csv')
            (p/'metadata.json').write_text(json.dumps(dict(outcome='completed',label='A',files={'sensor':'A.csv'})))
        self.window.refresh_records(); self.window.records_table.selectAll()
        with patch('xz_workbench.ui.QMessageBox.question',return_value=QMessageBox.No): self.window.delete_records()
        self.assertFalse(any(n=='trash_records' for n,_ in self.worker.requests))
        with patch('xz_workbench.ui.QMessageBox.question',return_value=QMessageBox.Yes): self.window.delete_records()
        name,payload=self.worker.requests[-1]; self.assertEqual(name,'trash_records'); self.assertEqual(len(payload['paths']),2)

class RealtimeOptInTests(UITests):
    def test_forward_requires_checkbox_then_start_and_supports_stop(self):
        self.assertFalse(self.window.realtime_enabled.isChecked())
        self.assertFalse(self.window.realtime_start.isEnabled())
        self.window.realtime_enabled.setChecked(True)
        self.window.realtime_start.click()
        self.assertIn(('realtime_forward',{'enabled':True}),self.worker.requests)
        self.worker.state['realtime_forward']={'enabled':True,'clients':1,'dropped':0,'error':''}
        self.window.poll()
        self.window.realtime_stop.click()
        self.assertEqual(self.worker.requests[-1],('realtime_forward',{'enabled':False}))
