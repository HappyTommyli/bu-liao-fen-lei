import json
import unittest
from pathlib import Path
from unittest.mock import patch
from PySide6.QtWidgets import QMessageBox
import test_workbench_ui as ui_tests
from test_workbench_ui import DeviceBoundary
from xz_workbench.ui import Window
from xz_workbench.model import Step

class OriginUITests(unittest.TestCase):
    setUpClass=classmethod(ui_tests.UITests.setUpClass.__func__)
    setUp=ui_tests.UITests.setUp
    tearDown=ui_tests.UITests.tearDown
    def test_origin_confirmation_is_explicit_and_sends_named_reference(self):
        self.window.origin_name.setText('定位块A')
        with patch('xz_workbench.ui.QMessageBox.question',return_value=QMessageBox.No): self.window.confirm_origin()
        self.assertFalse(any(n=='calibrate_origin' for n,_ in self.worker.requests))
        with patch('xz_workbench.ui.QMessageBox.question',return_value=QMessageBox.Yes): self.window.confirm_origin()
        self.assertEqual(self.worker.requests[-1],('calibrate_origin',{'origin_name':'定位块A'}))
    def test_save_current_start_and_restart_preserves_point_without_confirmation(self):
        self.worker.state['status'].update(x=250,z=500)
        self.window.start_clearance.setValue(-1)
        self.window.save_capture_start(True)
        self.assertEqual(self.window.config().start_x,1)
        self.assertEqual(self.window.config().start_z,2)
        self.window.replace_rows([Step('capture_start')])
        self.window.close(); worker=DeviceBoundary(); worker.settings_path=self.worker.settings_path
        worker.state['manual_reference']={'valid':False}; self.worker=worker
        self.window=Window(worker)
        self.assertEqual(self.window.config().clearance_z,-1)
        self.assertEqual(self.window.read_recipe().steps[0].kind,'capture_start')
        self.assertFalse(any(n=='calibrate_origin' for n,_ in worker.requests))
        self.assertFalse(self.window.go_start_button.isEnabled())
    def test_temporary_alignment_blocks_functions_but_keeps_manual_directions(self):
        self.worker.state['manual_reference']={'valid':False}; self.window.poll()
        self.assertFalse(self.window.run_button.isEnabled()); self.assertFalse(self.window.batch_button.isEnabled())
        self.assertTrue(self.window.direction_buttons['right'].isEnabled())
        self.assertFalse(self.window.absolute_button.isEnabled())
    def test_scale_or_origin_name_change_cannot_reuse_saved_point(self):
        self.window.save_capture_start(True)
        self.window.ppmm['x'].setValue(500)
        self.assertIsNone(self.window.config().start_x)
        self.window.ppmm['x'].setValue(250); self.window.origin_name.setText('另外一处基准')
        self.assertIsNone(self.window.config().start_x)

class AppliedScaleUITests(unittest.TestCase):
    setUpClass=classmethod(ui_tests.UITests.setUpClass.__func__)
    setUp=ui_tests.UITests.setUp
    tearDown=ui_tests.UITests.tearDown
    def test_applied_scale_clear_event_removes_persisted_point(self):
        self.window.save_capture_start(True)
        self.worker.events.put({'type':'start_cleared'})
        self.worker.state['manual_reference']={'valid':False}; self.window.poll(); self.window.save_session()
        self.assertIsNone(self.window.saved_start)
        data=json.loads(self.worker.settings_path.read_text())
        self.assertIsNone(data['session']['manual_origin']['point'])
