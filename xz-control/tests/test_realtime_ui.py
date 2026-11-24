"""Original UI forwarding opt-in test; fake device boundary, no worker or devices."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import tempfile
import unittest
from pathlib import Path
from PySide6.QtWidgets import QApplication
from tests import test_workbench_ui as fixtures
from xz_workbench.ui import Window


class RealtimeOptInTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_forward_requires_checkbox_then_start_and_supports_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            worker=fixtures.DeviceBoundary();worker.settings_path=Path(folder)/'settings.json'
            window=Window(worker)
            try:
                self.assertFalse(window.realtime_enabled.isChecked())
                self.assertFalse(window.realtime_start.isEnabled())
                self.assertEqual(window.windowTitle(),'XZ 工作台')
                window.realtime_enabled.setChecked(True);window.realtime_start.click()
                self.assertIn(('realtime_forward',{'enabled':True}),worker.requests)
                worker.state['realtime_forward']={'enabled':True,'clients':1,'dropped':0,'error':''}
                window.poll();window.realtime_stop.click()
                self.assertEqual(worker.requests[-1],('realtime_forward',{'enabled':False}))
                window.realtime_enabled.setChecked(False)
                self.assertEqual(worker.requests[-1],('realtime_forward',{'enabled':False}))
            finally:window.close();self.app.processEvents()
