import subprocess
import unittest
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class C3FirmwareTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.binary=Path(cls.tmp.name)/'c3'
  p=subprocess.run(['g++','-std=c++11','-Wall','-Wextra','-Werror','-Wno-comment','-I',str(ROOT/'tests/c3_fakes'),str(ROOT/'tests/c3_reboot_test.cpp'),'-o',str(cls.binary)],capture_output=True,text=True)
  if p.returncode:raise AssertionError(p.stderr)
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def run_scenario(self,scenario):
  try:p=subprocess.run([str(self.binary),scenario],capture_output=True,text=True,timeout=3)
  except subprocess.TimeoutExpired:self.fail('REBOOT did not interrupt '+scenario)
  self.assertEqual(p.returncode,0,p.stdout+p.stderr)
 def test_known_connected_pins_initialize_without_scanning_unwired_pairs(self):self.run_scenario('known_pins')
 def test_reboot_is_acknowledged_and_flushed_before_restart(self):self.run_scenario('reboot')
 def test_partial_command_and_crlf_are_supported(self):self.run_scenario('partial')
 def test_invalid_or_overlong_input_never_restarts(self):self.run_scenario('invalid')
 def test_existing_pin_probe_and_raw_csv_stream_are_preserved(self):self.run_scenario('stream')
 def test_reboot_can_interrupt_sensor_calibration(self):self.run_scenario('warmup')
