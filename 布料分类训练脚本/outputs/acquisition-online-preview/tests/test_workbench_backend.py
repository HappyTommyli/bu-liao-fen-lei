import csv
import json
import tempfile
import time
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from xz_workbench.backend import Worker
from xz_workbench.model import Recipe, Step
from xz_control.protocol import Simulator


class BrokenSensor:
    @property
    def in_waiting(self): raise OSError('sensor unplugged')
    def read(self,n): raise OSError('sensor unplugged')
    def close(self): pass


class FirmwarePort:
    """Serial boundary using the real protocol simulator and newline JSON frames."""
    def __init__(self,split_delay=0):
        self.sim=Simulator(); self.buffer=bytearray(); self.writes=[]
        self.last=time.monotonic(); self.closed=False
        self.split_delay=split_delay; self.status_due=None
    def emit(self,packets):
        for p in packets: self.buffer.extend((json.dumps(p)+'\n').encode())
    @property
    def in_waiting(self):
        now=time.monotonic()
        if self.status_due is not None:
            if now>=self.status_due: self.emit([self.sim.s.copy()]); self.status_due=None
        else: self.emit(self.sim.tick(now-self.last))
        self.last=now
        return len(self.buffer)
    def read(self,n):
        data=bytes(self.buffer[:n]); del self.buffer[:n]; return data
    def write(self,data):
        line=data.decode().strip(); self.writes.append(line)
        if line=='HELLO':
            self.emit([dict(type='hello',protocol='XZ1',move_accel=True,jog_continuous=True,numeric_ranges=True)])
            if self.split_delay: self.status_due=time.monotonic()+self.split_delay
            else: self.emit([self.sim.s.copy()])
        else: self.emit(self.sim.command(line))
        return len(data)
    def close(self): self.closed=True


class SensorPort:
    """A timed raw Hall stream with the firmware's actual REBOOT/READY exchange."""
    def __init__(self,reboot=True,disconnect=False,startup=False):
        self.buffer=bytearray(b'# MLX90393 monitor fixed90 @ XIAO ESP32-S3\nREADY\n' if startup else b'')
        self.stamp=1000000; self.last=0.; self.reboot=reboot; self.disconnect=disconnect
        self.boot_due=None; self.writes=[]; self.closed=False
    @property
    def in_waiting(self):
        now=time.monotonic()
        if self.boot_due is not None and now>=self.boot_due:
            if self.disconnect: raise OSError('USB reset')
            self.buffer.extend(b'# MLX90393 monitor fixed90 @ XIAO ESP32-S3\nREADY\n')
            self.stamp=0; self.boot_due=None
        if now-self.last>=.02:
            self.last=now
            for _ in range(5):
                self.stamp+=4000; self.buffer.extend(f'{self.stamp},100,20,30\n'.encode())
        return len(self.buffer)
    def read(self,n):
        data=bytes(self.buffer[:n]); del self.buffer[:n]; return data
    def write(self,data):
        self.writes.append(data)
        if data==b'REBOOT\n' and self.reboot:
            self.buffer.extend(b'# fw=fixed90 reboot requested\n'); self.boot_due=time.monotonic()+.15
        return len(data)
    def reset_input_buffer(self): self.buffer.clear()
    def close(self): self.closed=True


def until(predicate,seconds=4):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        if predicate(): return
        time.sleep(.015)
    raise AssertionError('等待工作台状态超时')


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.settings=tempfile.TemporaryDirectory(); self.addCleanup(self.settings.cleanup)
        self.worker=Worker(settings_path=Path(self.settings.name)/'settings.json'); self.worker.start()
        self.worker.submit('connect_motor',port='DEMO')
        until(lambda:self.worker.snapshot().get('ready'))
        self.worker.submit('zero',axis='ALL')
        until(lambda:(self.worker.snapshot().get('status') or {}).get('refx')==1)
        self.worker.submit('arm')
        until(lambda:(self.worker.snapshot().get('status') or {}).get('armed')==1)

    def tearDown(self): self.worker.shutdown()

    def sensor(self):
        self.worker.submit('connect_sensor',port='DEMO',baud=115200)
        until(lambda:self.worker.snapshot().get('sensor_alive'))

    def test_restart_then_baseline_can_start_recording_without_previous_baseline(self):
        self.sensor()
        with tempfile.TemporaryDirectory() as td:
            recipe=Recipe('fresh',[Step('restart_sensor',duration=1),Step('baseline',duration=.15),Step('relative','X',.1)])
            self.worker.submit('run_record',root=td,label='棉布',recipe=recipe)
            until(lambda:self.worker.snapshot().get('owner')=='recipe')
            self.assertFalse(list(Path(td).glob('*/metadata.json')))
            until(lambda:self.worker.snapshot().get('saved_path'))
            meta=list(Path(td).glob('*/metadata.json')); self.assertEqual(len(meta),1)
            m=json.loads(meta[0].read_text()); self.assertEqual(m['outcome'],'completed')
            self.assertEqual(m['label'],'棉布'); self.assertGreater(m['sample_count'],0)
            self.assertEqual(self.worker.snapshot()['status']['refx'],1)

    def test_capture_restart_requires_a_new_baseline_before_motion(self):
        self.sensor(); self.worker.submit('baseline',seconds=.15)
        until(lambda:self.worker.snapshot().get('baseline') is not None)
        with tempfile.TemporaryDirectory() as td:
            self.worker.submit('run_record',root=td,label='A',recipe=Recipe('bad',[Step('restart_sensor'),Step('relative','X',.1)]))
            until(lambda:'基线' in self.worker.snapshot().get('notice',''))
            self.assertEqual(self.worker.snapshot()['status']['x'],0)
            self.assertFalse(list(Path(td).glob('*/metadata.json')))

    def test_restart_splits_capture_into_two_numbered_files(self):
        self.sensor(); self.worker.submit('baseline',seconds=.15)
        until(lambda:self.worker.snapshot().get('baseline') is not None)
        with tempfile.TemporaryDirectory() as td:
            recipe=Recipe('segments',[Step('relative','X',.05),Step('restart_sensor',duration=1),Step('baseline',duration=.15),Step('relative','X',.05)])
            self.worker.submit('run_record',root=td,label='B',recipe=recipe)
            until(lambda:self.worker.snapshot().get('saved_path') and self.worker.snapshot().get('owner') is None)
            until(lambda:len(list(Path(td).glob('*/metadata.json')))==2 and all(json.loads(p.read_text())['outcome']=='completed' for p in Path(td).glob('*/metadata.json')))
            self.assertEqual({p.parent.name for p in Path(td).glob('*/metadata.json')},{'B_001','B_002'})
            metas=[json.loads(p.read_text()) for p in Path(td).glob('*/metadata.json')]
            self.assertEqual(len({m['context']['capture_session'] for m in metas}),1)
            self.assertTrue(all(m['sample_count']>0 for m in metas))

    def test_stop_during_restart_cancels_future_motion_and_capture(self):
        self.sensor()
        with tempfile.TemporaryDirectory() as td:
            self.worker.submit('run_record',root=td,label='C',recipe=Recipe('cancel',[Step('restart_sensor',duration=1),Step('baseline',duration=.15),Step('relative','X',1)]))
            until(lambda:self.worker.snapshot().get('restarting'))
            self.worker.submit('stop'); until(lambda:self.worker.snapshot().get('owner') is None)
            time.sleep(.3)
            self.assertEqual(self.worker.snapshot()['status']['x'],0)
            self.assertFalse(self.worker.snapshot()['recording'])
            self.assertFalse(list(Path(td).glob('*/metadata.json')))

    def test_live_values_are_delta_and_baseline_clears_history(self):
        self.sensor(); self.assertIsNone(self.worker.snapshot().get('sensor_values'))
        self.worker.submit('baseline',seconds=.15)
        until(lambda:self.worker.snapshot().get('baseline') is not None)
        until(lambda:self.worker.snapshot().get('sensor_values') is not None)
        values=self.worker.snapshot()['sensor_values']; self.assertLess(abs(values[0]),5)

    def test_real_sensor_reboot_ignores_old_stream_and_waits_for_ready(self):
        port=SensorPort()
        with patch('xz_workbench.backend.open_port',return_value=port):
            self.worker.submit('connect_sensor',port='/dev/test-hall')
            until(lambda:self.worker.snapshot().get('sensor_alive'))
            self.worker.submit('run',recipe=Recipe('boot',[Step('restart_sensor',duration=1),Step('baseline',duration=.15),Step('relative','X',.05)]))
            until(lambda:bool(port.writes)); time.sleep(.06)
            self.assertEqual(self.worker.snapshot()['status']['x'],0)
            self.assertTrue(self.worker.snapshot().get('restarting'))
            until(lambda:self.worker.snapshot()['status']['x']==12 and self.worker.snapshot().get('owner') is None)
            self.assertEqual(port.writes,[b'REBOOT\n']); self.assertIsNotNone(self.worker.snapshot()['baseline'])
            self.assertEqual(self.worker.snapshot()['status']['refx'],1)

    def test_unresponsive_reboot_aborts_without_closing_motor(self):
        port=SensorPort(reboot=False)
        with patch('xz_workbench.backend.open_port',return_value=port):
            self.worker.submit('connect_sensor',port='/dev/test-hall')
            until(lambda:self.worker.snapshot().get('sensor_alive'))
            self.worker.submit('run',recipe=Recipe('timeout',[Step('restart_sensor',duration=.2),Step('relative','X',1)]))
            until(lambda:'重启超时' in self.worker.snapshot().get('notice',''))
            self.assertTrue(self.worker.snapshot()['ready']); self.assertIsNone(self.worker.snapshot()['baseline'])
            self.assertEqual(self.worker.snapshot()['status']['x'],0)

    def test_expected_usb_disconnect_reconnects_sensor_only(self):
        first=SensorPort(disconnect=True); second=SensorPort(startup=True)
        with patch('xz_workbench.backend.open_port',side_effect=[first,second]):
            self.worker.submit('connect_sensor',port='/dev/test-hall')
            until(lambda:self.worker.snapshot().get('sensor_alive'))
            self.worker.submit('run',recipe=Recipe('usb',[Step('restart_sensor',duration=2),Step('baseline',duration=.15),Step('relative','X',.05)]))
            until(lambda:self.worker.snapshot()['status']['x']==12 and self.worker.snapshot().get('owner') is None)
            self.assertTrue(first.closed); self.assertFalse(second.closed)
            self.assertEqual(self.worker.snapshot()['status']['refx'],1)

    def test_wait_before_first_baseline_does_not_need_an_old_baseline(self):
        self.sensor()
        with tempfile.TemporaryDirectory() as td:
            self.worker.submit('run_record',root=td,label='wait',recipe=Recipe('prelude',[Step('wait',duration=.05),Step('baseline',duration=.15),Step('relative','X',.05)]))
            until(lambda:self.worker.snapshot().get('saved_path'))
            self.assertEqual(self.worker.snapshot()['status']['x'],12)

    def test_motor_heartbeat_continues_during_sensor_reboot_wait(self):
        self.worker.submit('disconnect_motor'); until(lambda:not self.worker.snapshot().get('ready'))
        motor=FirmwarePort(); sensor=SensorPort(reboot=False)
        with patch('xz_workbench.backend.open_port',side_effect=lambda port,baud:motor if port=='/dev/test-motor' else sensor):
            self.worker.submit('connect_motor',port='/dev/test-motor')
            until(lambda:self.worker.snapshot().get('ready') and not self.worker.snapshot().get('pending'))
            self.worker.submit('zero',axis='ALL'); until(lambda:self.worker.snapshot()['status']['refx']==1)
            self.worker.submit('arm'); until(lambda:self.worker.snapshot()['status']['armed']==1)
            self.worker.submit('connect_sensor',port='/dev/test-hall'); until(lambda:self.worker.snapshot().get('sensor_alive'))
            self.worker.submit('run',recipe=Recipe('heartbeat',[Step('restart_sensor',duration=.8)]))
            until(lambda:self.worker.snapshot().get('restarting')); before=motor.writes.count('PING')
            time.sleep(.55)
            self.assertGreaterEqual(motor.writes.count('PING')-before,2)
            self.assertTrue(self.worker.snapshot()['ready']); self.assertEqual(self.worker.snapshot()['status']['refx'],1)
            self.assertNotIn('REBOOT',motor.writes)

    def test_duplicate_run_record_request_does_not_abort_first_capture(self):
        self.sensor(); self.worker.submit('baseline',seconds=.15)
        until(lambda:self.worker.snapshot().get('baseline') is not None)
        with tempfile.TemporaryDirectory() as td:
            recipe=Recipe('repeat',[Step('relative','X',.2)])
            self.worker.submit('run_record',root=td,label='repeat',recipe=recipe)
            self.worker.submit('run_record',root=td,label='repeat',recipe=recipe)
            until(lambda:'请先连接' in self.worker.snapshot().get('notice',''))
            self.assertTrue(self.worker.snapshot()['recording'])
            self.assertIn(self.worker.snapshot()['owner'],('preparing','recipe'))
            until(lambda:self.worker.snapshot().get('saved_path'))
            self.assertEqual(self.worker.snapshot()['status']['x'],50)
            self.assertEqual(len(list(Path(td).glob('*/metadata.json'))),1)

    def test_delayed_segment_save_failure_aborts_its_remaining_recipe(self):
        from xz_workbench.recording import Recorder
        self.sensor(); self.worker.submit('baseline',seconds=.15)
        until(lambda:self.worker.snapshot().get('baseline') is not None)
        original=Recorder.close; release=threading.Event(); closing=threading.Event()
        def close(recorder,*args,**kwargs):
            result=original(recorder,*args,**kwargs)
            if recorder.path.name=='failure_001':
                closing.set(); release.wait(3); raise OSError('save failed')
            return result
        with tempfile.TemporaryDirectory() as td,patch.object(Recorder,'close',new=close):
            try:
                recipe=Recipe('error',[Step('relative','X',.05),Step('restart_sensor',duration=1),Step('baseline',duration=1),Step('relative','X',1)])
                self.worker.submit('run_record',root=td,label='failure',recipe=recipe)
                until(lambda:closing.is_set() and self.worker.snapshot().get('calibrating'))
                release.set()
                until(lambda:not self.worker.snapshot().get('recording'),seconds=.4)
                self.assertIsNone(self.worker.snapshot()['owner'])
                self.assertEqual(self.worker.snapshot()['status']['x'],12)
                self.assertTrue(self.worker.snapshot()['ready'])
            finally: release.set()

    def test_delayed_old_session_failure_does_not_abort_new_capture(self):
        from xz_workbench.recording import Recorder
        self.sensor(); self.worker.submit('baseline',seconds=.15)
        until(lambda:self.worker.snapshot().get('baseline') is not None)
        original=Recorder.close; release=threading.Event(); closing=threading.Event()
        def close(recorder,*args,**kwargs):
            result=original(recorder,*args,**kwargs)
            if recorder.path.name=='old_001':
                closing.set(); release.wait(3); raise OSError('old save failed')
            return result
        with tempfile.TemporaryDirectory() as td,patch.object(Recorder,'close',new=close):
            try:
                self.worker.submit('record_start',root=td,label='old')
                until(lambda:self.worker.snapshot().get('recording'))
                self.worker.submit('record_stop'); until(closing.is_set)
                self.worker.submit('record_start',root=td,label='new')
                until(lambda:self.worker.snapshot().get('recording'))
                release.set(); time.sleep(.15)
                self.assertTrue(self.worker.snapshot()['recording'])
                self.assertEqual(self.worker.capture['label'],'new')
                self.worker.submit('record_stop'); until(lambda:not self.worker.snapshot().get('recording'))
                until(lambda:not any(t.is_alive() for t in self.worker.finalizers))
            finally: release.set()

    def test_demo_motion_never_opens_serial_and_finishes_recipe(self):
        with patch('serial.Serial',side_effect=AssertionError('demo opened hardware')):
            self.worker.submit('run',recipe=Recipe('demo',[Step('relative','X',.1)]))
            until(lambda:(self.worker.snapshot().get('status') or {}).get('x')==25 and self.worker.snapshot().get('owner') is None)

    def test_function_recording_waits_for_sensor_and_preserves_csv(self):
        self.worker.submit('connect_sensor',port='DEMO',baud=115200)
        until(lambda:self.worker.snapshot().get('sensor_alive'))
        self.worker.submit('baseline',seconds=.12)
        until(lambda:self.worker.snapshot().get('baseline') is not None)
        with tempfile.TemporaryDirectory() as td:
            self.worker.submit('run_record',root=td,recipe=Recipe('small',[Step('relative','X',.1)]))
            until(lambda:list(Path(td).glob('*/metadata.json')) and not self.worker.snapshot().get('recording') and (self.worker.snapshot().get('status') or {}).get('x')==25)
            until(lambda:any('completed' in p.read_text() for p in Path(td).glob('*/metadata.json')))
            path=next(Path(td).glob('*/*.csv'))
            with path.open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
            self.assertGreater(len(rows),1)
            self.assertIn('step_001',{r['stage'] for r in rows})
            self.assertTrue(all(r['motor_fault']=='NONE' for r in rows))

    def test_missing_sensor_prevents_run_and_capture(self):
        with tempfile.TemporaryDirectory() as td:
            self.worker.submit('run_record',root=td,recipe=Recipe('x',[Step('relative','X',1)]))
            until(lambda:bool(self.worker.snapshot().get('notice')))
            self.assertEqual(self.worker.snapshot()['status']['x'],0)
            self.assertFalse(list(Path(td).iterdir()))

    def test_quick_press_and_release_cannot_start_late_jog(self):
        self.worker.submit('jog',axis='X',direction=1)
        self.worker.submit('release')
        until(lambda:self.worker.snapshot().get('owner') is None)
        time.sleep(.1)
        self.assertIsNone(self.worker.snapshot().get('owner'))

    def test_failed_baseline_keeps_devices_connected(self):
        self.worker.submit('connect_sensor',port='DEMO',baud=115200)
        until(lambda:self.worker.snapshot().get('sensor_alive'))
        self.worker.submit('baseline',seconds=.001)
        until(lambda:'不足' in self.worker.snapshot().get('notice',''))
        self.assertTrue(self.worker.snapshot()['ready'])
        self.assertTrue(self.worker.snapshot()['sensor_alive'])

    def test_unrecorded_sensor_failure_does_not_interrupt_motor(self):
        with patch('xz_workbench.backend.open_port',return_value=BrokenSensor()):
            self.worker.submit('connect_sensor',port='/dev/test-sensor',baud=115200)
            until(lambda:'sensor unplugged' in self.worker.snapshot().get('notice',''))
            self.assertTrue(self.worker.snapshot()['ready'])
            self.assertEqual(self.worker.snapshot()['status']['armed'],1)

    def test_real_transport_handshake_and_movement_use_one_serial_owner(self):
        self.worker.submit('disconnect_motor')
        until(lambda:not self.worker.snapshot().get('ready'))
        port=FirmwarePort()
        with patch('xz_workbench.backend.open_port',return_value=port):
            self.worker.submit('connect_motor',port='/dev/test-motor')
            until(lambda:self.worker.snapshot().get('ready') and self.worker.snapshot().get('pending') is None)
            self.assertIn('HELLO',port.writes)
            self.assertIn('DISARM',port.writes)
            self.assertEqual(self.worker.snapshot()['status']['armed'],0)
            self.worker.submit('zero',axis='ALL')
            until(lambda:self.worker.snapshot()['status']['refx']==1)
            self.worker.submit('arm')
            until(lambda:self.worker.snapshot()['status']['armed']==1)
            self.worker.submit('run',recipe=Recipe('short',[Step('relative','X',.1)]))
            until(lambda:self.worker.snapshot()['status']['x']==25 and self.worker.snapshot()['owner'] is None)
            self.assertTrue(any(c.startswith('MOVE ABS') for c in port.writes))

    def test_hello_and_initial_status_can_arrive_in_separate_reads(self):
        self.worker.submit('disconnect_motor'); until(lambda:not self.worker.snapshot().get('ready'))
        port=FirmwarePort(split_delay=.2)
        with patch('xz_workbench.backend.open_port',return_value=port):
            self.worker.submit('connect_motor',port='/dev/test-motor')
            until(lambda:'HELLO' in port.writes)
            time.sleep(.1)
            self.assertFalse(port.closed,'完整握手尚在进行，不应因 status 晚到关闭串口')
            until(lambda:self.worker.snapshot().get('ready'))


class RebootDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.worker=Worker(); self.worker.sim=Simulator()
        state=self.worker.sim.s.copy(); state.update(armed=1,refx=1,refz=1)
        self.worker.engine.connect(state)
        self.worker.sensor=SensorPort(reboot=False); self.worker.sensor_port='/dev/test-hall'
        self.worker.engine.run(Recipe('diagnose',[Step('restart_sensor',duration=1)]))
    def timeout(self):
        self.worker._tick_external(self.worker.external['deadline']+.01)
        return self.worker.notice
    def test_board_identity_is_not_required_to_send_reboot(self):
        self.assertIsNone(self.worker.sensor_identity)
        self.assertEqual(self.worker.sensor.writes,[b'REBOOT\n'])
    def test_timeout_distinguishes_old_stream_without_command_ack(self):
        self.worker._restart_bytes(b'90000000,100,20,30\n90004000,101,20,30\n')
        reason=self.timeout()
        self.assertIn('未收到 REBOOT',reason); self.assertIn('仍收到',reason)
        self.assertIn('/dev/test-hall',reason)
    def test_timeout_after_ack_reports_missing_ready(self):
        self.worker._restart_bytes(b'# fw=fixed90 reboot requested\n')
        reason=self.timeout()
        self.assertIn('已收到 REBOOT',reason); self.assertIn('READY',reason)
        self.assertNotIn('未收到 REBOOT',reason)
    def test_timeout_after_ready_reports_missing_sensor_frame(self):
        self.worker._restart_bytes(b'# fw=fixed90 reboot requested\nREADY\n')
        reason=self.timeout()
        self.assertIn('已收到 READY',reason); self.assertIn('有效磁数据',reason)
    def test_sensor_initialization_failure_is_in_timeout_diagnostic(self):
        self.worker._restart_bytes(b'# MLX90393 monitor fixed90 @ XIAO ESP32-S3\n# fw=fixed90 init failed; retry in 2s\n')
        reason=self.timeout()
        self.assertIn('init failed',reason); self.assertIn('READY',reason)
    def test_plain_error_response_is_preserved_in_timeout(self):
        self.worker._restart_bytes(b'ERR unknown command REBOOT\n')
        self.assertIn('ERR unknown command REBOOT',self.timeout())
    def test_reconnect_open_error_is_preserved_in_timeout(self):
        self.worker.sensor=None; self.worker.external['disconnected']=True
        with patch('xz_workbench.backend.open_port',side_effect=OSError('permission denied')):
            self.worker._tick_external(self.worker.external['requested']+.3)
        reason=self.timeout()
        self.assertIn('未重新连接',reason); self.assertIn('permission denied',reason)

    def test_disconnect_after_ready_reports_current_reconnect_failure(self):
        self.worker._restart_bytes(b'# fw=fixed90 reboot requested\nREADY\n')
        self.worker.sensor=None; self.worker.external['disconnected']=True
        with patch('xz_workbench.backend.open_port',side_effect=OSError('permission denied')):
            self.worker._tick_external(self.worker.external['requested']+.3)
        reason=self.timeout()
        self.assertIn('未重新连接',reason); self.assertIn('permission denied',reason)
        self.assertNotIn('请检查传感器初始化和输出格式',reason)

    def test_c3_firmware_reboot_ack_boot_banner_and_stream_are_accepted(self):
        frames=self.worker._restart_bytes(b'# fw=mlx_c3 reboot requested\n# MLX90393 monitor-compat @ XIAO ESP32-C3 (S3-style GPIO1/GPIO2)\n# fw=mlx_c3 build=reboot-v1 target=XIAO_ESP32C3\nREADY\n200000,100,20,30\n')
        self.assertTrue(self.worker.external['ack']); self.assertTrue(self.worker.external['boot'])
        self.assertTrue(self.worker.external['ready'])
        self.assertEqual([frame.values for frame in frames],[(100.,20.,30.)])

    def test_phase_logs_expose_ack_and_ready(self):
        self.worker._restart_bytes(b'# fw=fixed90 reboot requ')
        self.worker._restart_bytes(b'ested\nREADY\n')
        events=[]
        while not self.worker.events.empty(): events.append(self.worker.events.get_nowait())
        logs=[e.get('message','') for e in events if e['type']=='log']
        self.assertTrue(any('REBOOT 确认' in line for line in logs))
        self.assertTrue(any('READY' in line for line in logs))



class PreparationTests(unittest.TestCase):
    def test_waiting_for_first_sample_reserves_motion_owner(self):
        settings=tempfile.TemporaryDirectory(); self.addCleanup(settings.cleanup)
        worker=Worker(settings_path=Path(settings.name)/'settings.json'); state=Simulator().s.copy(); state.update(armed=1,refx=1,refz=1)
        worker.sensor_port='DEMO'; worker.engine.connect(state); worker.baseline=(0,0,0); worker.last_sensor=time.monotonic()
        worker.sim=Simulator(); worker.sim.s.update(state)
        worker._handle('calibrate_origin',{'origin_name':'测试固定基准'})
        worker._receive_motor(time.monotonic(),0)
        with tempfile.TemporaryDirectory() as td:
            try:
                worker._start_record(td,Recipe('pending',[Step('relative','X',1)]))
                self.assertFalse(worker.engine.idle())
                with self.assertRaises(ValueError): worker.engine.move(Step('relative','Z',1))
            finally:
                worker._end_record('aborted','test end'); worker.shutdown()


if __name__=='__main__': unittest.main()
