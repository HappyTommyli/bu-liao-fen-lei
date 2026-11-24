"""Offline only: real loopback sockets and decoded frame fixtures, no devices."""
import json
import socket
import time
import unittest
from types import SimpleNamespace
from xz_workbench import realtime


class RealtimeTests(unittest.TestCase):
    def test_raw_by_coordinates_baseline_and_unwrapped_clock(self):
        stream = realtime.FrameStream()
        frame = SimpleNamespace(unwrapped_us=4294968000, segment=0, by=37.25)
        packets = stream.frame(frame, capture_id='capture-a', stage='step_003',
            status={'x':-1250,'fault':'NONE'}, x_ppmm=250, baseline=(1,30,3), active=True)
        self.assertEqual(packets[0]['type'], 'reset')
        self.assertEqual(packets[-1], dict(type='sample',capture_id='capture-a',stage='step_003',
            t_s=4294.968,x_mm=-5.0,by_uT=37.25,baseline_by_uT=30.0,valid=True,
            invalid_reason='',motor_fault=False,motor_fault_code='NONE'))

    def test_calibration_fault_and_unsupported_stage_are_invalid(self):
        stream = realtime.FrameStream()
        frame = SimpleNamespace(unwrapped_us=1000, segment=0, by=20.)
        for stage, baseline, active, status, reason in [
            ('step_003',None,True,{'x':1,'fault':'NONE'},'baseline_unknown'),
            ('step_002',(1,2,3),True,{'x':1,'fault':'NONE'},'stage_not_classifiable'),
            ('step_003',(1,2,3),False,{'x':1,'fault':'NONE'},'not_recording'),
            ('step_003',(1,2,3),True,{'x':1,'fault':'LIMIT'},'motor_fault'),
            ('step_003',(1,2,3),True,None,'motor_status_unknown')]:
            packet = stream.frame(frame,capture_id='a',stage=stage,status=status,
                x_ppmm=250,baseline=baseline,active=active)[-1]
            self.assertFalse(packet['valid']); self.assertEqual(packet['invalid_reason'],reason)

    def test_stage_clock_baseline_changes_reset_and_capture_end(self):
        stream = realtime.FrameStream()
        frame = SimpleNamespace(unwrapped_us=1000,segment=0,by=20.)
        args=dict(capture_id='a',stage='step_003',status={'x':1,'fault':'NONE'},x_ppmm=250,baseline=(1,2,3),active=True)
        stream.frame(frame,**args)
        self.assertEqual(len(stream.frame(frame,**args)),1)
        args['stage']='step_004'
        self.assertEqual(stream.frame(frame,**args)[0]['type'],'reset')
        frame.segment=1
        self.assertEqual(stream.frame(frame,**args)[0]['type'],'reset')
        args['baseline']=(1,3,3)
        self.assertEqual(stream.frame(frame,**args)[0]['type'],'reset')
        self.assertEqual(stream.end('completed')['capture_id'],'a')

    def test_disabled_publisher_no_listener_and_loopback_ndjson(self):
        pub=realtime.Publisher(port=0)
        self.assertFalse(pub.running)
        self.assertFalse(pub.publish({'type':'sample'}))
        pub.start()
        try:
            with socket.create_connection(pub.address,timeout=1) as client:
                client.settimeout(2); reader=client.makefile('rb')
                self.assertEqual(json.loads(reader.readline())['type'],'reset')
                self.assertTrue(pub.publish({'type':'sample','by_uT':31.5}))
                self.assertEqual(json.loads(reader.readline()),{'type':'sample','by_uT':31.5})
        finally: pub.stop()

    def test_full_queue_never_blocks_and_inserts_reset(self):
        pub=realtime.Publisher(port=0,queue_size=2)
        # Exercise bounded enqueue without a consumer thread.
        pub._running=True
        self.assertTrue(pub.publish({'type':'sample','capture_id':'a','stage':'step_003','t_s':1}))
        pub.publish({'type':'sample','capture_id':'a','stage':'step_003','t_s':2})
        start=time.monotonic()
        pub.publish({'type':'sample','capture_id':'a','stage':'step_003','t_s':3})
        self.assertLess(time.monotonic()-start,.1)
        self.assertEqual(json.loads(pub.items.get_nowait())['type'],'reset')
        self.assertEqual(json.loads(pub.items.get_nowait())['t_s'],3)
        self.assertEqual(pub.dropped,2)
        pub._running=False

    def test_bind_failure_and_peer_disconnect_do_not_raise_in_publish(self):
        pub=realtime.Publisher(port=0); pub.start()
        second=realtime.Publisher(port=pub.address[1])
        try:
            self.assertFalse(second.start()); self.assertTrue(second.error)
            peer=socket.create_connection(pub.address,timeout=1); peer.close()
            for i in range(100): self.assertTrue(pub.publish({'type':'sample','t_s':i}))
        finally: pub.stop(); second.stop()

if __name__=='__main__': unittest.main()

class BackendForwardingTests(unittest.TestCase):
    def test_opt_in_at_frame_boundary_and_end_without_starting_worker(self):
        import tempfile
        from pathlib import Path
        from xz_workbench.backend import Worker
        class BytesSensor:
            def __init__(self): self.buffer=b'4294967000,100,37.25,30\n1000,101,38,31\n'
            @property
            def in_waiting(self): return len(self.buffer)
            def read(self,n): value=self.buffer[:n]; self.buffer=self.buffer[n:]; return value
        with tempfile.TemporaryDirectory() as folder:
            worker=Worker(settings_path=Path(folder)/'settings.json')
            self.assertFalse(worker.realtime.running)
            worker.realtime=realtime.Publisher(port=0)
            worker._handle('realtime_forward',{'enabled':True})
            try:
                with socket.create_connection(worker.realtime.address,timeout=1) as client:
                    client.settimeout(2); reader=client.makefile('rb'); reader.readline()
                    worker.sensor=BytesSensor(); worker.sensor_port='OFFLINE_FIXTURE'
                    worker.baseline=(100,30,30); worker.engine.stage='step_003'
                    worker.engine.status={'x':-1250,'fault':'NONE'}
                    worker.capture={'id':'capture-test'}
                    # Recorder boundary fake: only append, no device/filesystem activity.
                    class FrameSink:
                        error=None
                        def append(self,*args): pass
                    worker.recorder=FrameSink()
                    worker._receive_sensor(time.monotonic())
                    reset=json.loads(reader.readline()); first=json.loads(reader.readline()); second=json.loads(reader.readline())
                    self.assertEqual(reset['type'],'reset')
                    self.assertEqual(first['by_uT'],37.25); self.assertEqual(first['x_mm'],-5.)
                    self.assertTrue(first['valid']); self.assertAlmostEqual(second['t_s'],4294.968296)
                    worker.recorder=None
                    worker._end_record('completed')
                    self.assertEqual(json.loads(reader.readline())['type'],'end')
            finally: worker.realtime.stop()

    def test_publisher_failure_does_not_interrupt_recording(self):
        import tempfile
        from pathlib import Path
        from xz_workbench.backend import Worker
        with tempfile.TemporaryDirectory() as folder:
            worker=Worker(settings_path=Path(folder)/'settings.json')
            class FailingPublisher:
                running=True
                def publish(self,packet): raise OSError('offline network failure')
            worker.realtime=FailingPublisher()
            frame=SimpleNamespace(unwrapped_us=1000,segment=0,by=20.)
            worker._forward_frame(frame)
            self.assertIn('offline network failure',worker.realtime_error)

class PeerBackpressureTests(unittest.TestCase):
    def test_idle_connected_peer_does_not_busy_spin(self):
        pub=realtime.Publisher(port=0); pub.start()
        try:
            with socket.create_connection(pub.address,timeout=1) as client:
                client.settimeout(1); client.recv(1024)
                start=time.process_time(); time.sleep(.15)
                self.assertLess(time.process_time()-start,.075)
        finally: pub.stop()

    def test_peer_exceeding_pending_limit_is_closed(self):
        pub=realtime.Publisher(port=0); pub.start()
        try:
            with socket.create_connection(pub.address,timeout=1) as client:
                client.settimeout(2); client.recv(1024)
                pub.publish({'type':'sample','payload':'x'*70000})
                self.assertEqual(client.recv(1024),b'')
        finally: pub.stop()
