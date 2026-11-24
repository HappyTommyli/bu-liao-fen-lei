import csv
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from xz_workbench.backend import Worker
from xz_workbench.model import AxisConfig,Recipe,Step
from xz_workbench.recording import Recorder,CSV_HEADER
from test_workbench_backend import until

class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'data'; self.root.mkdir()
        self.worker=Worker(Path(self.tmp.name)/'prefs.json'); self.worker.start(); self.addCleanup(self.worker.shutdown)
        self.worker.submit('connect_motor',port='DEMO'); until(lambda:self.worker.snapshot().get('ready'))
        self.worker.submit('zero',axis='ALL'); until(lambda:self.worker.snapshot()['status']['refx']==1)
        self.worker.submit('arm'); until(lambda:self.worker.snapshot()['status']['armed']==1)
        self.worker.submit('connect_sensor',port='DEMO'); until(lambda:self.worker.snapshot().get('sensor_alive'))
        self.worker.submit('baseline',seconds=.12); until(lambda:self.worker.snapshot().get('baseline') is not None)
    def start(self,count=3,interval=0):
        self.worker.submit('batch_record',root=str(self.root),label='棉布',recipe=Recipe('scan',[Step('relative','X',.02)]),count=count,interval=interval)
    def metas(self): return [json.loads(p.read_text()) for p in self.root.glob('*/metadata.json')]
    def active(self): return self.worker.snapshot().get('batch',{}).get('active',False)
    def test_three_rounds_save_separate_csv_and_batch_metadata(self):
        self.start(); until(lambda:len(self.metas())==3 and not self.active(),seconds=5)
        self.assertEqual(self.worker.snapshot()['status']['x'],15)
        self.assertTrue(all(m['outcome']=='completed' and m['sample_count']>0 for m in self.metas()))
        self.assertEqual({m['context']['batch']['round'] for m in self.metas()},{1,2,3})
        self.assertEqual(len({m['context']['batch']['id'] for m in self.metas()}),1)
        for p in self.root.glob('*/*.csv'):
            with p.open(encoding='utf-8-sig') as f: self.assertEqual(next(csv.reader(f)),CSV_HEADER)
    def test_stop_after_round_prevents_next(self):
        self.start(5); until(self.active)
        self.worker.submit('batch_stop_after'); until(lambda:not self.active())
        self.assertEqual(len(self.metas()),1); self.assertEqual(self.metas()[0]['outcome'],'completed')
    def test_interval_reserves_controls_and_immediate_stop_cancels_remaining(self):
        self.start(3,1); until(lambda:self.worker.snapshot().get('batch',{}).get('phase')=='interval')
        self.assertFalse(self.worker.snapshot()['idle'])
        self.worker.submit('move',step=Step('relative','Z',1)); until(lambda:'批量' in self.worker.snapshot().get('notice',''))
        self.worker.submit('stop'); until(lambda:not self.active()); time.sleep(.1)
        self.assertEqual(self.worker.snapshot()['status']['z'],0); self.assertEqual(len(self.metas()),1)
    def test_next_round_waits_for_final_save_and_failure_aborts_batch(self):
        original=Recorder.close; entered=threading.Event(); release=threading.Event()
        def close(recorder,*args,**kwargs):
            result=original(recorder,*args,**kwargs)
            if recorder.path.name=='棉布_001': entered.set(); release.wait(3); raise OSError('batch disk failure')
            return result
        with patch.object(Recorder,'close',new=close):
            try:
                self.start(); until(entered.is_set); time.sleep(.15)
                self.assertEqual(len(self.metas()),1); self.assertTrue(self.active())
                release.set(); until(lambda:not self.active())
                self.assertEqual(len(self.metas()),1)
                self.assertIn('batch disk failure',self.worker.snapshot()['notice'])
            finally: release.set()
    def test_later_round_bounds_are_rejected_before_any_motion(self):
        cfg=AxisConfig(x_max=.03); self.worker.submit('configure',config=cfg.to_dict())
        until(lambda:self.worker.snapshot()['config']['x_max']==.03)
        self.start(3); until(lambda:bool(self.worker.snapshot().get('notice')))
        self.assertEqual(self.worker.snapshot()['status']['x'],0); self.assertEqual(self.metas(),[])
        self.assertFalse(self.active())
        self.assertIn('第 2 轮',self.worker.snapshot()['notice'])

    def test_transient_fault_during_interval_aborts_before_recovery(self):
        self.start(3,2); until(lambda:self.worker.snapshot().get('batch',{}).get('phase')=='interval')
        status=self.worker.sim.s.copy(); status['fault']='LIMIT'
        self.worker.packets.append(status)
        until(lambda:not self.active(),seconds=.5)
        self.worker.packets.append(self.worker.sim.s.copy()); time.sleep(.1)
        self.assertEqual(self.worker.snapshot()['batch']['phase'],'aborted'); self.assertEqual(len(self.metas()),1)

    def test_reference_loss_during_final_save_cannot_report_completed(self):
        original=Recorder.close; entered=threading.Event(); release=threading.Event()
        def close(recorder,*args,**kwargs):
            result=original(recorder,*args,**kwargs); entered.set(); release.wait(3); return result
        with patch.object(Recorder,'close',new=close):
            try:
                self.start(1); until(entered.is_set)
                status=self.worker.sim.s.copy(); status['refx']=0; self.worker.packets.append(status)
                until(lambda:not self.active(),seconds=.5)
                self.assertEqual(self.worker.snapshot()['batch']['phase'],'aborted')
            finally: release.set()

    def test_motor_unplug_and_failed_stop_always_clear_batch(self):
        self.start(3,2); until(lambda:self.worker.snapshot().get('batch',{}).get('phase')=='interval')
        sim=self.worker.sim
        with patch.object(sim,'tick',side_effect=OSError('USB unplugged')),patch.object(sim,'command',side_effect=OSError('STOP failed')):
            until(lambda:not self.worker.snapshot().get('ready'))
            self.assertFalse(self.active())
            self.assertEqual(self.worker.snapshot()['batch']['phase'],'aborted')

    def test_motor_unplug_running_batch_does_not_leave_controls_reserved(self):
        self.worker.submit('batch_record',root=str(self.root),label='布料',recipe=Recipe('long',[Step('wait',duration=1),Step('relative','X',.02)]),count=2,interval=0)
        until(self.active); sim=self.worker.sim
        with patch.object(sim,'tick',side_effect=OSError('USB unplugged')),patch.object(sim,'command',side_effect=OSError('STOP failed')):
            until(lambda:not self.worker.snapshot().get('ready'))
            self.assertFalse(self.active())
            self.assertEqual(self.worker.snapshot()['batch']['phase'],'aborted')

class BatchValidationTests(unittest.TestCase):
    def test_reject_invalid_count_interval_and_future_baseline(self):
        from xz_workbench.batch import validate_batch
        recipe=Recipe('x',[Step('relative','X',1)])
        for n,interval in [(0,0),(True,0),(2,-1),(2,float('nan'))]:
            with self.subTest(n=n,interval=interval),self.assertRaises(ValueError):
                validate_batch(recipe,{'x':0,'z':0},AxisConfig(),n,interval,True)
