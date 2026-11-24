import json
import tempfile
import unittest
from pathlib import Path
from xz_workbench.model import AxisConfig,Recipe,Step,plan_steps,save_recipe,load_recipe

def isolated_worker(test):
    from xz_workbench.backend import Worker
    tmp=tempfile.TemporaryDirectory(); test.addCleanup(tmp.cleanup)
    return Worker(Path(tmp.name)/'prefs.json')

class OriginPlanningTests(unittest.TestCase):
    def test_capture_start_uses_fixed_coordinates_and_retracts_before_x(self):
        cfg=AxisConfig(start_x=10,start_z=5,clearance_z=-2)
        steps=plan_steps(Recipe('start',[Step('capture_start')]),{'x':5000,'z':2000},cfg)
        self.assertEqual([(s.x,s.z) for s in steps],[(5000,-500),(2500,-500),(2500,1250)])
        self.assertEqual([s.index for s in steps],[1,1,1])
        # Already higher than clearance: never lower Z before moving X.
        high=plan_steps(Recipe('start',[Step('capture_start')]),{'x':0,'z':-1000},cfg)
        self.assertEqual(high[0].z,-1000)
    def test_missing_start_bad_clearance_and_unsafe_intermediate_route_are_rejected(self):
        with self.assertRaises(ValueError): plan_steps(Recipe('none',[Step('capture_start')]),{'x':0,'z':0},AxisConfig())
        with self.assertRaises(ValueError): AxisConfig(start_x=1,start_z=2,clearance_z=3).validate()
        cfg=AxisConfig(start_x=1,start_z=2,clearance_z=-10,z_min=-5)
        with self.assertRaises(ValueError): plan_steps(Recipe('bounds',[Step('capture_start')]),{'x':0,'z':0},cfg)
    def test_old_recipe_config_and_new_start_round_trip(self):
        old=AxisConfig().to_dict()
        for name in ('start_x','start_z','clearance_z'): old.pop(name,None)
        self.assertFalse(Recipe('old',[Step()],old).config_changed(AxisConfig()))
        with tempfile.TemporaryDirectory() as td:
            cfg=AxisConfig(start_x=1,start_z=2,clearance_z=0)
            path=Path(td)/'recipe.json'; save_recipe(path,Recipe('new',[Step('capture_start')],cfg.to_dict()))
            loaded=load_recipe(path)
            self.assertEqual(loaded.steps[0].kind,'capture_start'); self.assertEqual(loaded.config['start_x'],1)

class ManualReferenceTests(unittest.TestCase):
    def setUp(self):
        from xz_workbench.backend import Worker
        from test_workbench_backend import until
        self.until=until; self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.worker=Worker(Path(self.tmp.name)/'prefs.json'); self.worker.start(); self.addCleanup(self.worker.shutdown)
        self.worker.submit('connect_motor',port='DEMO'); until(lambda:self.worker.snapshot().get('ready'))
    def confirm(self):
        self.worker.submit('calibrate_origin',origin_name='定位块A')
        self.until(lambda:self.worker.snapshot().get('manual_reference',{}).get('valid'))
        self.worker.submit('arm'); self.until(lambda:self.worker.snapshot()['status']['armed']==1)
    def test_temporary_alignment_allows_relative_move_but_blocks_automatic_run(self):
        self.worker.submit('begin_alignment'); self.until(lambda:self.worker.snapshot()['status']['refx']==1)
        self.worker.submit('arm'); self.until(lambda:self.worker.snapshot()['status']['armed']==1)
        self.worker.submit('move',step=Step('relative','X',.02))
        self.until(lambda:self.worker.snapshot()['status']['x']==5 and self.worker.snapshot()['owner'] is None)
        self.worker.submit('run',recipe=Recipe('blocked',[Step('relative','X',1)]))
        self.until(lambda:'人工标定' in self.worker.snapshot().get('notice',''))
        self.assertEqual(self.worker.snapshot()['status']['x'],5)
    def test_confirm_origin_and_reconnect_or_scale_change_invalidate(self):
        self.confirm(); self.assertEqual(self.worker.snapshot()['manual_reference']['name'],'定位块A')
        cfg=AxisConfig(x_ppmm=500); self.worker.submit('configure',config=cfg.to_dict())
        self.until(lambda:self.worker.snapshot()['config']['x_ppmm']==500)
        self.assertFalse(self.worker.snapshot()['manual_reference']['valid'])
        self.confirm(); self.worker.submit('disconnect_motor'); self.until(lambda:not self.worker.snapshot()['ready'])
        self.assertFalse(self.worker.snapshot()['manual_reference']['valid'])
    def test_point_action_and_record_metadata_use_confirmed_machine_frame(self):
        self.confirm(); cfg=AxisConfig(start_x=.04,start_z=.02,clearance_z=-.02)
        self.worker.submit('configure',config=cfg.to_dict()); self.until(lambda:self.worker.snapshot()['config'].get('start_x')==.04)
        self.worker.submit('connect_sensor',port='DEMO'); self.until(lambda:self.worker.snapshot().get('sensor_alive'))
        self.worker.submit('baseline',seconds=.12); self.until(lambda:self.worker.snapshot().get('baseline') is not None)
        root=Path(self.tmp.name)/'records'; root.mkdir()
        self.worker.submit('run_record',root=str(root),label='A',recipe=Recipe('fixed',[Step('capture_start')]))
        self.until(lambda:self.worker.snapshot().get('saved_path'))
        self.assertEqual((self.worker.snapshot()['status']['x'],self.worker.snapshot()['status']['z']),(10,5))
        meta=json.loads(next(root.glob('*/metadata.json')).read_text())
        self.assertEqual(meta['context']['coordinate_reference']['name'],'定位块A')
        self.assertEqual(meta['context']['coordinate_reference']['method'],'manual')
        self.assertEqual(meta['context']['capture_start_mm']['x'],.04)
    def test_partial_zero_invalidates_fixed_reference(self):
        self.confirm(); self.worker.submit('zero',axis='X')
        self.until(lambda:not self.worker.snapshot()['manual_reference']['valid'])

class ReferenceAckTests(unittest.TestCase):
    def test_ack_without_new_zero_status_does_not_mark_calibrated(self):
        from xz_workbench.backend import Worker
        from xz_control.protocol import Simulator
        w=isolated_worker(self); w.sim=Simulator(); w.engine.connect(w.sim.s.copy())
        w._handle('calibrate_origin',{'origin_name':'固定基准'})
        packets=list(w.packets); w.packets.clear()
        w.packets.extend(p for p in packets if p.get('type')!='status')
        from unittest.mock import patch
        with patch.object(w.sim,'tick',return_value=[]): w._receive_motor(0,0)
        w._publish()
        self.assertFalse(w.snapshot()['manual_reference']['valid'])

class AppliedScaleTests(unittest.TestCase):
    def test_applied_conversion_round_trip_requires_saving_point_again(self):
        from xz_workbench.backend import Worker
        from xz_control.protocol import Simulator
        w=isolated_worker(self); w.sim=Simulator(); w.engine.connect(w.sim.s.copy())
        cfg=AxisConfig(start_x=10,start_z=2,clearance_z=0)
        w._handle('configure',{'config':cfg.to_dict()})
        for scale in (500,250):
            cfg.x_ppmm=scale
            w._handle('configure',{'config':cfg.to_dict()})
            self.assertIsNone(w.engine.config.start_x)
        events=[]
        while not w.events.empty(): events.append(w.events.get_nowait())
        self.assertEqual(sum(e['type']=='start_cleared' for e in events),2)

    def test_fault_after_ack_cancels_pending_calibration(self):
        from xz_workbench.backend import Worker
        from xz_control.protocol import Simulator
        from unittest.mock import patch
        w=isolated_worker(self); w.sim=Simulator(); w.engine.connect(w.sim.s.copy())
        w._handle('calibrate_origin',{'origin_name':'固定基准'})
        acks=[p for p in w.packets if p.get('type')=='ack']; w.packets.clear(); w.packets.extend(acks)
        bad=w.sim.s.copy(); bad['fault']='LIMIT'; w.packets.append(bad)
        with patch.object(w.sim,'tick',return_value=[]):
            w._receive_motor(0,0)
            w.packets.append(w.sim.s.copy()); w._receive_motor(0,0)
        w._publish(); self.assertFalse(w.snapshot()['manual_reference']['valid'])

class RestoredReferenceTests(unittest.TestCase):
    def test_nondefault_saved_conversion_preserves_start_on_new_session_confirmation(self):
        from xz_workbench.backend import Worker
        from xz_control.protocol import Simulator
        cfg=AxisConfig(x_ppmm=500,start_x=10,start_z=2,clearance_z=0)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'prefs.json'; path.write_text(json.dumps({'session':{'config':cfg.to_dict(),'manual_origin':{'name':'定位块A'}}}))
            w=Worker(path); w.sim=Simulator(); w.engine.connect(w.sim.s.copy())
            w._handle('configure',{'config':cfg.to_dict()})
            self.assertFalse(w._reference_state()['valid'])
            w._handle('calibrate_origin',{'origin_name':'定位块A'}); w._receive_motor(0,0)
            self.assertTrue(w._reference_state()['valid'])
            self.assertEqual(w.engine.config.start_x,10)
