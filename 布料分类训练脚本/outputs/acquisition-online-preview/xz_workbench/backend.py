"""Background device ownership, heartbeat, sensing and recording orchestration."""
import copy
import json
import math
import queue
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import serial
from serial.tools import list_ports
from xz_control.protocol import Simulator, parse_status
from .realtime import Publisher, FrameStream
from .engine import Engine
from .batch import validate_batch
from .model import AxisConfig, plan_steps, validate_capture
from .catalog import Catalog, DEFAULT_SETTINGS, DEFAULT_LABEL, valid_label, read_json
from .recording import Frame, SensorDecoder, BaselineCollector, Recorder


def open_port(port,baud):
    conn=serial.Serial(port=None,baudrate=baud,timeout=0,write_timeout=.1,exclusive=True)
    conn.dtr=False; conn.rts=False; conn.port=port
    try: conn.open()
    except Exception:
        conn.close(); raise
    return conn


def same_device(a,b):
    if not a or not b or a=='DEMO' or b=='DEMO': return False
    return Path(a).resolve()==Path(b).resolve()


class Worker(threading.Thread):
    def __init__(self,settings_path=None):
        super().__init__(name='xz-device-worker',daemon=True)
        self.commands=queue.Queue(maxsize=256); self.events=queue.Queue(maxsize=200)
        self.writer_failures=queue.SimpleQueue()
        self.save_results=queue.SimpleQueue(); self.pending_saves={}
        self.batch=None; self.last_batch={}
        self.exit_event=threading.Event(); self.stop_event=threading.Event(); self.release_event=threading.Event()
        self.lock=threading.Lock(); self.latest={}; self.gui_lease=0.; self.generation=0
        self.engine=Engine(self._send,time.monotonic,self._engine_event)
        self.motor=None; self.motor_port=None; self.sensor=None; self.sensor_port=None
        self.sim=None; self.sensor_demo=False; self.decoder=SensorDecoder(); self.motor_buffer=bytearray()
        self.packets=deque(); self.hello=None; self.opened_at=0.; self.last_hello=0.; self.last_ping=0.
        self.handshake_complete=False
        self.collector=None; self.baseline=None; self.baseline_info=None
        self.last_sensor=0.; self.last_frame=None; self.history=deque(maxlen=2000); self.last_plot=0.
        self.demo_stamp=0; self.demo_frame_at=0.; self.previous_cycle=time.monotonic()
        self.recorder=None; self.record_owned=False; self.pending_recipe=None; self.record_wait=None
        self.finalizers=[]; self.finalizing=set(); self.saved_path=None; self.notice=''; self.last_publish=0.
        self.settings_path=Path(settings_path) if settings_path is not None else DEFAULT_SETTINGS
        self.capture=None; self.external=None; self.resume_external=None
        self.realtime=Publisher(); self.realtime_stream=FrameStream(); self.realtime_error=''
        self.sensor_baud=115200; self.sensor_identity=None
        self.manual_reference=None; self.reference_pending=None; self.origin_name='人工机械基准'
        session=read_json(self.settings_path).get('session',{})
        manual=session.get('manual_origin',{}) if isinstance(session,dict) else {}
        if isinstance(manual,dict) and isinstance(manual.get('name'),str) and manual['name'].strip(): self.origin_name=manual['name'].strip()
        # Restore parameters before the first UI configure, never physical position/trust.
        # This distinguishes startup restoration from an applied conversion change.
        if isinstance(session,dict):
            try: self.engine.configure(AxisConfig(**session.get('config',{})).without_bounds())
            except (TypeError,ValueError): pass

    def emit(self,event_type,**payload):
        event=dict(type=event_type,**payload)
        try: self.events.put_nowait(event)
        except queue.Full:
            try: self.events.get_nowait()
            except queue.Empty: pass
            try: self.events.put_nowait(event)
            except queue.Full: pass

    def snapshot(self):
        with self.lock: return copy.deepcopy(self.latest)

    def submit(self,name,**payload):
        if name in ('stop','release'):
            with self.lock: self.generation+=1
            (self.stop_event if name=='stop' else self.release_event).set()
            return
        with self.lock: generation=self.generation
        try: self.commands.put_nowait((generation,name,copy.deepcopy(payload)))
        except queue.Full:
            self.stop_event.set(); self.emit('error',message='操作队列已满，已请求停止')

    def lease(self):
        with self.lock: self.gui_lease=time.monotonic()

    def _send(self,line):
        if self.sim:
            self.packets.extend(self.sim.command(line)); return
        if not self.motor: raise OSError('电机串口未连接')
        data=(line+'\n').encode('ascii')
        if self.motor.write(data)!=len(data): raise OSError('电机串口写入不完整')

    def _engine_event(self,event):
        self.emit(event['type'],**{k:v for k,v in event.items() if k!='type'})
        if event['type']=='stopped':
            self.reference_pending=None
            if not self.engine.status or not all(self.engine.status.get(k) for k in ('refx','refz')): self.manual_reference=None
            self._finish_batch('aborted',event.get('reason'))
            self.collector=None; self.external=None; self.resume_external=None
            self._end_record('aborted',event.get('reason'))
        elif event['type']=='complete' and self.record_owned:
            self._end_record('completed')
            if self.batch: self.batch['phase']='saving'
        elif event['type']=='external':
            try: self._begin_external(event)
            except (ValueError,OSError,serial.SerialException) as exc:
                self.notice=str(exc); self.emit('error',message=str(exc)); self.engine.stop(str(exc))

    def _close_motor(self):
        try:
            if self.engine.ready: self.engine.stop('断开电机')
            elif self.motor: self._send('STOP')
        finally:
            self.manual_reference=None; self.reference_pending=None
            if self.motor: self.motor.close()
            self.motor=None; self.sim=None; self.motor_port=None; self.hello=None
            self.handshake_complete=False
            self.engine.ready=False; self.engine.status=None; self.engine.pending=None
            self.packets.clear(); self.motor_buffer.clear()

    def _connect_motor(self,port):
        if self.capture or self.engine.owner: raise ValueError('请先停止运动和采集')
        if same_device(port,self.sensor_port): raise ValueError('该设备已用于磁传感器')
        self._close_motor()
        self.motor_port=port; self.opened_at=time.monotonic(); self.last_hello=0.
        if port=='DEMO':
            self.sim=Simulator(); self.engine.connect(self.sim.s.copy())
            self.handshake_complete=True
            self.emit('log',message='模拟电机：不连接硬件，运动仅为理想模型')
        else:
            try: self.motor=open_port(port,115200)
            except Exception: self.motor_port=None; raise
            self.emit('log',message='等待 XZ1 固件握手')

    def _connect_sensor(self,port,baud):
        if self.capture or self.collector or self.engine.owner: raise ValueError('请先结束函数、采集或基线校准')
        if same_device(port,self.motor_port): raise ValueError('该设备已用于电机')
        if self.sensor: self.sensor.close()
        self.sensor=None; self.sensor_port=None; self.sensor_demo=False
        self.baseline=None; self.baseline_info=None; self.last_sensor=0.; self.last_frame=None
        self.decoder=SensorDecoder(); self.history.clear()
        self.sensor_baud=baud; self.sensor_identity=None
        if port:
            if port=='DEMO': self.sensor_demo=True
            else: self.sensor=open_port(port,baud)
            self.sensor_port=port
            for item in list_ports.comports():
                if same_device(item.device,port) and item.serial_number:
                    self.sensor_identity=(item.vid,item.pid,item.serial_number); break

    def _start_record(self,root,recipe=None,label=DEFAULT_LABEL):
        self.engine.require_idle()
        if self.capture or self.collector: raise ValueError('正在采集或校准')
        label=valid_label(label)
        planned=None
        if recipe:
            self._require_reference()
            self.engine.require_idle(armed=True)
            if recipe.config_changed(self.engine.config): raise ValueError('流程配置已改变，请检查并重新保存')
            planned=plan_steps(recipe,self.engine.status,self.engine.config)
            validate_capture(recipe,self.baseline is not None)
        elif self.baseline is None: raise ValueError('请先完成静止空载基线校准')
        if not self.sensor_port: raise ValueError('请先连接磁传感器采集板')
        if time.monotonic()-self.last_sensor>2 and not (planned and planned[0].kind=='restart_sensor'):
            raise ValueError('传感器没有持续输出有效数据')
        self.capture=dict(root=str(root),label=label,recipe=recipe,id=uuid.uuid4().hex,segment=0)
        self.engine.recording=True; self.record_owned=recipe is not None
        if planned and (self.baseline is None or planned[0].kind in ('restart_sensor','baseline')):
            self.engine.run(recipe)
        else:
            self._open_segment()
            self.pending_recipe=recipe
            if recipe is not None:
                self.engine.owner='preparing'; self.engine.stage='record_ready'

    def _open_segment(self):
        if not self.capture or self.recorder: return
        if self.baseline is None: raise ValueError('当前传感器尚无有效基线，不能开始记录')
        self._discard_sensor_backlog()
        capture=self.capture; recipe=capture['recipe']; capture['segment']+=1
        context=dict(motor_port=self.motor_port,sensor_port=self.sensor_port,config=self.engine.config.to_dict(),
                     demo=bool(self.sim or self.sensor_demo),baseline=self.baseline_info,
                     capture_session=capture['id'],segment=capture['segment'],
                     coordinate_reference=self._reference_state(),
                     capture_start_mm=dict(x=self.engine.config.start_x,z=self.engine.config.start_z,clearance_z=self.engine.config.clearance_z),
                     recipe=({'name':recipe.name,'steps':[vars(step) for step in recipe.steps]} if recipe else None))
        if self.batch:
            context['batch']=dict(id=self.batch['id'],round=self.batch['round'],count=self.batch['count'])
        allocation=Catalog(capture['root'],self.settings_path).reserve(capture['label'])
        self.recorder=Recorder(capture['root'],self.baseline,context,allocation=allocation)
        self.record_wait=time.monotonic()+2.
        self.emit('label',label=capture['label'])
        self.emit('log',message='记录已创建，等待新的传感器样本：'+str(self.recorder.path))

    def _end_segment(self,outcome='completed',reason=None):
        recorder=self.recorder; self.recorder=None; self.record_wait=None
        if not recorder: return
        # Final flushing never blocks motor heartbeat or user stop processing.
        with self.lock: self.finalizing.add(recorder.path)
        session=recorder.context.get('capture_session')
        self.pending_saves[session]=self.pending_saves.get(session,0)+1
        def finalize():
            error=None
            try:
                path=recorder.close(outcome,reason); self.saved_path=str(path)
                self.emit('saved',path=str(path),outcome=outcome)
            except Exception as exc:
                error=str(exc)
            finally:
                with self.lock:
                    if not recorder.thread.is_alive(): self.finalizing.discard(recorder.path)
                self.save_results.put((session,error))
        thread=threading.Thread(target=finalize,name='xz-record-finalize',daemon=True)
        self.finalizers=[t for t in self.finalizers if t.is_alive()]+[thread]
        thread.start()

    def _check_writer_failures(self):
        # Results must be processed before a batch can advance to another round.
        while True:
            try: session,error=self.save_results.get_nowait()
            except queue.Empty: break
            count=self.pending_saves.get(session,1)-1
            if count: self.pending_saves[session]=count
            else: self.pending_saves.pop(session,None)
            if error: self.writer_failures.put((session,error))
        # Only this worker mutates the engine; a finalizer may belong to an older session.
        while True:
            try: session,message=self.writer_failures.get_nowait()
            except queue.Empty: return
            reason='CSV 保存失败：'+message
            self.notice=reason; self.emit('error',message=reason)
            if (self.capture and self.capture['id']==session) or (self.batch and self.batch.get('session')==session):
                with self.lock: self.generation+=1
                self.engine.stop(reason)

    def _end_record(self,outcome='completed',reason=None):
        if self.realtime.running:
            try: self.realtime.publish(self.realtime_stream.end(reason or outcome))
            except Exception as exc: self.realtime_error=str(exc)
        self._end_segment(outcome,reason)
        self.capture=None; self.engine.recording=False; self.record_owned=False
        self.pending_recipe=None; self.record_wait=None; self.resume_external=None

    def _start_batch(self,payload):
        self.engine.require_idle(armed=True)
        if self.capture or self.collector: raise ValueError('正在采集或校准')
        self._require_reference()
        if not self.sensor_port: raise ValueError('请先连接磁传感器采集板')
        root=Path(payload['root'])
        if not root.is_dir(): raise ValueError('保存目录不存在，请重新选择')
        label=valid_label(payload.get('label',DEFAULT_LABEL))
        recipe=copy.deepcopy(payload['recipe']); count=payload['count']; interval=payload.get('interval',0)
        validate_batch(recipe,self.engine.status,self.engine.config,count,interval,self.baseline is not None)
        self.batch=dict(id=uuid.uuid4().hex,recipe=recipe,root=str(root),label=label,count=count,
                        interval=interval,round=0,completed=0,phase='starting',stop_after=False,session=None)
        try: self._begin_batch_round()
        except Exception:
            self._finish_batch('aborted','启动失败')
            if self.capture: self.engine.stop('批量启动失败')
            raise

    def _begin_batch_round(self):
        batch=self.batch; batch['round']+=1; batch['phase']='running'
        self._start_record(batch['root'],batch['recipe'],batch['label'])
        batch['session']=self.capture['id']
        self.emit('log',message=f'批量采集：第 {batch["round"]}/{batch["count"]} 轮')

    def _finish_batch(self,outcome,reason=None):
        if not self.batch: return
        batch=self.batch
        self.last_batch={k:batch[k] for k in ('id','round','count','completed')}
        self.last_batch.update(active=False,phase=outcome,reason=reason or '')
        self.batch=None
        self.emit('batch_done',**self.last_batch)

    def _tick_batch(self,now):
        batch=self.batch
        if not batch or batch['phase']=='running': return
        # No user command can interleave here: the Worker reserves the entire batch.
        if batch['phase']=='saving':
            if self.pending_saves.get(batch['session'],0): return
            batch['completed']+=1
            if batch['stop_after'] or batch['completed']>=batch['count']:
                self._finish_batch('completed'); return
            batch['phase']='interval'; batch['next_at']=now+batch['interval']
        if batch['phase']=='interval':
            if batch['stop_after']:
                self._finish_batch('completed'); return
            if not self.sensor_port or now-self.last_sensor>2:
                self.notice='批量间隔期间传感器断流'; self.emit('error',message=self.notice); self.engine.stop(self.notice); return
            if now>=batch['next_at']:
                try: self._begin_batch_round()
                except (ValueError,OSError) as exc:
                    self.notice=str(exc); self.emit('error',message=str(exc)); self.engine.stop(str(exc))

    def _invalidate_baseline(self):
        self.baseline=None; self.baseline_info=None; self.history.clear(); self.last_plot=0.

    def _begin_baseline(self,duration):
        if not self.sensor_port or time.monotonic()-self.last_sensor>2:
            raise ValueError('请先连接传感器并确认数据')
        if not isinstance(duration,(float,int)) or not math.isfinite(duration) or not 0<duration<=600:
            raise ValueError('基线时间必须在 0–600 秒之间')
        self._discard_sensor_backlog(); self._invalidate_baseline()
        self.collector=BaselineCollector(time.monotonic(),duration)
        self.emit('log',message='正在采集空载基线，请保持静止')

    def _begin_external(self,event):
        if not self.sensor_port: raise ValueError('请先连接磁传感器采集板')
        self._end_segment('completed','动作切换：'+event['kind'])
        now=time.monotonic()
        self.external=dict(kind=event['kind'],index=event['index'],deadline=now+event['duration'],
                           requested=now,ack=False,ready=False,boot=False,disconnected=False,
                           buffer=bytearray(),next_open=now+.2,rx_bytes=0,data_lines=0,last_message='',open_error='')
        if event['kind']=='baseline':
            self._begin_baseline(event['duration'])
            # Samples are collected until duration, then normal freshness timeout applies.
            self.external['deadline']=now+event['duration']+2.
        else:
            self.collector=None; self._invalidate_baseline(); self.last_frame=None; self.last_sensor=0.
            self._discard_sensor_backlog(); self.decoder=SensorDecoder()
            if not self.sensor_demo:
                if not self.sensor: raise ValueError('采集板串口未打开，请重新连接')
                command=b'REBOOT\n'
                if self.sensor.write(command)!=len(command): raise OSError('采集板重启命令写入不完整')
            self.emit('log',message=f'采集板重启：{self.sensor_port} @ {self.sensor_baud}，'+('模拟重启' if self.sensor_demo else '已发送 REBOOT\\n')+'，等待确认/启动信息 → READY → 新数据')

    def _finish_external(self):
        action=self.external
        if not action: return
        self.external=None
        self.engine.complete_external(action['index'])

    def _tick_external(self,now):
        action=self.external
        if not action: return
        if now>action['deadline']:
            reason=self._restart_timeout_reason(action) if action['kind']=='restart_sensor' else '基线校准或恢复采集超时'
            self.notice=reason; self.emit('error',message=reason); self.engine.stop(reason); return
        if action['kind']!='restart_sensor': return
        if self.sensor_demo:
            if not action['ready'] and now-action['requested']>=.12:
                self.demo_stamp=0; self.decoder=SensorDecoder(); action['ready']=True
            return
        if self.sensor is None and now>=action['next_open']:
            action['next_open']=now+.25
            port=self.sensor_port
            if self.sensor_identity:
                matches=[p.device for p in list_ports.comports() if (p.vid,p.pid,p.serial_number)==self.sensor_identity]
                if len(matches)!=1:
                    action['open_error']='尚未发现原 USB 序列号对应的唯一设备'; return
                port=matches[0]
            if same_device(port,self.motor_port): return
            try:
                self.sensor=open_port(port,self.sensor_baud); self.sensor_port=port
                action['open_error']=''; self.emit('log',message='采集板 USB 已重连：'+port+'，继续等待启动信息和 READY')
            except (OSError,serial.SerialException) as exc: action['open_error']=str(exc)

    def _restart_timeout_reason(self,action):
        if action['disconnected'] and self.sensor is None:
            detail='采集板 USB 已断开，但未重新连接'
            if action['open_error']: detail+='；'+action['open_error']
        elif action['ready']:
            detail='已收到 READY，但未收到有效磁数据；请检查传感器初始化和输出格式'
        elif action['ack']:
            detail='已收到 REBOOT 确认，但未收到 READY；请检查重启后的启动和传感器初始化'
        elif action['boot']:
            detail='已收到启动信息，但未收到 READY；请检查传感器初始化'
        elif action['disconnected']:
            detail='采集板 USB 已重连，但未收到 READY；请检查固件启动输出'
        else:
            detail='未收到 REBOOT 确认或启动信息'
            if action['data_lines']:
                detail+=f'；命令发送后仍收到 {action["data_lines"]} 行四列数据，请检查当前固件是否支持 REBOOT 及其回应格式'
            elif not action['rx_bytes']:
                detail+='；采集板没有返回串口数据，请核对串口、波特率和板子状态'
            else: detail+='；请核对固件 REBOOT 命令及启动回应格式'
        if action['last_message']: detail+='；最后设备消息：'+action['last_message']
        return f'采集板重启超时：{detail}（{self.sensor_port} @ {self.sensor_baud}，接收 {action["rx_bytes"]} 字节）'

    def _restart_bytes(self,data):
        action=self.external; action['rx_bytes']+=len(data); action['buffer'].extend(data)
        frames=[]
        while b'\n' in action['buffer']:
            raw,_,rest=action['buffer'].partition(b'\n'); action['buffer']=bytearray(rest)
            line=raw.decode('utf-8',errors='replace').strip()
            if line.count(',')==3 and line[:1].isdigit(): action['data_lines']+=1
            elif line: action['last_message']=line[:240]
            if 'reboot requested' in line.lower() and not action['ack']:
                action['ack']=True; self.emit('log',message='已收到采集板 REBOOT 确认，等待重新启动和 READY')
            if line.startswith('# MLX90393 monitor') and not action['boot']:
                action['boot']=True; self.emit('log',message='收到采集板启动信息：'+line[:240])
            if line=='READY' and (action['ack'] or action['boot'] or action['disconnected']):
                if not action['ready']: self.emit('log',message='已收到 READY，等待第一帧有效磁数据')
                action['ready']=True; self.decoder=SensorDecoder()
            elif action['ready']:
                frames.extend(self.decoder.feed(raw+b'\n'))
        if len(action['buffer'])>4096: action['buffer'].clear()
        return frames

    def _reference_state(self):
        if self.manual_reference: return dict(self.manual_reference,valid=True)
        return dict(valid=False,method='manual',name=self.origin_name)

    def _require_reference(self):
        if not self.manual_reference: raise ValueError('请先人工标定固定机械原点；临时对位坐标不能运行自动流程或绝对定位')

    def _confirm_reference(self,name):
        if not isinstance(name,str) or not name.strip() or len(name)>80: raise ValueError('请输入 1–80 字的机械基准名称')
        if self.capture or self.collector: raise ValueError('请先结束采集或校准')
        self.engine.require_idle()
        self.engine.zero('ALL')
        self.manual_reference=None
        self.reference_pending=dict(name=name.strip(),id=uuid.uuid4().hex,method='manual',confirmed_at=time.time(),
                                    x_ppmm=self.engine.config.x_ppmm,z_ppmm=self.engine.config.z_ppmm)

    def _observe_reference(self,packet):
        if packet.get('type')!='status': return
        status=self.engine.status or {}
        if not all(status.get(k) for k in ('refx','refz')) or status.get('fault')!='NONE':
            self.manual_reference=None
        if status.get('fault')!='NONE': self.reference_pending=None
        if (self.reference_pending and self.engine.pending is None and not status.get('busy')
            and status.get('refx') and status.get('refz') and status.get('x')==0 and status.get('z')==0 and status.get('fault')=='NONE'):
            reference=self.reference_pending; self.reference_pending=None
            if reference['name']!=self.origin_name:
                self.engine.config.start_x=self.engine.config.start_z=self.engine.config.clearance_z=None
                self.emit('start_cleared')
            self.origin_name=reference['name']; self.manual_reference=reference
            self.emit('log',message='人工机械原点已确认：'+self.origin_name+'；当前 X=0、Z=0。这依赖你已对准实体固定基准。')

    def _handle(self,name,payload):
        if name=='realtime_forward':
            self.realtime_error=''
            if payload.get('enabled'):
                self.realtime_stream=FrameStream()
                if not self.realtime.start(): self.realtime_error=self.realtime.error
            else:
                self.realtime.stop()
            return
        if name=='batch_stop_after':
            if self.batch: self.batch['stop_after']=True
            return
        if self.batch and name not in ('rename_record','trash_records','restore_record'):
            raise ValueError('批量任务进行中，请先结束或中止批量')
        if name=='batch_record': self._start_batch(payload)
        elif name=='connect_motor': self._connect_motor(payload['port'])
        elif name=='disconnect_motor': self._close_motor()
        elif name=='connect_sensor': self._connect_sensor(payload.get('port'),payload.get('baud',115200))
        elif name=='configure':
            cfg=AxisConfig(**payload['config']); old=self.engine.config
            self.engine.configure(cfg)
            if cfg.x_ppmm!=old.x_ppmm or cfg.z_ppmm!=old.z_ppmm:
                self.manual_reference=None; self.reference_pending=None
                cfg.start_x=cfg.start_z=cfg.clearance_z=None
                self.emit('start_cleared')
                self.emit('log',message='脉冲换算已改变，请重新人工标定机械原点并保存采集起点。')
        elif name=='calibrate_origin': self._confirm_reference(payload.get('origin_name',self.origin_name))
        elif name=='begin_alignment':
            if self.capture or self.collector: raise ValueError('请先结束采集或校准')
            self.engine.zero('ALL'); self.manual_reference=None; self.reference_pending=None
            self.emit('log',message='临时对位坐标已请求建立；手动使能后可点动对准基准，自动流程不可用。对准后必须确认人工机械原点。')
        elif name=='zero':
            if payload['axis']=='ALL': self._confirm_reference(self.origin_name)
            else:
                self.engine.zero(payload['axis']); self.manual_reference=None; self.reference_pending=None
        elif name=='arm': self.engine.arm()
        elif name=='clear': self.engine.clear()
        elif name=='jog':
            if self.collector: raise ValueError('基线采集中不能移动')
            self.engine.jog(payload['axis'],payload['direction'])
        elif name in ('move','run'):
            if self.collector: raise ValueError('基线采集中不能移动')
            if name=='move':
                if payload['step'].kind!='relative': self._require_reference()
                self.engine.move(payload['step'])
            else:
                self._require_reference()
                recipe=payload['recipe']
                if any(step.enabled and step.kind in ('restart_sensor','baseline') for step in recipe.steps) and not self.sensor_port:
                    raise ValueError('流程含采集板动作，请先连接磁传感器采集板')
                if self.capture: validate_capture(recipe,self.baseline is not None)
                self.engine.run(recipe)
        elif name=='baseline':
            self.engine.require_idle()
            if self.capture or self.collector: raise ValueError('正在录制或校准')
            self._begin_baseline(payload.get('seconds',2))
        elif name=='record_start': self._start_record(payload['root'],label=payload.get('label',DEFAULT_LABEL))
        elif name=='run_record': self._start_record(payload['root'],payload['recipe'],payload.get('label',DEFAULT_LABEL))
        elif name=='rename_record':
            with self.lock: blocked=list(self.finalizing)
            if self.recorder: blocked.append(self.recorder.path)
            target=Catalog(payload['root'],self.settings_path).rename(payload['path'],payload['label'],blocked)
            self.saved_path=str(target); self.emit('renamed',path=str(target),label=payload['label'].strip())
        elif name in ('trash_records','restore_record'):
            with self.lock: blocked=list(self.finalizing)
            if self.recorder: blocked.append(self.recorder.path)
            catalog=Catalog(payload['root'],self.settings_path)
            if name=='trash_records':
                paths=catalog.trash(payload['paths'],blocked)
                self.emit('trashed',paths=[str(p) for p in paths])
            else:
                path=catalog.restore(payload['path'],blocked)
                self.emit('restored',path=str(path))
        elif name=='record_stop':
            if self.record_owned: raise ValueError('函数录制请等待完成，或点击中止')
            resuming=self.resume_external is not None
            self._end_record()
            if resuming: self._finish_external()
        else: raise ValueError('未知工作台操作')

    def _discard_sensor_backlog(self):
        if self.sensor: self.sensor.reset_input_buffer()
        self.decoder.buffer.clear(); self.decoder.discarding=False

    def _receive_motor(self,now,dt):
        if self.sim:
            self.packets.extend(self.sim.tick(min(dt,.1)))
        elif self.motor:
            if not self.engine.ready:
                if now-self.opened_at>8: raise OSError('XZ1 握手超时，请核对固件和串口')
                if now-self.last_hello>=1: self._send('HELLO'); self.last_hello=now
            if now-self.last_ping>=.25: self._send('PING'); self.last_ping=now
            self.motor_buffer.extend(self.motor.read(min(self.motor.in_waiting,65536)))
            if len(self.motor_buffer)>65536: raise OSError('电机串口数据过长')
            while b'\n' in self.motor_buffer:
                raw,_,rest=self.motor_buffer.partition(b'\n'); self.motor_buffer=bytearray(rest)
                try: packet=json.loads(raw)
                except (ValueError,UnicodeDecodeError): continue
                if not isinstance(packet,dict): continue
                if packet.get('type')=='hello':
                    if packet.get('protocol')!='XZ1' or not all(packet.get(k) is True for k in ('move_accel','jog_continuous','numeric_ranges')):
                        raise OSError('固件不兼容，请使用配套 XZ_Controller 1.4.0')
                    self.hello=packet
                elif packet.get('type')=='status' and not self.engine.ready and self.hello:
                    self.engine.connect(parse_status(packet)); self.handshake_complete=True
                    self.engine.issue('DISARM')
                elif self.engine.ready: self.packets.append(packet)
        for _ in range(min(200,len(self.packets))):
            packet=self.packets.popleft()
            if self.batch and packet.get('type')=='status':
                status=parse_status(dict(packet))
                if status['fault']!='NONE' or not all(status.get(k) for k in ('armed','refx','refz')):
                    reason='批量期间控制器故障、取消使能或零点失效'
                    self.notice=reason; self.emit('error',message=reason)
                    self._finish_batch('aborted',reason)
                    self.engine.stop(reason)
            self.engine.packet(packet)
            self._observe_reference(packet)

    def _forward_frame(self,frame):
        if not self.realtime.running: return
        try:
            # Raw By + actual baseline only; the consumer performs By-only QC.
            packets=self.realtime_stream.frame(frame,
                capture_id=self.capture['id'] if self.capture else None,
                stage=self.engine.stage,status=self.engine.status,x_ppmm=self.engine.config.x_ppmm,
                baseline=self.baseline,active=bool(self.recorder and not self.collector and not self.external))
            for packet in packets: self.realtime.publish(packet)
        except Exception as exc:
            # Forwarding failure cannot reach the capture/motor exception handler.
            self.realtime_error=str(exc)

    def _receive_sensor(self,now):
        frames=[]
        restarting=self.external and self.external['kind']=='restart_sensor'
        if self.sensor_demo:
            if restarting and not self.external['ready']: return
            if now-self.demo_frame_at>=.02:
                self.demo_frame_at=now
                x=(self.engine.status or {}).get('x',0)/250
                for _ in range(5):
                    self.demo_stamp+=4000
                    b=math.sin(self.demo_stamp/1e5+x)*2
                    frames.append(Frame(self.demo_stamp,self.demo_stamp,0,time.monotonic_ns(),100+b,20+b*.2,30-b*.4))
        elif self.sensor:
            try:
                data=self.sensor.read(min(self.sensor.in_waiting,65536))
                frames=self._restart_bytes(data) if restarting else self.decoder.feed(data)
            except (OSError,serial.SerialException) as exc:
                if restarting:
                    try: self.sensor.close()
                    except (OSError,serial.SerialException): pass
                    self.sensor=None; self.external['disconnected']=True
                    self.emit('log',message='重启期间采集板 USB 断开，正在等待同一设备重连：'+str(exc)); return
                if self.capture: raise
                try: self.sensor.close()
                except OSError: pass
                self.sensor=None; self.sensor_port=None; self.last_sensor=0.
                self.collector=None; self.baseline=None; self.baseline_info=None
                self.notice=str(exc); self.emit('error',message=str(exc)); return
        for frame in frames:
            self.last_frame=frame; self.last_sensor=now
            if restarting and self.external and self.external.get('ready'):
                self.emit('log',message='磁采集板启动已确认，等待新的空载基线')
                self._finish_external(); restarting=False
            if self.collector: self.collector.add(frame)
            if self.recorder:
                self.recorder.append(frame,self.engine.stage,self.engine.status)
                self.record_wait=None
            self._forward_frame(frame)
            if self.baseline is not None and now-self.last_plot>=.02:
                self.history.append((now,*frame.values)); self.last_plot=now
        if self.collector and self.collector.ready(now):
            collector=self.collector; self.collector=None
            try:
                info=collector.finish(now)
                self.baseline_info=info; self.baseline=info['baseline']; self.history.clear(); self.last_plot=0.
                self.emit('log',message='基线校准完成：'+str(tuple(round(v,3) for v in self.baseline)))
                if self.external and self.external['kind']=='baseline':
                    next_pos=self.engine.step_pos+1
                    next_step=self.engine.steps[next_pos] if next_pos<len(self.engine.steps) else None
                    if self.capture and (not self.record_owned or (next_step and next_step.kind in ('move','wait'))):
                        self._open_segment(); self.resume_external=self.external['index']
                    else: self._finish_external()
            except ValueError as exc:
                self.notice=str(exc); self.emit('error',message=str(exc))
                if self.external: self.engine.stop(str(exc))
        if self.recorder:
            if self.recorder.error: raise OSError('CSV 写入失败：'+self.recorder.error)
            if now-self.last_sensor>2 or (self.record_wait and now>self.record_wait):
                raise OSError('传感器超过 2 秒没有有效数据')
            if self.resume_external and self.record_wait is None:
                self.resume_external=None; self._finish_external()
            if self.pending_recipe and self.record_wait is None:
                recipe=self.pending_recipe; self.pending_recipe=None
                self.engine.owner=None
                self.engine.run(recipe)

    def _publish(self):
        snap=self.engine.snapshot()
        snap.update(realtime_forward=dict(enabled=self.realtime.running,clients=self.realtime.clients,
                    dropped=self.realtime.dropped,error=self.realtime_error or self.realtime.error),motor_port=self.motor_port,sensor_port=self.sensor_port,
                    sensor_alive=time.monotonic()-self.last_sensor<2,
                    baseline=self.baseline,calibrating=self.collector is not None,
                    sensor_values=tuple(v-b for v,b in zip(self.last_frame.values,self.baseline)) if self.last_frame and self.baseline is not None else None,
                    manual_reference=self._reference_state(),
                    restarting=bool(self.external and self.external['kind']=='restart_sensor'),
                    history=list(self.history),record_owned=self.record_owned,
                    samples=self.recorder.count if self.recorder else 0,
                    bad_lines=self.decoder.bad_lines,saved_path=self.saved_path,notice=self.notice,
                    demo=bool(self.sim or self.sensor_demo))
        if self.batch:
            snap['batch']={k:self.batch[k] for k in ('id','round','count','completed','phase','stop_after')}
            snap['batch']['active']=True; snap['idle']=False
            if not snap.get('owner'): snap['owner']='batch_wait'
        else: snap['batch']=dict(self.last_batch)
        with self.lock: self.latest=snap

    def run(self):
        try:
            while not self.exit_event.is_set():
                now=time.monotonic(); dt=now-self.previous_cycle; self.previous_cycle=now
                try:
                    self._check_writer_failures()
                    if self.stop_event.is_set():
                        self.stop_event.clear(); self.collector=None
                        self.engine.stop('用户停止全部')
                        while True:
                            try: self.commands.get_nowait()
                            except queue.Empty: break
                    if self.release_event.is_set(): self.release_event.clear(); self.engine.release()
                    for _ in range(16):
                        try: generation,name,payload=self.commands.get_nowait()
                        except queue.Empty: break
                        if generation!=self.generation: continue
                        previous_capture=self.capture
                        try: self._handle(name,payload); self.notice=''
                        except ValueError as exc:
                            self.notice=str(exc); self.emit('error',message=str(exc))
                            if name in ('run_record','batch_record') and self.capture is not None and self.capture is not previous_capture:
                                self.engine.stop(str(exc))
                    self._receive_motor(now,dt)
                    self._tick_external(now)
                    self._receive_sensor(now)
                    with self.lock: gui_lease=self.gui_lease
                    if gui_lease>self.engine.lease_at: self.engine.lease_at=gui_lease
                    self.engine.tick()
                    self._tick_batch(now)
                    if self.motor and self.handshake_complete and not self.engine.ready: self._close_motor()
                except Exception as exc:
                    self.notice=str(exc); self.emit('error',message=str(exc))
                    # Cancellation must not depend on a successful physical STOP write.
                    self._finish_batch('aborted',str(exc))
                    self.collector=None
                    try: self._close_motor()
                    except Exception: self.engine.ready=False; self.engine.status=None
                    self._end_record('aborted',str(exc))
                    if self.sensor:
                        try: self.sensor.close()
                        except Exception: pass
                    self.sensor=None; self.sensor_port=None; self.sensor_demo=False
                if now-self.last_publish>=.04:
                    self._publish(); self.last_publish=now
                self.exit_event.wait(.005)
        finally:
            self._finish_batch('aborted','工作台关闭')
            try: self._close_motor()
            except Exception as exc: self.emit('error',message=str(exc))
            self._end_record('aborted','工作台关闭')
            self.realtime.stop()
            if self.sensor:
                try: self.sensor.close()
                except Exception: pass
            self._publish()

    def shutdown(self):
        self.exit_event.set()
        if self.is_alive(): self.join(timeout=3)
        for finalizer in self.finalizers: finalizer.join(timeout=6)
