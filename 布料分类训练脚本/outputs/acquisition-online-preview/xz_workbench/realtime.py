"""Opt-in loopback NDJSON telemetry. Device/CSV paths never perform socket I/O."""
import json
import math
import queue
import select
import socket
import threading
import time


class FrameStream:
    """Derive wire fields from raw frames, never the recorder's global quality flags."""
    def __init__(self):
        self.context=None
        self.capture_id=None
        self.stage=None

    def frame(self,frame,*,capture_id,stage,status,x_ppmm,baseline,active):
        base=float(baseline[1]) if baseline is not None else None
        context=(capture_id,stage,frame.segment,tuple(baseline) if baseline is not None else None,active)
        packets=[]
        if self.context!=context:
            if self.capture_id is not None and self.capture_id!=capture_id:
                packets.append(self.end('capture_changed'))
            packets.append(dict(type='reset',capture_id=capture_id,stage=stage,reason='context_changed'))
            self.context=context
        self.capture_id=capture_id; self.stage=stage
        status=status or {}
        fault=status.get('fault')
        motor_fault=fault!='NONE'
        x=status.get('x')
        x_mm=float(x)/x_ppmm if x is not None and x_ppmm>0 else None
        reason=''
        if fault is None: reason='motor_status_unknown'
        elif motor_fault: reason='motor_fault'
        elif base is None: reason='baseline_unknown'
        elif not active or not capture_id: reason='not_recording'
        elif stage not in ('step_003','step_004'): reason='stage_not_classifiable'
        elif x_mm is None: reason='coordinate_unknown'
        elif not all(math.isfinite(v) for v in (float(frame.by),base,x_mm)): reason='nonfinite_sample'
        packets.append(dict(type='sample',capture_id=capture_id,stage=stage,
            t_s=frame.unwrapped_us/1e6,x_mm=x_mm,by_uT=float(frame.by),baseline_by_uT=base,
            valid=not reason,invalid_reason=reason,motor_fault=motor_fault,motor_fault_code=fault))
        return packets

    def end(self,reason):
        packet=dict(type='end',capture_id=self.capture_id,stage=self.stage,reason=reason)
        self.context=None; self.capture_id=None; self.stage=None
        return packet


class Publisher:
    """Bounded queue to one broadcaster; no network backpressure reaches capture.

    Slow peers are closed on a pending buffer >64 KiB or age >0.5 s. Overflow
    drops queued samples and places a reset before the latest sample.
    """
    def __init__(self,port=8766,queue_size=2048):
        self.port=port; self.items=queue.Queue(maxsize=max(2,queue_size))
        self._running=False; self._stop=threading.Event(); self.thread=None
        self.address=None; self.error=''; self.dropped=0; self.clients=0

    @property
    def running(self): return self._running

    def start(self):
        if self.running: return True
        self.error=''
        listener=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            listener.bind(('127.0.0.1',self.port)); listener.listen(4); listener.setblocking(False)
        except OSError as exc:
            listener.close(); self.error=str(exc); return False
        while True:
            try: self.items.get_nowait()
            except queue.Empty: break
        self.address=listener.getsockname(); self._stop=threading.Event(); self._running=True
        self.thread=threading.Thread(target=self._serve,args=(listener,self._stop),name='xz-realtime-publisher',daemon=True)
        self.thread.start(); return True

    def publish(self,packet):
        if not self.running: return False
        try:
            payload=(json.dumps(packet,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode('utf8')
            try: self.items.put_nowait(payload)
            except queue.Full:
                while True:
                    try: self.items.get_nowait(); self.dropped+=1
                    except queue.Empty: break
                reset=dict(type='reset',capture_id=packet.get('capture_id'),stage=packet.get('stage'),reason='publisher_overflow')
                self.items.put_nowait((json.dumps(reset)+'\n').encode())
                self.items.put_nowait(payload)
            return True
        except Exception as exc:
            self.error=str(exc); self.dropped+=1; return False

    def stop(self):
        self._running=False; self._stop.set()
        # No join here: stopping forwarding must not delay motor/recording work.

    def _serve(self,listener,stop):
        peers={}
        def drop(peer):
            peers.pop(peer,None)
            try: peer.close()
            except OSError: pass
            self.clients=len(peers)
        try:
            while not stop.is_set():
                readable,writable,_=select.select([listener]+list(peers),[p for p,state in peers.items() if state[0]],[],.01)
                if listener in readable:
                    peer,_=listener.accept(); peer.setblocking(False)
                    if len(peers)>=4: peer.close()
                    else:
                        peers[peer]=[bytearray(b'{"type":"reset","capture_id":null,"reason":"client_connected"}\n'),time.monotonic()]
                        self.clients=len(peers)
                payloads=[]
                for _ in range(256):
                    try: payloads.append(self.items.get_nowait())
                    except queue.Empty: break
                payload=b''.join(payloads)
                now=time.monotonic()
                for peer in list(peers):
                    pending,since=peers[peer]
                    if peer in readable:
                        try:
                            if not peer.recv(1024): drop(peer); continue
                        except BlockingIOError: pass
                        except OSError: drop(peer); continue
                    if payload:
                        if not pending: peers[peer][1]=now
                        pending.extend(payload)
                    if len(pending)>65536 or (pending and now-peers[peer][1]>.5): drop(peer); continue
                    if pending and peer in writable:
                        try: del pending[:peer.send(pending)]
                        except BlockingIOError: pass
                        except OSError: drop(peer)
        except Exception as exc:
            self.error=str(exc)
        finally:
            for peer in list(peers): drop(peer)
            listener.close()
            # A restarted publisher owns a different stop event/thread.
            if self._stop is stop: self._running=False
