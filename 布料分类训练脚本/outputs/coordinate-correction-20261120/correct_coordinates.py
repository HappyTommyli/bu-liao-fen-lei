"""User-authorized constant-X correction, with complete backups and cell checks."""
import csv,json,hashlib,os,shutil,io,sys
from decimal import Decimal
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
TARGET=Decimal('-24.348');SHIFT_MM=Decimal('24.348')
SOURCES=[ROOT/'xz-control/data/布料采集数据',ROOT/'xz-control/data/布料采集数据新']

def sha(data):return hashlib.sha256(data).hexdigest()
def load(path):
 raw=Path(path).read_bytes();rows=list(csv.reader(io.StringIO(raw.decode('utf-8-sig'),newline='')))
 return raw,rows

def inspect(path):
 p=Path(path);raw,rows=load(p);mp=p.parent/'metadata.json';mb=mp.read_bytes() if mp.exists() else None;meta=json.loads(mb) if mb else {}
 result=dict(source=str(p.relative_to(ROOT/'xz-control/data').parts[0]),label=meta.get('label',p.stem.rsplit('_',1)[0]),record=p.stem,path=str(p),metadata_path=str(mp),before_sha256=sha(raw),metadata_before_sha256=sha(mb) if mb else None,status='未修改',reason='',rows=len(rows)-1)
 header=rows[0];ix={k:i for i,k in enumerate(header)};ppmm=Decimal(str(meta.get('context',{}).get('config',{}).get('x_ppmm',250)))
 result['ppmm']=float(ppmm)
 if not all(k in ix for k in ['stage','motor_x_pulses','motor_target_x_pulses']):result['reason']='缺少必要位置字段';return result
 base=[];out=[];back=[]
 for row in rows[1:]:
  if len(row)!=len(header):result.update(reason='CSV列数异常，留待核对');return result
  try:x=int(row[ix['motor_x_pulses']])
  except ValueError:continue
  stage=row[ix['stage']]
  if stage=='step_002':base.append(x)
  elif stage=='step_003':out.append(x)
  elif stage=='step_004':back.append(x)
 if not out:result['reason']='无去程step_003';return result
 origin=Decimal(Counter(base).most_common(1)[0][0])/ppmm if base else Decimal(out[0])/ppmm
 result.update(baseline_mm=float(origin) if base else None,old_start_mm=float(Decimal(out[0])/ppmm),old_end_mm=float(Decimal(out[-1])/ppmm),old_xmin_mm=float(Decimal(min(out))/ppmm),old_xmax_mm=float(Decimal(max(out))/ppmm),old_return_end_mm=float(Decimal(back[-1])/ppmm) if back else None)
 if origin!=TARGET:result['reason']='起点不属于−24.348mm偏移';return result
 if not base or min(base)!=max(base):result['reason']='基线位置不恒定，留待核对';return result
 if abs(Decimal(out[0])/ppmm-origin)>Decimal('.1'):result['reason']='去程首点与基线位置不符，留待核对';return result
 if not mb or meta.get('outcome')!='completed':result['reason']='采集未明确完成，留待核对';return result
 if meta.get('coordinate_correction') is not None:result['reason']='已有坐标修改标记，留待核对';return result
 steps=meta.get('context',{}).get('recipe',{}).get('steps',[])
 if any(s.get('enabled',True) and s.get('axis')=='X' and s.get('kind') in ('absolute','zero','capture_start') for s in steps):result['reason']='含绝对坐标动作，留待核对';return result
 shift=SHIFT_MM*ppmm
 if shift!=int(shift):result['reason']='平移量不能整脉冲表示，留待核对';return result
 result.update(status='待平移',reason='基线恒定−24.348mm且去程起点一致；用户确认记录零点错误',shift_mm=float(SHIFT_MM),shift_pulses=int(shift),new_start_mm=float(Decimal(out[0])/ppmm+SHIFT_MM),new_end_mm=float(Decimal(out[-1])/ppmm+SHIFT_MM),new_return_end_mm=float(Decimal(back[-1])/ppmm+SHIFT_MM) if back else None)
 return result

def transform(raw,shift):
 rows=list(csv.reader(io.StringIO(raw.decode('utf-8-sig'),newline='')));header=rows[0];idx=[header.index('motor_x_pulses'),header.index('motor_target_x_pulses')]
 preserved_before=hashlib.sha256();preserved_after=hashlib.sha256();changed=0
 for row in rows[1:]:
  if len(row)!=len(header):raise ValueError('CSV列数异常')
  untouched=[v for i,v in enumerate(row) if i not in idx];preserved_before.update(json.dumps(untouched,ensure_ascii=False,separators=(',',':')).encode())
  for i in idx:
   if row[i]=='':continue
   value=Decimal(row[i]);assert value==int(value),'非整数脉冲'
   row[i]=str(int(value)+shift);changed+=1
  preserved_after.update(json.dumps([v for i,v in enumerate(row) if i not in idx],ensure_ascii=False,separators=(',',':')).encode())
 assert preserved_before.digest()==preserved_after.digest()
 stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\r\n');writer.writerows(rows)
 return stream.getvalue().encode('utf-8-sig'),dict(non_x_values_sha256=preserved_after.hexdigest(),non_x_values_unchanged=True,cells_shifted=changed,rows=len(rows)-1)

def atomic(path,data):
 tmp=path.with_name(path.name+'.coord-correction.tmp')
 with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.chmod(tmp,path.stat().st_mode & 0o777)
 os.replace(tmp,path)

def plan():
 assert not (OUT/'applied.json').exists(),'已执行过，禁止重复平移'
 paths=sorted({p.resolve() for root in SOURCES for p in root.rglob('*.csv')})
 result=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for r in pool.map(inspect,paths):
   result.append(r)
   if len(result)%200==0:print('SCAN',len(result),'/',len(paths),flush=True)
 summary={source:dict(Counter(r['status'] for r in result if r['source']==source)) for source in sorted({r['source'] for r in result})}
 (OUT/'plan.json').write_text(json.dumps(dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),shift_mm=float(SHIFT_MM),summary=summary,records=result),ensure_ascii=False,indent=2))
 print('PLAN',json.dumps(summary,ensure_ascii=False),flush=True)

def apply():
 assert not (OUT/'applied.json').exists(),'已执行过，禁止重复平移'
 doc=json.loads((OUT/'plan.json').read_text());todo=[r for r in doc['records'] if r['status']=='待平移'];backup=OUT/'原始备份';log=[]
 # Back up ALL candidates before replacing the first source file.
 for r in todo:
  for key,hashkey in [('path','before_sha256'),('metadata_path','metadata_before_sha256')]:
   p=Path(r[key]);assert sha(p.read_bytes())==r[hashkey],f'扫描后来源改变：{p}'
   dest=backup/p.relative_to(ROOT/'xz-control/data');dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():assert sha(dest.read_bytes())==r[hashkey]
   else:shutil.copy2(p,dest)
   assert sha(dest.read_bytes())==r[hashkey]
  r['backup_csv']=str(backup/Path(r['path']).relative_to(ROOT/'xz-control/data'));r['backup_metadata']=str(backup/Path(r['metadata_path']).relative_to(ROOT/'xz-control/data'))
 print('BACKUPS VERIFIED',len(todo),'CSV + metadata',flush=True)
 operation_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
 for r in todo:
  p=Path(r['path']);mp=Path(r['metadata_path']);original=Path(r['backup_csv']).read_bytes();meta_raw=Path(r['backup_metadata']).read_bytes()
  try:
   assert sha(p.read_bytes())==r['before_sha256'];assert sha(mp.read_bytes())==r['metadata_before_sha256']
   new,proof=transform(original,r['shift_pulses']);m=json.loads(meta_raw)
   m['coordinate_correction']=dict(version='x-translation-v1',applied_at=operation_at,shift_mm=r['shift_mm'],shift_pulses=r['shift_pulses'],columns=['motor_x_pulses','motor_target_x_pulses'],reason='用户确认记录零点错误，将基线−24.348mm整体平移到0mm',original_csv_sha256=r['before_sha256'],original_metadata_sha256=r['metadata_before_sha256'],backup_manifest=str(OUT/'applied.json'))
   atomic(p,new);atomic(mp,json.dumps(m,ensure_ascii=False,indent=2).encode('utf-8'))
   assert p.read_bytes()==new
   r.update(status='已平移',after_sha256=sha(new),metadata_after_sha256=sha(mp.read_bytes()),verification=proof)
   actual=inspect_after(r);r.update(actual)
   with (OUT/'journal.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
   log.append(r)
  except Exception:
   # Restore this pair and every previous pair if anything fails.
   for x in log+[r]:
    shutil.copy2(x['backup_csv'],x['path']);shutil.copy2(x['backup_metadata'],x['metadata_path'])
   raise
  if len(log)%25==0:print('APPLIED',len(log),'/',len(todo),flush=True)
 # Files not selected must still match their original hashes.
 unchanged=[r for r in doc['records'] if r['status']=='未修改']
 for r in unchanged:assert sha(Path(r['path']).read_bytes())==r['before_sha256']
 doc.update(completed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),backup_root=str(backup),modified_count=len(log),unchanged_count=len(unchanged),verification=dict(all_modified_non_x_values_unchanged=True,all_unchanged_csv_hashes_match=True))
 doc['summary']={source:dict(Counter(r['status'] for r in doc['records'] if r['source']==source)) for source in doc['summary']}
 (OUT/'applied.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2))
 print('DONE',json.dumps(doc['summary'],ensure_ascii=False),flush=True)

def inspect_after(r):
 raw,rows=load(r['path']);hdr=rows[0];stage=hdr.index('stage');xi=hdr.index('motor_x_pulses');tx=hdr.index('motor_target_x_pulses');_,old=load(r['backup_csv']);n=0
 for a,b in zip(old[1:],rows[1:]):
  for i in [xi,tx]:
   if a[i]:assert int(b[i])-int(a[i])==r['shift_pulses'];n+=1
 assert len(rows)==len(old)
 out=[int(x[xi]) for x in rows[1:] if x[stage]=='step_003'];base=[int(x[xi]) for x in rows[1:] if x[stage]=='step_002']
 assert min(base)==max(base)==0
 assert float(Decimal(out[0])/Decimal(str(r['ppmm'])))==r['new_start_mm']
 return dict(actual_new_baseline_mm=0.,actual_new_start_mm=out[0]/r['ppmm'],actual_new_end_mm=out[-1]/r['ppmm'],verified_cells_shifted=n)

if __name__=='__main__':
 if sys.argv[1]=='plan':plan()
 elif sys.argv[1]=='apply':apply()
 elif sys.argv[1]=='selftest':
  raw=b'\xef\xbb\xbfstage,motor_x_pulses,motor_target_x_pulses,dBy_uT,sample_valid\r\nstep_003,-6087,32663,-12.000,0\r\n';out,p=transform(raw,6087);rr=list(csv.reader(io.StringIO(out.decode('utf-8-sig'))));assert rr[1]==['step_003','0','38750','-12.000','0'];assert p['non_x_values_unchanged'];print('PASS translation, target coordinates, unchanged magnetic values and flags')
