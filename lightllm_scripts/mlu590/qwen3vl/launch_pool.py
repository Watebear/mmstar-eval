"""Launch only this round's isolated instances after the explicit A100 gate."""
import argparse, json, os, pathlib, signal, socket, subprocess, time, urllib.request
ROOT=pathlib.Path('/tmp/wushuo-mlu-qwen3vl8b-20261106'); SERV=ROOT/'serving'; REC=SERV/'records'
def save(p,d):
 q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(d,indent=2)+'\n'); q.replace(p)
def healthy(card):
 try:
  with urllib.request.urlopen(f'http://127.0.0.1:{18200+card}/get_model_name',timeout=3) as r: d=json.load(r)
  return d.get('model_name')=='Qwen3-VL-8B-Instruct'
 except (OSError,ValueError): return False
def launch(cards):
 assert json.load(open(REC/'a100_pilot_pass.json'))['passed']
 assert json.load(open(REC/'model_cache_verified.json'))['all_files_sha256_verified']
 pending=[]
 for card in cards:
  directory=SERV/'logs'/f'qwen3vl8b-{card}'; directory.mkdir(parents=True,exist_ok=True); record=directory/'process.json'
  if record.exists():
   previous=json.load(open(record))
   try: os.kill(previous['pid'],0); exists=True
   except ProcessLookupError: exists=False
   if exists:
    assert previous['round']=='qwen3vl8b-20261106'
    pending.append((card,previous['pid'])); continue
  for port in [18200+card,30200+card]:
   deadline=time.monotonic()+240
   while True:
    with socket.socket() as probe:
     if probe.connect_ex(('127.0.0.1',port))==0:raise RuntimeError(f'Port {port} has a live listener outside the recorded instance')
    try:
     with socket.socket() as probe:probe.bind(('127.0.0.1',port))
     break
    except OSError:
     if time.monotonic()>deadline:raise TimeoutError(f'Port {port} did not leave TIME_WAIT within240seconds')
     print(json.dumps({'waiting_for_port_reuse':port}),flush=True);time.sleep(5)
  with (directory/'console.log').open('ab') as log:
   p=subprocess.Popen(['/bin/bash',str(SERV/'start_instance.sh'),str(card)],cwd=SERV,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save(record,{'round':'qwen3vl8b-20261106','pid':p.pid,'pgid':p.pid,'card':card,'port':18200+card,'nccl_port':30200+card,'started_at':time.time()})
  pending.append((card,p.pid))
 start=time.monotonic(); ready={}
 while len(ready)<len(pending):
  for card,pid in pending:
   if card in ready: continue
   try: os.kill(pid,0)
   except ProcessLookupError: raise RuntimeError(f'Card {card} service {pid} exited; inspect own console.log')
   if healthy(card):
    with urllib.request.urlopen(f'http://127.0.0.1:{18200+card}/get_server_info',timeout=5) as r: info=json.load(r)
    ready[card]={'card':card,'port':18200+card,'pid':pid,'server_info':info,'ready_at':time.time()}
    save(REC/f'service_ready.card{card}.json',ready[card]); print(json.dumps({'ready_card':card,'seconds':time.monotonic()-start}),flush=True)
  if time.monotonic()-start>1200: raise TimeoutError('New Qwen3-VL instances did not become healthy within20min')
  if len(ready)<len(pending): time.sleep(5)
 save(REC/'pool.latest.json',{'cards':cards,'ready':ready,'updated_at':time.time()})
 return ready
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--cards',type=int,nargs='+',required=True); a=p.parse_args(); assert all(0<=n<=7 for n in a.cards); launch(a.cards)
