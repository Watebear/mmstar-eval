"""Rebuild the unmodified LightLLM base and copied overlay without GPU loading."""
import argparse,hashlib,json,pathlib,tarfile
p=argparse.ArgumentParser();p.add_argument('--destination',required=True); a=p.parse_args(); here=pathlib.Path(__file__).resolve().parent; dest=pathlib.Path(a.destination); dest.mkdir(parents=True,exist_ok=True)
archive=here/'lightllm-runtime-source-0db819b.tar.gz'; expected='b18ec050716f408e5ef4754a4c474546b5901e76db49bd1f5bd67a34cec53b0f'; assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
with tarfile.open(archive) as t: t.extractall(dest/'lightllm-runtime-0db819b',filter='data')
with tarfile.open(here/'pydeps.tar.gz') as t: t.extractall(dest/'serving/pydeps',filter='data')
manifest=json.load(open(here/'runtime_files.json')); compared=0
for section,path in [('source',dest/'lightllm-runtime-0db819b'),('pydeps',dest/'serving/pydeps')]:
 for name,digest in manifest[section].items():
  assert hashlib.sha256((path/name).read_bytes()).hexdigest()==digest,name
  compared+=1
if (here/'runtime_delta.json').exists():
 delta=json.load(open(here/'runtime_delta.json'))
 for section,prefix in [('source','lightllm-runtime-0db819b'),('pydeps','serving/pydeps')]:
  for name in delta['deleted'][section]: (dest/prefix/name).unlink()
 with tarfile.open(here/'runtime_delta.tar.gz') as t:t.extractall(dest,filter='data')
 final=json.load(open(here/'runtime_final_files.json'));compared=0
 for section,prefix in [('source','lightllm-runtime-0db819b'),('pydeps','serving/pydeps')]:
  for name,digest in final[section].items():
   assert hashlib.sha256((dest/prefix/name).read_bytes()).hexdigest()==digest,name
   compared+=1
print(json.dumps({'verified':True,'files_compared':compared,'patched_files':[n for n in delta['changed']['source'] if n.endswith('.py')] if (here/'runtime_delta.json').exists() else [],'sitecustomize_used':False,'base_commit':'0db819b929e713f9337723ff9dcd6a7a04924821'}))
