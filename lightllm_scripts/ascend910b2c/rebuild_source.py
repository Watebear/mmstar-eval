"""Rebuild this run's source without touching the installed vendor packages."""
import argparse,hashlib,json,pathlib,shutil,subprocess,tarfile
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('target',type=pathlib.Path);a=p.parse_args();base=pathlib.Path(__file__).resolve().parent;materials=base/'provenance';expected=json.loads((materials/'rebuild_expected.json').read_text());runtime=json.loads((base/'records/ascend_runtime_patches.json').read_text())
assert not a.target.exists() or not any(a.target.iterdir()),'Target must be new/empty'
archive=materials/'source-original.tar.gz';assert sha(archive)==expected['archive_sha256'];a.target.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive) as t:t.extractall(a.target,filter='data')
shutil.copy2(materials/'runtime_sitecustomize.py',a.target/'sitecustomize.py')
for name,digest in expected['patches'].items():
    patch=materials/name;assert sha(patch)==digest
    subprocess.run(['patch','--batch','--forward','-p1','-d',str(a.target),'-i',str(patch)],check=True)
for name,digest in expected['files'].items():assert sha(a.target/name)==digest,name
entries=runtime['patches']+json.loads((base/'records/gemma_mm_prefill_patch.json').read_text())['patches']
for record in ['qwen_npu_prefill_patch.json', 'qwen_position_patch.json']:
    extra=base/'records'/record
    if extra.exists():entries+=json.loads(extra.read_text())['patches']
for entry in entries:
    relative=entry['file']
    if entry['original_sha256'] is None:assert not (a.target/relative).exists()
    else:assert sha(a.target/relative)==entry['original_sha256']
    patch=base/(relative.replace('/','_')+'.patch');assert sha(patch)==entry['patch_sha256']
    subprocess.run(['patch','--batch','--forward','-p1','-d',str(a.target),'-i',str(patch)],check=True)
    assert sha(a.target/relative)==entry['patched_sha256'],relative
full=materials/'source_tree_expected.json'
if full.exists():
    expected_tree=json.loads(full.read_text());generated=materials/'runtime-generated-files.tar.gz';assert sha(generated)==expected_tree['generated_archive_sha256'];assert not expected_tree['missing_original_files']
    with tarfile.open(generated) as t:t.extractall(a.target,filter='data')
    actual={str(f.relative_to(a.target)):sha(f) for f in a.target.rglob('*') if f.is_file() and '__pycache__' not in f.parts};assert actual==expected_tree['files'],[n for n in set(actual)|set(expected_tree['files']) if actual.get(n)!=expected_tree['files'].get(n)][:10]
print(json.dumps({'rebuilt':str(a.target),'commit':'0db819b929e713f9337723ff9dcd6a7a04924821','baseline_patches_verified':True,'ascend_runtime_patches_verified':True,'complete_tree_verified':full.exists()},indent=2))
