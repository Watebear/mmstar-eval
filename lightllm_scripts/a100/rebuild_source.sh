#!/bin/bash
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
target=${1:-/tmp/mmbench_a100_lightllm_0db819b929e7}
if [[ -d "$target" && -n "$(ls -A -- "$target")" ]]; then
  echo "Rebuild target must be new or empty: $target" >&2
  exit 1
fi
/opt/conda/bin/python - "$base" "$target" <<'PY'
import hashlib,json,pathlib,py_compile,shutil,subprocess,sys,tarfile
base,target=map(pathlib.Path,sys.argv[1:])
expected=json.loads((base/'records/rebuild_expected.json').read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
archive=base/'records/source_snapshot.tar.gz'
assert sha(archive)==expected['archive_sha256']
for name,checksum in expected['patches'].items():assert sha(base/'records'/name)==checksum
assert sha(base/'records/runtime_sitecustomize.py')==expected['files']['sitecustomize.py']
target.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive,'r:gz') as source:source.extractall(target,filter='data')
shutil.copy2(base/'records/runtime_sitecustomize.py',target/'sitecustomize.py')
for name in ['qwen_fused_gating_nameerror.patch','gemma_tokenizer_add_special_tokens.patch']:
    subprocess.run(['patch','--batch','--forward','-p1','-d',str(target),'-i',str((base/'records'/name).resolve())],check=True)
for relative,checksum in expected['files'].items():
    assert sha(target/relative)==checksum,relative
    py_compile.compile(str(target/relative),doraise=True)
print(json.dumps({'target':str(target),'archive_sha256':expected['archive_sha256'],'verified_files':expected['files']},indent=2))
PY
