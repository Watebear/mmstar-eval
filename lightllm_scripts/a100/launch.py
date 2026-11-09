#!/opt/conda/bin/python
"""Replay an exact final serving command; qwen3 reuses Gemma's GPUs only after Gemma stops."""
import argparse
import datetime
import json
import os
from pathlib import Path

base=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('instance',choices=['gemma0','gemma1','qwen0','qwen1','qwen2','qwen3'])
parser.add_argument('--dry-run',action='store_true')
args=parser.parse_args()
record=json.loads((base/'records/launch_templates.json').read_text())[args.instance]
record.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),replay_from='records/launch_templates.json')
command=record['argv']
environment={**os.environ,**record['environment']}
if args.dry_run:
    print(json.dumps(record,indent=2))
    raise SystemExit(0)
for folder in ['pids','logs','records/restarts']:
    (base/folder).mkdir(parents=True,exist_ok=True)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
(base/'records/restarts'/f'{args.instance}.{stamp}.json').write_text(json.dumps(record,indent=2)+'\n')
(base/'pids'/f'{args.instance}.pid').write_text(str(os.getpid())+'\n')
print(json.dumps(record),flush=True)
os.chdir(record['source_snapshot'])
os.execve(command[0],command,environment)
