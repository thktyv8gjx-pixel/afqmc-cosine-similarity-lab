"""Audit an explicit changed-file allowlist and Git-index bytes before publication."""
import argparse
import json
import subprocess
from pathlib import Path
from review_public_staging import SECRET_PATTERNS,FORBIDDEN_SUFFIXES

ROOT=Path(__file__).resolve().parents[1]
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--allowlist',type=Path,required=True); a=p.parse_args()
    expected=set(json.loads(a.allowlist.read_text(encoding='utf-8'))['files'])
    actual={x.decode() for x in git('diff','--cached','--name-only','-z').split(b'\0') if x}
    if actual!=expected: raise AssertionError('staged files differ from explicit allowlist: '+repr((sorted(actual-expected),sorted(expected-actual))))
    hits=[]
    for name in sorted(actual):
        if '/local_only/' in name or '/outputs/' in name or Path(name).suffix.lower() in FORBIDDEN_SUFFIXES:
            hits.append(name+': forbidden data/material path')
        payload=git('show',':'+name)
        for category,pattern in SECRET_PATTERNS.items():
            if pattern.search(payload): hits.append(name+': '+category)
    if hits: raise AssertionError('; '.join(hits))
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--quiet'],cwd=ROOT).returncode!=0: raise AssertionError('unstaged tracked edits remain')
    print(json.dumps({'passed':True,'staged_file_count':len(actual),'secret_or_restricted_hits':0,'exact_changed_file_allowlist':True}))
if __name__=='__main__': main()
