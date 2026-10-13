"""Enabled only by the recovery test launcher; deny actual AFQMC Dev reads."""
import json
import os
from pathlib import Path
import zipfile

blocked=os.environ.get('RECOVERY_FORBIDDEN_ARCHIVE')
audit=os.environ.get('RECOVERY_GUARD_AUDIT')
if blocked:
    original=zipfile.ZipFile.open
    def guarded(archive,name,*args,**kwargs):
        member=name.filename if isinstance(name,zipfile.ZipInfo) else name
        if member=='dev.json':
            forbidden=Path(archive.filename).resolve()==Path(blocked).resolve()
            if audit:
                with open(audit,'a',encoding='utf-8') as handle:
                    handle.write(json.dumps({'dev_open':'forbidden_real' if forbidden else 'synthetic'})+'\n')
            if forbidden: raise AssertionError('real AFQMC Dev reads are forbidden during synthetic tests')
        return original(archive,name,*args,**kwargs)
    zipfile.ZipFile.open=guarded
