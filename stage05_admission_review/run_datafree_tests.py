"""Run original synthetic suites in a copied tree without AFQMC or Git history."""
import importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1];O=R/'stage05_admission_review';L=O/'local_only'
L.mkdir(exist_ok=True)
tree=L/('source_only_test_tree_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));tree.mkdir(exist_ok=False)
freeze=json.loads((R/'stage03b_official_validation/pre_dev_freeze_manifest.v2.json').read_text(encoding='utf-8'))
files=set(freeze['frozen_source_files_sha256'])|{'stage02_math_protocol/tests/__init__.py','stage03a_internal_experiment/tests/__init__.py','stage03a_internal_experiment/src/__init__.py','stage03b_official_validation/pre_dev_freeze_manifest.json','stage03b_official_validation/pre_dev_freeze_manifest.v2.json'}
for name in files:
 target=tree/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/name,target)
temp=L/'synthetic_tmp';temp.mkdir(exist_ok=True)
child=r'''import sys,json,unittest,importlib.util,zipfile
from pathlib import Path
root=Path.cwd();sys.path[:0]=[str(root/'stage03a_internal_experiment/src'),str(root/'stage03b_official_validation'),str(root/'stage02_math_protocol')]
events=[];original=zipfile.ZipFile.open
def guarded(z,name,*args,**kwargs):
    filename=Path(z.filename).resolve();member=name.filename if isinstance(name,zipfile.ZipInfo) else name
    if not filename.is_relative_to(Path(__import__('os').environ['TMP']).resolve()):raise AssertionError('Only temporary synthetic ZIP members are permitted')
    events.append(str(member));return original(z,name,*args,**kwargs)
zipfile.ZipFile.open=guarded
cases=[];groups={};excluded=[]
for label,source in [('math','stage02_math_protocol/tests/test_math_validation.py'),('implementation','stage03a_internal_experiment/tests/test_stage3a.py'),('official_helpers','stage03b_official_validation/test_official_dev_runner.py'),('recovery','stage03b_official_validation/test_recovery_runner.py')]:
    spec=importlib.util.spec_from_file_location('admission_'+label,root/source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def walk(suite):
        for t in suite:
            if isinstance(t,unittest.TestSuite):yield from walk(t)
            else:yield t
    selected=[]
    for t in walk(unittest.defaultTestLoader.loadTestsFromModule(module)):
        if t._testMethodName=='test_zip_loader_opens_only_train_member':excluded.append(t.id())
        else:selected.append(t)
    groups[label]=len(selected);cases.extend(selected)
result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(cases))
record={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'groups':groups,'excluded_train_archive_test_count':len(excluded),'original_v1_git_proof_not_run':True,'zip_member_reads':events,'raw_afqmc_member_reads':0,'source_tree_contains_afqmc_or_git':False,'environment':'Existing frozen interpreter; copied sources only, not clean installed environment'}
Path('synthetic_test_result.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');raise SystemExit(0 if result.wasSuccessful() else 1)
'''
(tree/'admission_synthetic_runner.py').write_text(child,encoding='utf-8')
env=os.environ.copy();env.update(TMP=str(temp.resolve()),TEMP=str(temp.resolve()),PYTHONIOENCODING='utf-8',PYTHONUTF8='1');env.pop('PYTHONPATH',None)
start=datetime.now(timezone.utc).isoformat()
p=subprocess.run([sys.executable,'admission_synthetic_runner.py'],cwd=tree,env=env,capture_output=True,text=True,encoding='utf-8')
(L/'synthetic_test_log.txt').write_text(p.stdout+p.stderr,encoding='utf-8')
record=json.loads((tree/'synthetic_test_result.json').read_text(encoding='utf-8'));record.update(started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode)
(O/'datafree_test_result.json').write_bytes((json.dumps(record,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in record.items() if k!='zip_member_reads'},ensure_ascii=False))
assert p.returncode==0
