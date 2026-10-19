"""Execute the original 49 data-free tests against an explicitly staged tree.

Actual experiment test files remain unchanged. A raw-Train test and historical
Git proof are excluded explicitly; only temporary synthetic ZIP reads are legal.
"""
import argparse,importlib.util,json,os,sys,tempfile,unittest,zipfile
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();a.output.parent.mkdir(parents=True,exist_ok=True)
    assert not (root/'.git').exists() and not (root/'stage01_data_audit/raw').exists()
    sys.path[:0]=[str(root/'stage03a_internal_experiment/src'),str(root/'stage03b_official_validation'),str(root/'stage02_math_protocol')]
    tmpbase=Path(os.environ['TMP']).resolve();events=[];original=zipfile.ZipFile.open
    def guarded(z,name,*args,**kwargs):
        member=name.filename if isinstance(name,zipfile.ZipInfo) else str(name)
        if not Path(z.filename).resolve().is_relative_to(tmpbase):raise AssertionError('Only temporary synthetic ZIP reads permitted')
        events.append(member);return original(z,name,*args,**kwargs)
    zipfile.ZipFile.open=guarded
    suites=[];groups={};excluded=[]
    for group,path in [('math','stage02_math_protocol/tests/test_math_validation.py'),('implementation','stage03a_internal_experiment/tests/test_stage3a.py'),('official_helpers','stage03b_official_validation/test_official_dev_runner.py'),('recovery','stage03b_official_validation/test_recovery_runner.py')]:
        spec=importlib.util.spec_from_file_location('closure_'+group,root/path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        def flatten(s):
            for t in s:
                if isinstance(t,unittest.TestSuite):yield from flatten(t)
                else:yield t
        selected=[]
        for t in flatten(unittest.defaultTestLoader.loadTestsFromModule(mod)):
            if t._testMethodName=='test_zip_loader_opens_only_train_member':excluded.append(t.id())
            else:selected.append(t)
        groups[group]=len(selected);suites.extend(selected)
    assert groups=={'math':21,'implementation':13,'official_helpers':4,'recovery':11} and len(excluded)==1
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(suites))
    record={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'groups':groups,'excluded_real_train_tests':1,'git_history_proof_tests_not_run':2,'raw_afqmc_reads':0,'official_dev_evaluation':False,'synthetic_zip_member_reads':len(events),'uses_isolated_virtual_environment':sys.prefix!=sys.base_prefix,'python_version':sys.version.split()[0]}
    a.output.write_bytes((json.dumps(record,indent=2)+'\n').encode());print(json.dumps(record));return 0 if result.wasSuccessful() else 1
if __name__=='__main__':raise SystemExit(main())
