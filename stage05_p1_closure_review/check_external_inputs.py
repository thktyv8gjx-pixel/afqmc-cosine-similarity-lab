"""Report formal-data prerequisites without opening any archive or Manifest."""
import argparse,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    freeze=json.loads((a.root/'stage03b_official_validation/pre_dev_freeze_manifest.v2.json').read_text(encoding='utf-8'))
    required=[{'path':'stage01_data_audit/raw/afqmc_public.zip','sha256':freeze['inputs']['archive_sha256']},{'path':'stage01_data_audit/split_manifest.csv','sha256':freeze['inputs']['manifest_sha256']}]
    for x in required:x['present']=(a.root/x['path']).is_file()
    missing=[x['path'] for x in required if not x['present']]
    print(json.dumps({'status':'MISSING_EXTERNAL_PREREQUISITES' if missing else 'PRESENT_NOT_YET_HASH_OR_RIGHTS_VERIFIED','required':required,'files_opened':0,'official_dev_authorized':False,'warning':'Presence does not establish provenance, matching SHA-256, or legal permission. No automatic download, Manifest regeneration, fit, or Dev evaluation.'},indent=2))
    return 2 if missing else 0
if __name__=='__main__':raise SystemExit(main())
