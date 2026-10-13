"""Independent metrics/identity verification from saved rows only; no archive access."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score,average_precision_score,balanced_accuracy_score,
    confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def recheck_saved_results(output: Path, freeze: dict):
    summary=json.loads((output/'dev_metrics.json').read_text(encoding='utf-8'))
    execution=json.loads((output/'execution_record.json').read_text(encoding='utf-8'))
    with (output/'dev_predictions_local_only.csv').open(encoding='utf-8',newline='') as h: rows=list(csv.DictReader(h))
    with (output/'dev_metrics.csv').open(encoding='utf-8',newline='') as h: csv_metrics={r['method']:r for r in csv.DictReader(h)}
    records=[json.loads(line) for line in (output/'dev_records_local_only.jsonl').read_text(encoding='utf-8').splitlines()]
    ids=[r['source_row_id'] for r in records]; labels=np.asarray([r['label'] for r in records],dtype=np.int8)
    n=freeze['inputs']['dev_rows']
    assert len(records)==n and len(set(ids))==n and len(rows)==2*n
    for name,expected in execution['result_fingerprints'].items(): assert sha(output/name)==expected
    assert summary['input_fingerprints']['freeze_manifest_sha256']==freeze['freeze_manifest_sha256']
    checked={}; distributions=[]
    for method,target in {**summary['metrics'],**summary['constant_baselines']}.items():
        selected=[r for r in rows if r['method']==method]
        if selected:
            assert [r['source_row_id'] for r in selected]==ids
            y=np.asarray([int(r['label']) for r in selected]); np.testing.assert_array_equal(y,labels)
            p=np.asarray([int(r['prediction']) for r in selected]); s=np.asarray([float(r['score']) for r in selected])
            assert np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
            t=float(freeze['thresholds'][method]['value'])
            assert all(float(r['threshold']).hex()==t.hex() for r in selected)
            np.testing.assert_array_equal(p,(s>=t).astype(int))
            for side in ('left','right'):
                count=sum(r['zero_vector_'+side]=='True' for r in selected)
                assert count==target['zero_vector_'+side+'_count']
            assert sum(r['zero_vector_left']=='True' or r['zero_vector_right']=='True' for r in selected)==target['zero_vector_pair_count']
            for label in (0,1):
                values=s[y==label]
                q=np.quantile(values,[0,.05,.25,.5,.75,.95,1])
                distributions.append({'method':method,'label':label,'count':len(values),'mean':float(values.mean()),
                    **{k:float(v) for k,v in zip(('min','p05','p25','median','p75','p95','max'),q)}})
        else:
            y=labels; p=np.full(n,int(method[-1])); s=None
        tn,fp,fn,tp=map(int,confusion_matrix(y,p,labels=[0,1]).ravel())
        result={'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':float(precision_score(y,p,zero_division=0)),
            'recall':float(recall_score(y,p,zero_division=0)),'f1':float(f1_score(y,p,zero_division=0)),
            'accuracy':float(accuracy_score(y,p)),'false_positive_rate':fp/(fp+tn),
            'specificity':tn/(fp+tn),'balanced_accuracy':float(balanced_accuracy_score(y,p)),
            'predicted_positive_proportion':float(np.mean(p)),
            'roc_auc':float(roc_auc_score(y,s)) if s is not None else None,
            'average_precision':float(average_precision_score(y,s)) if s is not None else None}
        for key,value in result.items():
            if value is None: assert target[key] is None and csv_metrics[method][key]==''
            else:
                assert math.isclose(value,target[key],rel_tol=0,abs_tol=1e-12),(method,key)
                assert math.isclose(value,float(csv_metrics[method][key]),rel_tol=0,abs_tol=1e-12)
        assert tp+fp+fn+tn==n
        checked[method]=result
    return {'passed':True,'source_prediction_sha256':sha(output/'dev_predictions_local_only.csv'),
        'source_metrics_sha256':sha(output/'dev_metrics.json'),'row_count':n,'label_counts':{'0':int((labels==0).sum()),'1':int((labels==1).sum())},
        'methods_independently_recomputed':checked,'score_distributions':distributions,
        'raw_dev_reopened':False,'checks':['row IDs/labels/order','threshold bits and prediction direction','all rows','zero vectors',
        'saved file hashes','independent sklearn metrics','JSON/CSV agreement','finite score range']}


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--freeze-manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    result=recheck_saved_results(a.output,json.loads(a.freeze_manifest.read_text(encoding='utf-8')))
    path=a.output/'independent_recheck.json'
    with path.open('x',encoding='utf-8') as h: json.dump(result,h,ensure_ascii=False,indent=2,allow_nan=False); h.write('\n')
    print(json.dumps({'passed':result['passed'],'row_count':result['row_count'],'label_counts':result['label_counts']}))
