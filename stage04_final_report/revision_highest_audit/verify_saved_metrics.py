"""Recheck saved predictions and aggregates; never load or score raw Dev text.

Run in the frozen experiment environment. Reads pre-existing output CSV only.
"""
import csv, hashlib, json, math
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    freeze=read(ROOT/'stage03b_official_validation/pre_dev_freeze_manifest.v2.json')
    protected=freeze['frozen_source_files_sha256']
    for name,want in protected.items():assert sha(ROOT/name)==want,name
    registry=read(ROOT/'stage04_report_draft/report_numbers.json');numbers=registry['values']
    source_cache={}
    for name,want in registry['source_sha256'].items():
        assert sha(ROOT/name)==want,name
        if name.endswith('.json'):source_cache[name]=read(ROOT/name)
        else:
            with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:source_cache[name]=list(csv.DictReader(f))
    for field in registry['source_fields']:
        name=field['source'];keys=field['pointer'].strip('/').split('/');value=source_cache[name]
        if name.endswith('.csv'):
            assert keys[0]=='data_row';value=float(value[int(keys[1])-1][keys[2]])
        else:
            for key in keys:
                key=key.replace('~1','/').replace('~0','~');value=value[int(key)] if isinstance(value,list) else value[key]
        assert value==field['value'],field['pointer']
    saved=read(ROOT/'stage03b_official_validation/results/dev_metrics.json')
    assert saved==read(ROOT/'stage03b_official_validation/outputs/dev_metrics.json')
    official={**saved['metrics'],**saved['constant_baselines']}
    assert official==numbers['official_dev']
    predpath=ROOT/'stage03b_official_validation/outputs/dev_predictions_local_only.csv'
    with predpath.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==8632
    results={};groups={m:[r for r in rows if r['method']==m] for m in ['count_char2','tfidf_char2']}
    reference_ids=None;reference_y=None
    for m,g in groups.items():
        ids=[r['source_row_id'] for r in g];y=np.array([int(r['label']) for r in g]);p=np.array([int(r['prediction']) for r in g]);s=np.array([float(r['score']) for r in g])
        assert len(g)==4316 and len(set(ids))==4316 and np.isfinite(s).all()
        assert np.all(p==(s>=freeze['thresholds'][m]['value']))
        assert all(float(r['threshold'])==freeze['thresholds'][m]['value'] for r in g)
        if reference_ids is not None:assert ids==reference_ids and np.array_equal(y,reference_y)
        reference_ids,reference_y=ids,y
        tn,fp,fn,tp=map(int,confusion_matrix(y,p,labels=[0,1]).ravel())
        x=dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=float(precision_score(y,p,zero_division=0)),recall=float(recall_score(y,p,zero_division=0)),f1=float(f1_score(y,p,zero_division=0)),accuracy=float(accuracy_score(y,p)),false_positive_rate=fp/(fp+tn),specificity=tn/(tn+fp),balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,roc_auc=float(roc_auc_score(y,s)),average_precision=float(average_precision_score(y,s)),predicted_positive_proportion=float(p.mean()))
        for k,val in x.items():assert math.isclose(val,official[m][k],rel_tol=0,abs_tol=2e-15),(m,k)
        for label in [0,1]:
            a=s[y==label];dist=next(d for d in numbers['distributions'] if d['method']==m and d['label']==label)
            expected={'min':a.min(),'max':a.max(),'mean':a.mean(),'median':np.median(a),'p05':np.quantile(a,.05),'p25':np.quantile(a,.25),'p75':np.quantile(a,.75),'p95':np.quantile(a,.95)}
            for k,val in expected.items():assert math.isclose(float(val),dist[k],rel_tol=0,abs_tol=2e-15),(m,label,k)
            hist=read(ROOT/'stage03b_official_validation/results/plot_aggregate_distribution_data.json')[m+'__label'+str(label)]
            assert np.histogram(a,bins=hist['bin_edges'])[0].tolist()==hist['counts']
        results[m]=x
    for constant in [0,1]:
        m='always_predict_'+str(constant);p=np.full(4316,constant);y=reference_y
        tn,fp,fn,tp=map(int,confusion_matrix(y,p,labels=[0,1]).ravel())
        x=dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=float(precision_score(y,p,zero_division=0)),recall=float(recall_score(y,p,zero_division=0)),f1=float(f1_score(y,p,zero_division=0)),accuracy=float(accuracy_score(y,p)),false_positive_rate=fp/(fp+tn),specificity=tn/(tn+fp),balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,predicted_positive_proportion=float(p.mean()))
        for k,val in x.items():assert math.isclose(val,official[m][k],rel_tol=0,abs_tol=2e-15),(m,k)
        results[m]=x
    result={'passed':True,'report_number_source_bindings_verified':len(registry['source_fields']),'report_source_files_hash_verified':len(registry['source_sha256']),'saved_row_count':len(rows),'dev_sample_count':4316,'protected_source_file_count':len(protected),'official_dev_model_runs':0,'raw_dev_archive_reads':0,'saved_prediction_sha256':sha(predpath),'report_numbers_sha256':sha(ROOT/'stage04_report_draft/report_numbers.json'),'metric_source_sha256':sha(ROOT/'stage03b_official_validation/results/dev_metrics.json'),'metrics':results,'all_distribution_quantiles_and_histogram_bins_match':True,'maximum_absolute_metric_tolerance':2e-15}
    (OUT/'saved_metrics_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='metrics'}))
if __name__=='__main__':main()
