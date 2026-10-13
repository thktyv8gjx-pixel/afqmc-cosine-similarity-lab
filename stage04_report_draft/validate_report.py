"""Independent report checks using saved aggregate files, never model/Dev input."""
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    numbers=read(HERE/'report_numbers.json'); vals=numbers['values']; checks=[]
    for rel,wanted in numbers['source_sha256'].items():
        assert digest(ROOT/rel)==wanted,rel
    checks.append('all_saved_source_bytes_unchanged')
    cache={}
    for entry in numbers['source_fields']:
        p=ROOT/entry['source']; pointer=entry['pointer']
        if p.suffix=='.csv':
            if str(p) not in cache:
                with p.open(encoding='utf-8',newline='') as h: cache[str(p)]=list(csv.DictReader(h))
            _,_,row,key=pointer.split('/'); value=float(cache[str(p)][int(row)-1][key])
        else:
            if str(p) not in cache: cache[str(p)]=read(p)
            value=cache[str(p)]
            for segment in pointer.lstrip('/').split('/'): value=value[int(segment)] if isinstance(value,list) else value[segment]
        assert value==entry['value'],entry
    checks.append('all_number_source_pointers_match')
    dev=read(ROOT/'stage03b_official_validation/results/dev_metrics.json')
    dev={**dev['metrics'],**dev['constant_baselines']}
    report=(HERE/'报告正文初稿_公开安全版.md').read_text(encoding='utf-8')
    names={'count_char2':'Count','tfidf_char2':'TF-IDF','always_predict_0':'恒预测0','always_predict_1':'恒预测1'}
    for m,v in dev.items():
        assert vals['official_dev'][m]==v
        tp,fp,fn,tn=[v[k] for k in ['tp','fp','fn','tn']]
        assert sum([tp,fp,fn,tn])==4316 and tp+fn==1338 and fp+tn==2978
        fullrow='| '+names[m]+' | '+' | '.join(str(v[k]) if k in ['tp','fp','fn','tn'] else f'{v[k]:.6f}' for k in ['tp','fp','fn','tn','precision','recall','f1','accuracy'])+' |'
        assert fullrow in report,(m,'main table')
        auxrow='| '+names[m]+' | '+' | '.join('—' if v[k] is None else f'{v[k]:.6f}' for k in ['false_positive_rate','specificity','balanced_accuracy','roc_auc','average_precision','predicted_positive_proportion'])+' |'
        assert auxrow in report,(m,'aux table')
        denom=2*tp+fp+fn; f1=2*tp/denom if denom else 0.
        assert math.isclose(f1,v['f1'],abs_tol=1e-14)
    checks.append('four_dev_table_rows_and_confusion_formulas_match')
    a,b=dev['count_char2'],dev['tfidf_char2']; delta=vals['derived']
    assert delta['fp_reduction_count_minus_tfidf']==a['fp']-b['fp']==455
    assert delta['fn_increase_tfidf_minus_count']==b['fn']-a['fn']==127
    assert delta['correct_increase_tfidf_minus_count']==b['tp']+b['tn']-a['tp']-a['tn']==328
    assert delta['f1_tfidf_minus_count']==b['f1']-a['f1']
    assert delta['f1_tfidf_minus_all1']==b['f1']-dev['always_predict_1']['f1']
    assert f"{b['false_positive_rate']*100:.2f}" in report
    assert f"{100*b['fp']/(b['tp']+b['fp']):.2f}" in report
    checks.append('aggregate_differences_and_fpr_fdr_denominators_match')
    f=read(ROOT/'stage03b_official_validation/pre_dev_freeze_manifest.v2.json')
    for m in ['count_char2','tfidf_char2']:
        assert dev[m]['threshold'].hex()==f['thresholds'][m]['value'].hex()
        assert repr(dev[m]['threshold']) in report
    checks.append('two_full_precision_threshold_values_match')
    rec=read(HERE/'figures/plot_generation_record.json'); assert rec['raw_dev_read'] is False
    for fname,entry in rec['figures'].items(): assert digest(HERE/'figures'/fname)==entry['sha256']
    plotted=rec['figures']['01_四种方法指标比较.png']['plot_values']
    for m,row in plotted.items():
        for key,value in row.items(): assert value==dev[m][key]
    matrix=rec['figures']['02_TFIDF混淆矩阵.png']['plot_values']['matrix_rows_true_columns_predicted']
    assert matrix==[[b['tn'],b['fp']],[b['fn'],b['tp']]]
    v=rec['figures']['03_正负分布统一纵轴.png']['plot_values']
    hist=read(ROOT/'stage03b_official_validation/results/plot_aggregate_distribution_data.json')
    assert v['histograms']==hist and v['bin_count']==24
    for key,row in hist.items():
        assert sum(row['counts'])==row['sample_count']
        assert row['bin_edges']==hist['count_char2__label0']['bin_edges']
        assert max(row['counts'])/row['sample_count']<=v['common_ymax']
    comparison=rec['figures']['04_Tune与Dev对照.png']['plot_values']
    for scope in ['internal_tune','official_dev']:
        for m,value in comparison[scope].items(): assert value==vals[scope][m]['f1']
    checks.append('four_plot_files_and_underlying_values_match_common_distribution_scale')
    required=read(HERE/'report_consistency_check.json')['required_headings']
    assert re.findall(r'^#{1,3} (\d[^\n]*)',report,re.M)==required
    assert not re.search(r'\$(?:dev_|tune_|case_|threshold_)',report)
    for rel in re.findall(r'!\[[^\]]*\]\(([^)]*)\)',report): assert (HERE/rel).is_file()
    checks.append('full_structure_and_all_image_links_resolve')
    localcases=ROOT/'stage03b_official_validation/local_only/manual_case_review.json'
    restricted=[]
    if localcases.exists():
        saved=read(localcases)
        for row in saved['cases']: restricted += [row['sentence1'],row['sentence2'],row['source_row_id']]
    public_files=[p for p in HERE.rglob('*') if p.is_file() and 'local_only' not in p.relative_to(HERE).parts and p.name!='报告正文初稿.md' and '__pycache__' not in p.parts]
    for p in public_files:
        if p.suffix in ('.png',): continue
        text=p.read_text(encoding='utf-8')
        assert not re.search(r'dev:dev\.json:\d{6}',text),p
        assert not re.search(r'[CD]:[\\/]',text),p
        for raw in restricted: assert raw not in text,p
    ignored=subprocess.run(['git','check-ignore','--quiet','stage04_report_draft/报告正文初稿.md'],cwd=ROOT)
    assert ignored.returncode==0
    checks.append('public_files_exclude_saved_original_cases_ids_and_machine_paths_local_draft_ignored')
    diffs=subprocess.check_output(['git','diff','--name-only','-z'],cwd=ROOT).split(b'\0')
    for path in diffs:
        if path: assert path.decode()=='.gitignore' or path.decode().startswith('stage04_report_draft/'),path
    checks.append('historical_tracked_files_not_modified')
    references=read(HERE/'references.json'); assert len(references['references'])==4 and all(r['verified'] for r in references['references'])
    checks.append('four_primary_references_verified')
    result={'passed':True,'checks_passed':len(checks),'checks':checks,'source_field_count':len(numbers['source_fields']),
            'new_afqmc_experiments':0,'original_dev_opens':0,'model_tests_rerun':0,
            'plot_visual_review':'Recorded separately after actual inspection; numerical checks do not prove legibility'}
    (HERE/'validation_results.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__': main()
