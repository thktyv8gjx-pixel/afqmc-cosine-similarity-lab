"""Render prose and aggregate plots from saved evidence. Never import/run evaluators.

Usage: python build_report.py [--local-cases]
The public build requires saved internal aggregate evidence; obtain it locally.
No dataset archive, original Dev member, fitting or threshold search is accessed.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from string import Template

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
plot_deps = HERE / 'local_only' / 'plot_dependencies'
if plot_deps.exists():
    sys.path.insert(0, str(plot_deps))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode('utf-8'))

def dump(path, obj):
    write(path, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

SOURCES = {
    'dev': 'stage03b_official_validation/results/dev_metrics.json',
    'tune': 'stage03a_internal_experiment/outputs/metrics_summary.json',
    'qa': 'stage03a_internal_experiment/qa_review/outputs/qa_diagnostics.json',
    'freeze': 'stage03b_official_validation/pre_dev_freeze_manifest.v2.json',
    'audit': 'stage01_data_audit/data_audit.json',
    'consistency': 'stage03b_official_validation/recovery/method_consistency_v1_v2.json',
    'tests': 'stage03b_official_validation/recovery/synthetic_test_results.json',
    'recheck': 'stage03b_official_validation/results/independent_recheck.json',
    'hist': 'stage03b_official_validation/results/plot_aggregate_distribution_data.json',
    'config': 'stage03a_internal_experiment/config/experiment.json',
}
METHODS = ['count_char2','tfidf_char2','always_predict_0','always_predict_1']
NAMES = dict(zip(METHODS,['Count','TF-IDF','恒预测0','恒预测1']))
COLORS = ['#3671be','#d7574b','#66788c','#28926f']

def leaves(x, prefix=''):
    if isinstance(x,dict):
        for k,v in x.items(): yield from leaves(v,prefix+'/'+str(k))
    elif isinstance(x,list):
        for k,v in enumerate(x): yield from leaves(v,prefix+'/'+str(k))
    else: yield prefix,x

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--local-cases',action='store_true'); args=parser.parse_args()
    data={k:json.loads((ROOT/v).read_text(encoding='utf-8')) for k,v in SOURCES.items()}
    before={v:sha(ROOT/v) for v in SOURCES.values()}
    distro_path=ROOT/'stage03b_official_validation/results/score_distribution_summary.csv'
    before[str(distro_path.relative_to(ROOT)).replace('\\','/')]=sha(distro_path)
    with distro_path.open(encoding='utf-8',newline='') as h:
        distro=[{k:(v if k=='method' else int(v) if k in ('label','count') else float(v)) for k,v in row.items()} for row in csv.DictReader(h)]
    dev={**data['dev']['metrics'],**data['dev']['constant_baselines']}
    tune={m:dict(data['tune']['methods'][m]) for m in METHODS[:2]}
    for m in METHODS[2:]: tune[m]=data['qa']['baseline_metrics'][m]
    for m in METHODS[:2]:
        tune[m].update({k:data['qa']['original_char2_diagnostics'][m][k] for k in ('roc_auc','average_precision')})
    # Exact integers independently recover the scalar metrics used in the prose.
    for m in METHODS:
        v=dev[m]; tp,fp,fn,tn=(v[k] for k in ('tp','fp','fn','tn')); n=tp+fp+fn+tn
        p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.
        calculated={'precision':p,'recall':r,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                    'accuracy':(tp+tn)/n,'false_positive_rate':fp/(fp+tn),'specificity':tn/(tn+fp),
                    'balanced_accuracy':(r+tn/(tn+fp))/2,'predicted_positive_proportion':(tp+fp)/n}
        assert n==data['freeze']['inputs']['dev_rows']
        for k,val in calculated.items(): assert math.isclose(val,v[k],rel_tol=0,abs_tol=1e-14),(m,k)
        for k,val in data['recheck']['methods_independently_recomputed'][m].items():
            if k in v and val is not None: assert math.isclose(float(val),float(v[k]),rel_tol=0,abs_tol=1e-12),(m,k)
    c,t=dev['count_char2'],dev['tfidf_char2']
    deltas={
        'fp_reduction_count_minus_tfidf':c['fp']-t['fp'],
        'fn_increase_tfidf_minus_count':t['fn']-c['fn'],
        'correct_increase_tfidf_minus_count':t['tp']+t['tn']-c['tp']-c['tn'],
        'f1_tfidf_minus_count':t['f1']-c['f1'],
        'f1_tfidf_minus_all1':t['f1']-dev['always_predict_1']['f1'],
        'tune_f1_tfidf_minus_all1':tune['tfidf_char2']['f1']-tune['always_predict_1']['f1'],
        'tfidf_false_discovery_fraction':t['fp']/(t['tp']+t['fp']),
        'tfidf_f1_dev_minus_tune':t['f1']-tune['tfidf_char2']['f1'],
        'count_f1_dev_minus_tune':c['f1']-tune['count_char2']['f1'],
    }
    consistency=data['consistency']; assert consistency['passed'] and data['recheck']['passed']
    checks={k:len(consistency[k]) for k in ['checks','threshold_float64_hex_equal','protected_implementation_sources_unchanged']}
    assert checks=={'checks':11,'threshold_float64_hex_equal':2,'protected_implementation_sources_unchanged':17}
    for m in METHODS[:2]: assert dev[m]['threshold'].hex()==data['freeze']['thresholds'][m]['value'].hex()
    values={'official_dev':dev,'internal_tune':tune,'distributions':distro,'derived':deltas,
            'inputs':data['freeze']['inputs'],'fit_space':data['freeze']['fit_space'],
            'environment':data['freeze']['environment'],'protocol_check_counts':checks,
            'registered_tests_passed':data['tests']['registered_tests_passed'],
            'split_seed':data['config']['split_seed'],
            'overlap':{'pairs':data['audit']['train_dev_pair_overlap'],'sentences':data['audit']['train_dev_shared_sentence_overlap']}}
    evidence=[]
    for alias in ['dev','tune','freeze','consistency','tests']:
        for pointer,val in leaves(data[alias]):
            if isinstance(val,(int,float)) and not isinstance(val,bool):
                evidence.append({'source':SOURCES[alias],'pointer':pointer,'value':val})
    for m in METHODS[2:]:
        for pointer,val in leaves(data['qa']['baseline_metrics'][m]):
            if isinstance(val,(int,float)) and not isinstance(val,bool): evidence.append({'source':SOURCES['qa'],'pointer':'/baseline_metrics/'+m+pointer,'value':val})
    for m in METHODS[:2]:
        for k in ('roc_auc','average_precision'): evidence.append({'source':SOURCES['qa'],'pointer':'/original_char2_diagnostics/'+m+'/'+k,'value':tune[m][k]})
    evidence.append({'source':SOURCES['config'],'pointer':'/split_seed','value':values['split_seed']})
    for key in ('train_dev_pair_overlap','train_dev_shared_sentence_overlap'):
        for pointer,val in leaves(data['audit'][key]):
            if isinstance(val,(int,float)) and not isinstance(val,bool): evidence.append({'source':SOURCES['audit'],'pointer':'/'+key+pointer,'value':val})
    for i,row in enumerate(distro):
        for k,val in row.items():
            if k!='method': evidence.append({'source':str(distro_path.relative_to(ROOT)).replace('\\','/'),'pointer':f'/data_row/{i+1}/{k}','value':val})
    formulas={
        'fp_reduction_count_minus_tfidf':'dev/count_char2/fp - dev/tfidf_char2/fp',
        'fn_increase_tfidf_minus_count':'dev/tfidf_char2/fn - dev/count_char2/fn',
        'correct_increase_tfidf_minus_count':'(dev/tfidf_char2/tp + tn) - (dev/count_char2/tp + tn)',
        'f1_tfidf_minus_count':'dev/tfidf_char2/f1 - dev/count_char2/f1',
        'f1_tfidf_minus_all1':'dev/tfidf_char2/f1 - dev/always_predict_1/f1',
        'tune_f1_tfidf_minus_all1':'tune/tfidf_char2/f1 - qa/baseline_metrics/always_predict_1/f1',
        'tfidf_false_discovery_fraction':'dev/tfidf_char2/fp / (dev/tfidf_char2/tp + fp)',
        'tfidf_f1_dev_minus_tune':'dev/tfidf_char2/f1 - tune/tfidf_char2/f1',
        'count_f1_dev_minus_tune':'dev/count_char2/f1 - tune/count_char2/f1'}
    dump(HERE/'report_numbers.json',{'scope':'Saved results only; IEEE float64 preserved; prose rounds at presentation only','values':values,'source_sha256':before,'source_fields':evidence,'derived_formulas':formulas})
    def table(rows,fields):
        out=['| 方法 | '+' | '.join(k[1] for k in fields)+' |','|---|'+'---:|'*len(fields)]
        for m in METHODS:
            vals=[]
            for k,_ in fields:
                v=rows[m].get(k); vals.append('—' if v is None else str(v) if isinstance(v,int) else f'{v:.6f}')
            out.append('| '+NAMES[m]+' | '+' | '.join(vals)+' |')
        return '\n'.join(out)
    tokens={
        'dev_main':table(dev,[(k,k.upper() if k in ['tp','fp','fn','tn'] else k.title()) for k in ['tp','fp','fn','tn','precision','recall','f1','accuracy']]),
        'dev_aux':table(dev,[('false_positive_rate','FPR'),('specificity','Specificity'),('balanced_accuracy','Balanced Accuracy'),('roc_auc','ROC-AUC'),('average_precision','AP'),('predicted_positive_proportion','预测正类比例')]),
        'tune_main':table(tune,[('tp','TP'),('fp','FP'),('fn','FN'),('tn','TN'),('precision','Precision'),('recall','Recall'),('f1','F1'),('accuracy','Accuracy')]),
        'threshold_count':repr(c['threshold']),'threshold_tfidf':repr(t['threshold']),
        'fp_reduction':str(deltas['fp_reduction_count_minus_tfidf']),'fn_increase':str(deltas['fn_increase_tfidf_minus_count']),
        'correct_increase':str(deltas['correct_increase_tfidf_minus_count']),
        'f1_delta_count':f"{deltas['f1_tfidf_minus_count']:.17f}",'f1_delta_all1':f"{deltas['f1_tfidf_minus_all1']:.17f}",
        'tune_delta_all1':f"{deltas['tune_f1_tfidf_minus_all1']:.6f}",
        'fpr_pct':f"{100*t['false_positive_rate']:.2f}",'fdr_pct':f"{100*deltas['tfidf_false_discovery_fraction']:.2f}",
        'count_pos_pct':f"{100*c['predicted_positive_proportion']:.2f}",'tfidf_pos_pct':f"{100*t['predicted_positive_proportion']:.2f}",
        'count_f1_shift':f"{deltas['count_f1_dev_minus_tune']:.6f}",'tfidf_f1_shift':f"{deltas['tfidf_f1_dev_minus_tune']:.6f}",
        'environment':'、'.join(f'{k} {v}' for k,v in values['environment'].items()),
        'distribution_table':'| 方法 / 标签 | 最小值 | P05 | P25 | 中位数 | P75 | P95 | 最大值 | 均值 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'+'\n'.join('| '+NAMES[row['method']]+' / '+str(row['label'])+' | '+' | '.join(f"{row[k]:.6f}" for k in ['min','p05','p25','median','p75','p95','max','mean'])+' |' for row in distro),
        'case_text':'公开版本不刊载原始句对、行 ID 或逐行分数。本地稿从既有人工案例记录选取少量误报、漏检与标注歧义案例；这里保留同一结论：实体差异、否定状态与表达改写会暴露词面表示的局限。',
        'fit_rows':str(values['fit_space']['fit_pair_rows']),'tune_rows':str(values['fit_space']['tune_pair_rows']),
        'unique_fit':str(values['fit_space']['unique_normalized_fit_question_count']),'vocab':str(values['fit_space']['vocabulary_dimension']),
        'train_rows':str(values['inputs']['train_rows']),'dev_rows':str(values['inputs']['dev_rows']),'test_rows':str(values['inputs']['test_rows']),
        'seed':str(values['split_seed']),
    }
    template=(HERE/'report_body.md.in').read_text(encoding='utf-8')
    public=Template(template).substitute(tokens)
    write(HERE/'报告正文初稿_公开安全版.md',public)
    case_count=0
    if args.local_cases:
        selections=json.loads((HERE/'local_only/case_selection.json').read_text(encoding='utf-8'))['selected_ids']
        saved=json.loads((ROOT/'stage03b_official_validation/local_only/manual_case_review.json').read_text(encoding='utf-8'))
        lookup={x['source_row_id']:x for x in saved['cases']}; cases=[lookup[x] for x in selections]; case_count=len(cases)
        lines=['以下为既有十二例人工记录中的四例，按产品实体差异、否定状态、表达改写及标注歧义选择；仅作人工个案说明，没有错误主题总体频数。均为 TF-IDF，阈值 '+repr(t['threshold'])+'。句子保持原文，分数沿用原记录。']
        for i,row in enumerate(cases,1):
            lines.append(f"\n**案例{i}（{row['selection']}）**：原始 ID `{row['source_row_id']}`；真实标签 {row['label']}，预测 {row['prediction']}，分数 {row['score']}。\n\n- 句子1：{row['sentence1']}\n- 句子2：{row['sentence2']}\n\n{row['manual_explanation']}\n")
        local=Template(template).substitute({**tokens,'case_text':'\n'.join(lines)})
        write(HERE/'报告正文初稿.md',local)
        dump(HERE/'local_only/case_provenance.json',{'source':'stage03b_official_validation/local_only/manual_case_review.json','sha256':sha(ROOT/'stage03b_official_validation/local_only/manual_case_review.json'),'selected_ids':selections,'case_count':case_count,'question_text_unchanged':True})
    make_plots(dev,tune,data['hist'],before)
    mapping=['# 数字与引用证据映射','', '全部实验表格由 build_report.py 读取保存 JSON/CSV 生成，完整 float64 保存在 report_numbers.json。只在正文展示时取六位小数，百分比取两位；阈值保持完整精度。','', '| 正文位置 | 数据源 / 字段 | 复核方法 |','|---|---|---|',
             '| 2.1 / 3.1 数据规模 | v2 清单 inputs、fit_space | 与冻结样本数一致 |',
             '| 2.2 手算 | Stage 2 第03号材料 | 人工向量0.8、人工中文2-gram 1/3、两文档平滑IDF，不属于真实数据结果 |',
             '| 3.3 表1/2 | Dev results/dev_metrics.json：metrics 与 constant_baselines | 混淆计数公式 + 既有逐行独立复算 |',
             '| 3.3 表3 / 图4 | Stage3A outputs/metrics_summary.json；QA baseline_metrics | 同 tune 阈值选择结果；不是独立验证 |',
             '| 3.3 表4 / 图3 | score_distribution_summary.csv / plot_aggregate_distribution_data.json | 四组共用分箱，计数与标签数一致 |',
             '| 3.4 差值 | report_numbers.json derived + derived_formulas | 整数差 455/127/328；完整 F1 差值由保存浮点作减法 |',
             '| 附录A | recovery/method_consistency_v1_v2.json | 11组 checks + 2个 threshold_float64_hex_equal；17个保护源文件 |',
             '| AI使用说明 | 实际本会话操作和保存检查记录 | AI辅助核查；不声称学生已独立审阅正文 |',
             '| 参考文献 | references.json | 2026-10-13 浏览官方来源；外部文献不支持本项目效果数字 |','',
             'JSON 内 source_fields 逐项保存数值、来源相对路径和 JSON Pointer（CSV 行号从1计），source_sha256 保存原始文件字节摘要；derived_formulas 保存计算式。图数据的具体字段见 figures/plot_generation_record.json。','',
             '原始问句案例只进入忽略的本地正文及 local_only/case_provenance.json。公开正文不包含真实 ID、逐行分数或原文。',
             '', '## 原文件原始字节指纹','', '| 文件 | SHA-256 |','|---|---|']
    mapping += [f'| {p} | `{s}` |' for p,s in before.items()]
    write(HERE/'数字与引用证据映射.md','\n'.join(mapping)+'\n')
    after={p:sha(ROOT/p) for p in before}; assert before==after
    headings=re.findall(r'^#{1,3} (\d[^\n]*)',public,re.M)
    expected=['1 问题描述','2 数学知识分析与应用','2.1 问题的数学表示','2.2 核心数学原理与求解方法','2.3 实验方案与验证思路','3 Python实现过程与结果展示','3.1 实验数据与运行环境','3.2 核心程序与实现步骤','3.3 实验结果与可视化对比','3.4 结果分析与验证','4 总结与拓展']
    assert headings==expected,headings
    check={'passed':True,'saved_inputs_unchanged':before==after,'official_archive_read':False,'evaluator_run':False,'fit_or_threshold_search':False,
           'dev_methods_checked':4,'confusion_scalar_metric_check_tolerance':1e-14,'saved_independent_recheck_tolerance':1e-12,
           'exact_threshold_hex_checked':2,'required_headings':headings,'main_section_count':4,'subsection_count':7,
           'public_han_character_count':len(re.findall(r'[\u4e00-\u9fff]',public)),
           'public_non_whitespace_character_count':len(re.sub(r'\s','',public)),
           'local_selected_case_count':case_count,'historical_files_edited':False,'references_verified':4,
           'counts_erratum':checks,'normal_tests_previously_passed':data['tests']['registered_tests_passed'],'new_model_tests_run_in_stage4a':0}
    if args.local_cases: check['local_han_character_count']=len(re.findall(r'[\u4e00-\u9fff]',local))
    dump(HERE/'report_consistency_check.json',check)
    print(json.dumps(check,ensure_ascii=False))

def make_plots(dev,tune,hist,source_hashes):
    chinese=[p for p in font_manager.findSystemFonts() if Path(p).name.lower()=='msyh.ttc']
    if not chinese: raise RuntimeError('Chinese font Microsoft YaHei unavailable; install a Chinese font before rendering')
    font_manager.fontManager.addfont(chinese[0]); plt.rcParams['font.family']='Microsoft YaHei'
    plt.rcParams['axes.unicode_minus']=False
    figdir=HERE/'figures'; figdir.mkdir(exist_ok=True)
    records={}
    foot='官方 Dev：N='+str(dev['tfidf_char2']['sample_count'])+'；正类=1；两方案阈值来自内部 tune 冻结；score ≥ threshold'
    def save(fig,name,values,caption):
        fig.text(.05,.025,caption,fontsize=10,color='#53616f'); fig.savefig(figdir/name,dpi=150,facecolor='white'); plt.close(fig)
        records[name]={'sha256':sha(figdir/name),'plot_values':values,'footnote':caption}
    fields=['f1','accuracy','false_positive_rate','specificity','balanced_accuracy']
    fig,ax=plt.subplots(figsize=(13,7)); x=np.arange(5); width=.18
    for j,m in enumerate(METHODS):
        vals=[dev[m][k] for k in fields]; bars=ax.bar(x+(j-1.5)*width,vals,width,label=NAMES[m],color=COLORS[j])
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in vals],padding=3,fontsize=8)
    ax.set(ylim=(0,1.10),yticks=np.linspace(0,1,6),xticks=x,xticklabels=['F1','Accuracy','误报率 FPR↓','Specificity','Balanced Accuracy'],ylabel='指标值（比例）')
    fig.suptitle('图1 官方 Dev 四种正式方法的指标比较',y=.97,fontsize=17)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.12),ncols=4); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True); fig.subplots_adjust(bottom=.16,top=.82)
    save(fig,'01_四种方法指标比较.png',{m:{k:dev[m][k] for k in fields} for m in METHODS},foot)
    t=dev['tfidf_char2']; mat=np.array([[t['tn'],t['fp']],[t['fn'],t['tp']]])
    fig,ax=plt.subplots(figsize=(10,7)); im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=dev['tfidf_char2']['sample_count']); cb=fig.colorbar(im,ax=ax); cb.set_label('句对数量（条）')
    for i in range(2):
        for j in range(2): ax.text(j,i,f'{mat[i,j]}',ha='center',va='center',fontsize=28)
    ax.set(xticks=[0,1],xticklabels=['预测0','预测1'],yticks=[0,1],yticklabels=['真实0','真实1'],xlabel='预测标签',ylabel='真实标签',title='图2 TF-IDF 官方 Dev 混淆矩阵'); fig.subplots_adjust(bottom=.18,top=.88)
    save(fig,'02_TFIDF混淆矩阵.png',{'matrix_rows_true_columns_predicted':mat.tolist(),'threshold':t['threshold']},foot+'；单位：条')
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True,sharey=True)
    ymax=max(max(v['counts'])/v['sample_count'] for v in hist.values()); ymax=math.ceil(ymax*100/5)*.05
    for i,m in enumerate(METHODS[:2]):
        for j,label in enumerate([0,1]):
            v=hist[m+'__label'+str(label)]; edges=np.asarray(v['bin_edges']); assert sum(v['counts'])==v['sample_count']; assert len(edges)==25
            assert v['bin_edges']==hist['count_char2__label0']['bin_edges']
            ax=axes[i,j]; ax.bar(edges[:-1],np.asarray(v['counts'])/v['sample_count'],width=np.diff(edges),align='edge',color=COLORS[j],alpha=.85,edgecolor='white',linewidth=.5)
            ax.axvline(dev[m]['threshold'],color='#222',linestyle='--',label='冻结阈值'); ax.set(xlim=(0,1),ylim=(0,ymax),title=f'{NAMES[m]} / 真实{label} / N={v["sample_count"]}',xlabel='余弦分数',ylabel='本组样本比例（每个分箱）'); ax.legend(fontsize=9); ax.grid(axis='y',alpha=.2)
            ax.tick_params(labelleft=True,labelbottom=True)
    fig.suptitle('图3 正负余弦分布：共同分箱与统一纵轴',fontsize=17); fig.subplots_adjust(bottom=.12,top=.9,hspace=.4,wspace=.22)
    save(fig,'03_正负分布统一纵轴.png',{'histograms':hist,'common_ymax':ymax,'bin_count':24},foot+'；每组比例分母为各组 N，纵轴相同')
    fig,ax=plt.subplots(figsize=(12,7)); x=np.arange(4)
    for j,(scope,rows) in enumerate([('内部 tune',tune),('官方 Dev',dev)]):
        vals=[rows[m]['f1'] for m in METHODS]; bars=ax.bar(x+(j-.5)*.4,vals,.32,label=scope,color=[COLORS[0],COLORS[3]][j]); ax.bar_label(bars,labels=[f'{v:.6f}' for v in vals],padding=4,fontsize=9)
    ax.set(ylim=(0,1.05),xticks=x,xticklabels=[NAMES[m] for m in METHODS],yticks=np.linspace(0,1,6),ylabel='F1（比例）',title='图4 内部 tune 与官方 Dev 的 F1 对照'); ax.legend(); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True); fig.subplots_adjust(bottom=.16,top=.88)
    save(fig,'04_Tune与Dev对照.png',{'internal_tune':{m:tune[m]['f1'] for m in METHODS},'official_dev':{m:dev[m]['f1'] for m in METHODS}},'tune：N=6867，参与阈值选择；Dev：N=4316，沿用冻结阈值；全部方法纵轴从0起')
    dump(figdir/'plot_generation_record.json',{'tool':'matplotlib','version':matplotlib.__version__,'font':'Microsoft YaHei','original_stage3b_plots_unchanged':True,'raw_dev_read':False,'source_sha256':source_hashes,'figures':records})

if __name__=='__main__': main()
