"""Stage 3B aggregate delivery from saved results only; no archive access."""
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'stage03b_official_validation'
O=S/'outputs'
P=S/'results'
sys.path.insert(0,str(S))
from recheck_saved_results import recheck_saved_results
from recovery_runtime import method_consistency
import run_official_dev as runner

def load(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def doc(name,text): (S/name).write_text(text,encoding='utf-8',newline='\n')
def num(v): return '不适用' if v is None else f'{v:.6f}'
def csv_write(p,rows):
    with p.open('w',encoding='utf-8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader(); w.writerows(rows)

def main():
    v1=load(S/'pre_dev_freeze_manifest.json'); v2=load(S/'pre_dev_freeze_manifest.v2.json')
    runner.verify_frozen_code(v2)
    consistent=method_consistency(v1,v2); assert consistent['passed']
    q=recheck_saved_results(O,v2); assert q['passed']
    m=load(O/'dev_metrics.json'); metrics={**m['metrics'],**m['constant_baselines']}
    baseline=metrics['always_predict_1']; count=metrics['count_char2']; tfidf=metrics['tfidf_char2']
    tune=load(ROOT/'stage03a_internal_experiment/outputs/metrics_summary.json')['methods']
    rankings=load(ROOT/'stage03a_internal_experiment/qa_review/outputs/qa_diagnostics.json')['original_char2_diagnostics']
    attempt=load(S/'local_only/attempts/attempt_002/result.json')
    prior=load(S/'local_only/attempts/attempt_001/record.json')
    gate=load(S/'local_only/part_a_publication_record.v2.json')
    plot=load(O/'plot_rebuild_record.json'); plotcheck=load(O/'plot_consistency_check.json')
    review=load(S/'local_only/manual_case_review.json'); assert len(review['cases'])==12
    original_checks={}
    for line in (ROOT/'stage03a_internal_experiment/outputs/SHA256SUMS').read_text().splitlines():
        expected,relative=line.split(None,1)
        original_checks[relative]=sha(ROOT/'stage03a_internal_experiment'/relative.strip())==expected
    assert len(original_checks)==20 and all(original_checks.values())
    P.mkdir(exist_ok=True); (S/'figures').mkdir(exist_ok=True)
    for name in ('dev_metrics.json','dev_metrics.csv','independent_recheck.json','plot_rebuild_record.json','plot_consistency_check.json','plot_aggregate_distribution_data.json'):
        shutil.copyfile(O/name,P/name)
    for p in sorted((O/'figures').glob('*.png')): shutil.copyfile(p,S/'figures'/p.name)
    csv_write(P/'score_distribution_summary.csv',q['score_distributions'])
    comparisons=[]
    for method in ('count_char2','tfidf_char2'):
        t=tune[method]; d=metrics[method]
        comparisons.append({'method':method,'tune_n':t['sample_count'],'dev_n':d['sample_count'],
            'tune_f1':t['f1'],'dev_f1':d['f1'],'f1_delta_dev_minus_tune':d['f1']-t['f1'],
            'tune_precision':t['precision'],'dev_precision':d['precision'],'tune_recall':t['recall'],'dev_recall':d['recall'],
            'tune_auc':rankings[method]['roc_auc'],'dev_auc':d['roc_auc'],
            'tune_ap':rankings[method]['average_precision'],'dev_ap':d['average_precision']})
    csv_write(P/'internal_tune_vs_dev.csv',comparisons)
    events=[load(p) for p in sorted((S/'local_only/attempts/attempt_002').glob('0*.json'))]
    dump(P/'attempt_002_execution_summary.json',{**attempt,'review_status':'recovery completed, awaiting user review',
        'stages':events,'attempt_001_approximate_log_last_write_utc':prior['file_observed_last_write_utc'],
        'exact_attempt_001_process_time':'unknown; filesystem timestamp only','original_dev_replay_performed':False})
    dump(P/'publication_gate_verification.json',{'passed_before_dev_read':True,'gate':gate,
        'local_gate_sha256':sha(S/'local_only/part_a_publication_record.v2.json'),
        'authorization_record_sha256':sha(S/'local_only/recovery_authorization.json'),
        'attempt_001_record_sha256':sha(S/'local_only/attempts/attempt_001/record.json')})
    validation={'passed':True,'original_stage3a_outputs_20_verified':original_checks,'v1_v2_method_consistency':consistent,
        'saved_row_recheck_passed':q['passed'],'four_figure_rebuild_passed':plotcheck['passed'],
        'public_metrics_byte_identical_to_local_metrics':sha(P/'dev_metrics.json')==sha(O/'dev_metrics.json'),
        'public_figures_match_saved_result_build':all(sha(S/'figures'/Path(v['path']).name)==v['sha256'] for v in plot['figures'].values()),
        'manual_case_count':12,'manual_theme_population_counts':'not estimated','real_dev_reopened_for_delivery':False,
        'source_version_fingerprints_still_pass':True,'original_v1_manifest_bytes_unchanged':sha(S/'pre_dev_freeze_manifest.json')==v2['recovery']['v1_manifest_raw_bytes_sha256']}
    assert validation['public_metrics_byte_identical_to_local_metrics'] and validation['public_figures_match_saved_result_build']
    dump(P/'delivery_consistency_check.json',validation)
    fields=['tp','fp','fn','tn','precision','recall','f1','accuracy','false_positive_rate','specificity','balanced_accuracy','roc_auc','average_precision','predicted_positive_proportion']
    table='| 指标 | Count | TF-IDF主方案 | 恒0 | 恒1 |\n|---|---:|---:|---:|---:|\n'
    for k in fields:
        table+='| '+k+' | '+' | '.join(str(r[k]) if k in ('tp','fp','fn','tn') else num(r[k]) for r in metrics.values())+' |\n'
    start=events[0]['recorded_utc']; end=attempt['recorded_utc']
    start_local=datetime.fromisoformat(start).astimezone(ZoneInfo('Asia/Shanghai')).isoformat()
    end_local=datetime.fromisoformat(end).astimezone(ZoneInfo('Asia/Shanghai')).isoformat()
    doc('02_官方Dev评价结果.md',f"""# 官方 Dev 恢复评价结果

状态：attempt_002 完整评价完成，等待用户审核。此前 attempt_001 部分访问后失败；本次为工程恢复评价，不称首次盲评。开始/结束（Asia/Shanghai）：{start_local} / {end_local}。精确阶段时间见 results/attempt_002_execution_summary.json。

原冻结提交：{prior['original_freeze_commit_sha']}；v2冻结提交：{attempt['freeze_commit_sha']}。Dev 4,316行，标签0/1为2,978/1,338，全行保留，正类1表示相似。唯一fit问句54,330，共享词表35,242；Dev只transform，无fit或阈值搜索。

Count阈值 {count['threshold']!r}，TF-IDF阈值 {tfidf['threshold']!r}，均来自原内部tune。规则 score >= threshold。表格显示6位小数，机器JSON保留完整精度。

{table}

混淆矩阵按 [[TN,FP],[FN,TP]]：Count={[[count['tn'],count['fp']],[count['fn'],count['tp']]]}，TF-IDF={[[tfidf['tn'],tfidf['fp']],[tfidf['fn'],tfidf['tp']]]}。恒0/恒1分别 [[2978,0],[1338,0]] / [[0,2978],[0,1338]]。

两模型左/右/任一零向量、NaN/Inf和越界分数均为0；没有丢弃难例。常量标签无连续排序分数，ROC-AUC/AP按预登记规则为不适用。聚合数值保存在results/dev_metrics.json与.csv；原始句对与逐行结果只保存在本地忽略目录。
""")
    compare='| 方法 | tune F1 | Dev F1 | 差值 | tune AUC | Dev AUC | tune AP | Dev AP |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in comparisons: compare+='| '+r['method']+' | '+' | '.join(num(r[k]) for k in ('tune_f1','dev_f1','f1_delta_dev_minus_tune','tune_auc','dev_auc','tune_ap','dev_ap'))+' |\n'
    doc('03_Count与TFIDF正式对照分析.md',f"""# Count 与 TF-IDF 正式对照

{compare}

Dev没有明显下降；TF-IDF比tune高{tfidf['f1']-tune['tfidf_char2']['f1']:.6f}，Count高{count['f1']-tune['count_char2']['f1']:.6f}。tune同时用于选阈值与评价，不能当独立泛化估计；本次固定流程Dev结果也需披露部分访问后的工程恢复。

TF-IDF比Count的F1高{tfidf['f1']-count['f1']:.6f}，比全正类基线高{tfidf['f1']-baseline['f1']:.6f}；Count比全正类仅高{count['f1']-baseline['f1']:.6f}。不能仅拿恒0的F1=0夸大效果。TF-IDF的FPR仍{tfidf['false_positive_rate']:.2%}、预测正类比例{tfidf['predicted_positive_proportion']:.2%}；Count分别{count['false_positive_rate']:.2%}、{count['predicted_positive_proportion']:.2%}。

AUC分别{count['roc_auc']:.6f}/{tfidf['roc_auc']:.6f}，描述性地高于0.5但排序信号弱；没有统计显著性分析。AP分别{count['average_precision']:.6f}/{tfidf['average_precision']:.6f}，略高于正类占比{baseline['precision']:.6f}，Balanced Accuracy仍近0.5。

共同fit语料、列索引、规范化、行与标签一致，差别为计数/TF-IDF加权及各自原tune阈值。主方案始终TF-IDF，Count基础对照；未据Dev重选模型或阈值，char2/3未纳入。
""")
    dist='| 方法 | 标签 | 均值 | 中位数 | P25 | P75 | 最小 | 最大 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in q['score_distributions']: dist+='| '+r['method']+' | '+str(r['label'])+' | '+' | '.join(num(r[k]) for k in ('mean','median','p25','p75','min','max'))+' |\n'
    doc('04_高误报与识别局限分析.md',f"""# 高误报与客服场景局限

Count有{count['fp']}个FP和{count['fn']}个FN；TF-IDF有{tfidf['fp']}个FP和{tfidf['fn']}个FN。这是全部真实标签与冻结预测的精确错误总量。

{dist}

正负分数区间及四分位范围严重重叠。Count阈值0.1066低于负例P05约0.126，导致96.71%负例预测相似；TF-IDF阈值0.05574位于负例P05与P25间，仍误报81.43%。阈值是内部tune F1优化的冻结结果，本次没有新扫描。TF-IDF负例最大约0.9533、正例最小约0.00289。

从保存的句对人工查看12例：TF-IDF最高分5个FP、最低分5个FN，加2个否定线索筛出的高分FP。观察到产品/账号实体变化与长共同模板、否定和故障状态差异；同意图的账单、退款、时间、账号归属改写则可能漏检。部分高分标签0对的字面意义接近，缺上下文存在标注粒度/歧义，不据此重标。真实ID、原句、分数、标签、预测和解释保存在local_only/04_真实错误案例人工分析.md。

这12例是人工案例分析，不能作为全体错误类型分布。数字、金额、时间、否定、实体主题没有可靠整体计数；不把关键词线索冒充语义分类。字符保留不意味着执行数值或实体语义推理。

可作为数学教学或简单候选检索基线保留，不能宣称是可靠真实客服分类器。实际客服自动回复和账户/支付决策需要另外的任务数据、成本指标与验证，本阶段不扩展方法。
""")
    doc('05_正式结果可视化与一致性核验.md',f"""# 图表与保存结果一致性

指标源SHA-256：{plot['source_metrics_sha256']}；预测源SHA-256：{plot['source_predictions_sha256']}。四图来自同一保存源，未再次读取原始Dev。

1. [指标对照](figures/dev_metrics_comparison.png)
2. [TF-IDF混淆矩阵](figures/dev_tfidf_confusion_matrix.png)
3. [正负余弦分布](figures/dev_score_distributions.png)
4. [tune与Dev F1](figures/internal_tune_vs_dev_f1.png)

图注明集合、方法、正类、N与阈值来源。分布按标签组内比例、各面板自身峰值缩放，跨面板大小应参考数值分布表。24分箱计数各自总和等于标签组实际行数，保存为results/plot_aggregate_distribution_data.json。

人工查看全部四图，混淆553/2425/152/1186及F1/FPR标注与机器源一致。先留存第一轮图，再只从保存源重建一次，四张PNG字节哈希完全一致，见results/plot_consistency_check.json与plot_rebuild_record.json。这不是原始Dev或模型重放。
""")
    doc('06_独立复现与代码审计.md',"""# 测试、独立复算与实现审计

代码位置：recovery_runtime.py:62 validate_thresholds严格解析；run_official_dev.py:301 compute_run读取前预校验，:309复用normalized_pair_texts并核对ID；recovery_runtime.py:107 AttemptRecorder不可覆盖；run_official_dev.py:428记录正式各阶段与异常，:206核对v2、事故/授权、字节/规范化摘要及Git门禁。

normalize_text及Stage3A向量器、余弦、阈值、指标源码未变。Dev打开前只按原内部fit重建语料/词表/IDF；打开后只transform，没有fit、fit_transform或阈值搜索。原Stage3A20个输出SHA全部重新核对相同。

Stage2 21、Stage3A14、原Stage3B4、新增11，合计50项通过；原源码两用例分别预期1个TypeError和1个规范化失败，证明测试能暴露原缺陷。初始沙盒因临时目录权限产生20个fixture/cleanup错误，已记录环境失败并申请执行权限后完整通过。测试进程启动层拦截真实Dev，真实Dev读取尝试0。真实日志和JSON位于recovery，所用问句全是人工合成。

独立recheck_saved_results不调用项目classification_metrics，不读原始归档：从保存行ID/标签/score/threshold/prediction复核行序、完整行数、float64位值、预测方向、零向量、有限性和文件哈希，独立sklearn重算四方案指标与混淆矩阵，交叉核对JSON/CSV。全部通过，混淆数完全相同，浮点容差1e-12。

交付时清单/源码摘要仍通过，v1实际字节未改，活跃摘要64位且真实复算一致。旧62位辅助摘要保留历史并追加更正，未作为活跃门禁。v1工作CRLF与Git blob LF的字节差异已记录两种字节摘要；跨机器复现要审查行尾，不能把内容摘要当原字节摘要跳过核验。

本轮没有原始Dev重放；后续完整模型重放和独立机器复现需明确另行授权。
""")
    doc('07_Stage03B最终验收.md',f"""# Stage 3B 恢复交付审核

状态：**STAGE03B_RECOVERY_READY_FOR_USER_REVIEW**。工程恢复、评价与一致性核验完成，最终阶段验收等待用户审核。

attempt_001=failed_after_partial_dev_access，精确进程时间未知、文件时间近似与证据边界完整保留；attempt_002=completed_recovery_evaluation，{start_local}开始，{end_local}结束。13个协议字段与保护源码、精确阈值机器核对相同。原提交和v1文件未覆盖；普通修复冻结提交{attempt['freeze_commit_sha']}。

4,316行完成，四方案混淆/指标独立复算一致，零向量及异常值0。TF-IDF F1={tfidf['f1']:.6f}，全正类={baseline['f1']:.6f}，优势仅{tfidf['f1']-baseline['f1']:.6f}，FPR={tfidf['false_positive_rate']:.6f}、AUC={tfidf['roc_auc']:.6f}；仍是区分能力有限。

P0：当前无已知实现、数据对齐、活跃摘要阻断；两运行器问题修复并有合成回归证明。

P1：严重高误报及弱排序区分；原数据再分发许可尚未核清，公开排除原文/逐行结果但许可说明仍要保留，不能宣称生产就绪。

P2：Train/Dev问句对重叠0，共享170规范化单句，不能称完全句子隔离；人工错误主题仅12个案，未统计全体主题。首次部分访问及工程恢复须在最终报告披露。独立机器重放和v1字节行尾复现尚待后续明确授权。

具备进入Stage4的审核条件，可把不理想结果和工程恢复如实作为实验结论；实际Stage4写作与Stage5打包未启动。用户审核前不宣布最终验收通过。

索引：本目录02–07文档；recovery/01与02事故/v2冻结审计；v2清单；results机器证据；figures四图。原01文档保留为v1历史快照。真实句对、预测、人工案例都仅本地。
""")
    readme=ROOT/'README.md'; text=readme.read_text(encoding='utf-8')
    old='Official Dev evaluation is a separate, single authorized step.'
    start_old=text.index(old) if old in text else text.index('The controlled Stage 3B recovery evaluation is complete.')
    end_old=text.index('\n## ',start_old)
    fence=chr(96)*3
    replacement="""The controlled Stage 3B recovery evaluation is complete. Attempt 001 accessed Dev partially and failed; version 2 repairs nested threshold parsing and restores registered normalization. Attempt 002 completed under explicit recovery authorization. See [recovery evidence](stage03b_official_validation/recovery/02_v2冻结与恢复门槛.md) and [Stage 3B review](stage03b_official_validation/07_Stage03B最终验收.md). No further Dev run is authorized by these instructions.

For a Train-fit-only preflight, select v2 explicitly:
"""
    replacement+=fence+'powershell\npython stage03b_official_validation/run_official_dev.py --freeze-manifest stage03b_official_validation/pre_dev_freeze_manifest.v2.json\n'+fence+'\n'
    text=text[:start_old]+replacement+text[end_old:]
    marker='\n## Official Dev recovery results (awaiting user review)'
    if marker in text: text=text[:text.index(marker)]
    text+=marker+'\n\n| Method | F1 | ROC-AUC | AP | FPR |\n|---|---:|---:|---:|---:|\n'
    for method,r in metrics.items(): text+='| '+method+' | '+' | '.join('n/a' if r[k] is None else f'{r[k]:.6f}' for k in ('f1','roc_auc','average_precision','false_positive_rate'))+' |\n'
    text+=f"\nTF-IDF exceeds the always-positive F1 by only {tfidf['f1']-baseline['f1']:.6f}. All 4,316 rows are retained. High false-positive rates and weak separation remain limitations. Stage 4/5 have not started. Only reviewed aggregate evidence is public; questions, predictions and real cases remain local.\n"
    readme.write_text(text,encoding='utf-8',newline='\n')
    files=['README.md','scripts/build_stage3b_delivery.py']
    for p in P.iterdir():
        if p.is_file() and p.name!='SHA256SUMS': p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    files += [p.relative_to(ROOT).as_posix() for p in sorted(P.iterdir()) if p.is_file() and p.name!='SHA256SUMS']
    files += [p.relative_to(ROOT).as_posix() for p in sorted((S/'figures').glob('*.png'))]
    files += [p.relative_to(ROOT).as_posix() for p in sorted(S.glob('*.md')) if p.name[:2] in ('02','03','04','05','06','07')]
    files += ['scripts/public_stage3b_results_allowlist.json']
    (P/'SHA256SUMS').write_text('\n'.join(sha(ROOT/p)+'  '+p for p in files if p!='scripts/public_stage3b_results_allowlist.json')+'\n',encoding='utf-8',newline='\n')
    files.append('stage03b_official_validation/results/SHA256SUMS')
    dump(ROOT/'scripts/public_stage3b_results_allowlist.json',{'scope':'reviewed aggregate delivery only, no row-level source content','files':files})
    print(json.dumps({'delivery_ready':True,'public_changed_files':len(files),'independent_recheck':q['passed'],'original_stage3a_20_checks':True}))

if __name__=='__main__': main()
