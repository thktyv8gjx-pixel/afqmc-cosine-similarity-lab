# Chinese Customer-Service Question Similarity with Cosine Similarity

This project studies a narrow mathematical question: can character n-gram vectors and cosine similarity distinguish semantically similar Chinese customer-service questions? The experiment compares a character 2-gram CountVectorizer baseline with a character 2-gram TF-IDF model. It is an educational benchmark, not a deployed support system.

## Data source and use limits

The experiments use the AFQMC split distributed by the CLUE benchmark. The source archive is available from the [AFQMC download](https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip); the benchmark documentation is in the [CLUE repository](https://github.com/CLUEbenchmark/CLUE). The checked source pages did not specify redistribution permission for the data. Obtain the archive yourself, review the applicable terms, and do not commit the archive, extracted JSON, row-level questions, split manifest, or predictions containing examples.

The expected archive SHA-256 and member fingerprints are recorded in `stage03b_official_validation/pre_dev_freeze_manifest.json`. SHA-256 values identify the inputs; they do not grant redistribution rights. The repository intentionally has no software license file. Public visibility does not select a code license or change the dataset's terms.

## Methods

Both methods use the same vocabulary of character 2-grams and the same paired cosine calculation. The formal primary method is TF-IDF; Count is the fixed baseline. TF-IDF document frequency is fitted from the 54,330 unique, normalized question strings from internal fit only. The text normalizer applies Unicode NFC, trims surrounding whitespace, and collapses whitespace runs to one ASCII space. It preserves punctuation, digits, English letters, case, and negation.

Each method uses its threshold selected on the internal tune split by positive-class F1, with the frozen rule `score >= threshold`. These thresholds are not probabilities. Do not retune them on official Dev.

## 环境安装与运行边界

当前跨平台课程入口与准确安装命令见 [delivery/README.md](delivery/README.md)。
使用原生 Python 3.12.10 的独立虚拟环境；Mac ARM64 和 Windows x64 分别使用各自的
完整九包 wheel SHA 锁。不要将 Windows 锁用于 Mac，也不要改用其他数值库版本。

```sh
python scripts/verify_course_delivery.py
```

默认运行无数据验证并输出新日志/摘要。`--mode prepare`、`--mode train` 和
`--mode frozen` 分别表示数据准备、显式隔离 Train 实验、历史冻结结果核验。
支持状态以 [本轮验收记录](public_evidence/course_delivery/README.md) 和真实 Actions 为准。
旧阶段记录保留历史定位，不能替代本轮原生测试或实际报告文件。

### A. 无原始数据的数学测试与合成演示

```powershell
python scripts/check_afqmc_synthetic.py
python scripts/test_prepare_afqmc.py
```

第一条核对已登记的v1/v2规范化清单摘要、34个冻结源码及三个合成辅助文件的原始字节哈希，建立不含`.git`、原始数据和私有报告的39文件临时快照，运行原49项合成测试及自有两句语料的Count/TF-IDF余弦演示。第二条运行37项数据准备与入口合成测试（模拟网络故障、错版本、CRC、字段和Manifest门禁），不接触真实AFQMC。两条均需要上面的数值依赖；第一条快照保存在已忽略的local_only目录，第二条使用系统临时目录并自动清理。合成验证成功不等于完整AFQMC结果复现成功。

### B. 官方数据下载、完整性校验和Manifest准备

```powershell
python scripts/prepare_afqmc.py
```

完整模式先检查依赖版本、Stage1源码指纹及既有Manifest；发现已知问题时在下载或读取归档前退出。随后从CLUE公布的固定URL下载到`stage01_data_audit/raw/afqmc_public.zip`。使用同目录临时文件，只有固定SHA-256、3成员CRC和成员指纹、Train/Dev/Test数量34,334/4,316/3,861及字段全部通过后才以原始字节发布到本地。Test保留整数`id`，没有label，不补造标签。已有正确归档直接校验，已有错误归档保留并报错；不切换镜像或数据版本；重定向在访问目标地址前被拒绝，不解压生成重复问句文件。

随后复用原Stage1的固定种子2026、分组五折首折及写出函数，生成完整42,511行`stage01_data_audit/split_manifest.csv`。固定列序、UTF-8 BOM及CRLF是字节复现协议的一部分。必须与目标SHA-256 `e856f4b9fd42be0ecfe530c002048933e31d971a3ba8d070a0c13daed241d907`一致，才接纳候选；已有正确文件保留并比对，已有错误文件不覆盖。新入口不调用旧审计主函数，因此不会额外生成含真实案例的重复/冲突报告。

```powershell
# 已手工从官方入口获取同一归档，或重复离线核验
python scripts/prepare_afqmc.py --offline
# 只下载并检查归档、CRC、数量和字段；不重建Manifest，仅需Python标准库
python scripts/prepare_afqmc.py --archive-only
```

标准库归档检查使用Python 3.11及以上；完整Manifest重建使用已记录的Python 3.12.10和依赖版本。默认根目录由脚本位置决定，不依赖终端当前目录；`--root`只用于明确指定的项目副本。归档检查会读取三个拆分进行数据结构审计，Manifest划分只使用Train；两者均不构造模型特征、不预测、不搜索阈值、不重评Dev。

官方下载地址和手工放置路径会在网络失败时打印，退出码为2；失败临时文件清理，半文件不冒充可用归档。目标归档SHA-256为`5a4cb1556b833010c329fa2ad2207d9e98fc94071b7e474015e9dd7c385db4dc`。原始归档与Manifest仍被`.gitignore`排除。详见[来源说明](DATASET_PROVENANCE.md)、[实际准备验证](public_evidence/data_preparation/README.md)和[最终检查记录](public_evidence/data_preparation/final_review.md)。

### C. 前提齐备后可单独运行的真实Train实验

仅在B完整成功、环境与配置核验通过、并明确希望开展新的内部Train实验时使用：

```powershell
python scripts/verify_course_delivery.py --mode train --output local_delivery/train-001
```

该入口使用Train内部fit/tune；不应作为数据安装或合成测试的一部分自动执行。输出为本地忽略目录；若已有历史输出或门禁阻止执行，停止并检查，不删除历史文件来绕过。新入口只在不存在的私有快照中运行；Mac 本轮独立 Train 结果及浮点差异见最新验收记录，历史词表、IDF规则、正式阈值和结果不变。

### D. 正式Dev的历史冻结与恢复边界

Stage3B已有一次部分访问失败和一次经授权的v2恢复评价，全部4,316条记录保留。v2只修复嵌套阈值解析和既定规范化调用；方法及精确阈值未改变。见[恢复证据](stage03b_official_validation/recovery/02_v2冻结与恢复门槛.md)和[正式评价审查](stage03b_official_validation/07_Stage03B最终验收.md)。

下载或重建Manifest不授予新的模型评价权限；本说明不提供默认执行Dev评价的命令，也不删除尝试记录、改发布门禁或重用旧授权。正式Dev完整重放需要另行明确授权及历史发布门禁核验；保存结果的逐行独立复算还需未公开的本地预测文件，不能从公开汇总恢复逐行记录。

## Current internal results and limitations

All values below are from internal tune, where the thresholds were selected. They are not independent validation results.

| Method | F1 | ROC-AUC | Average Precision | False-positive rate |
|---|---:|---:|---:|---:|
| Count character 2-gram + cosine | 0.472888 | 0.552451 | 0.349964 | 0.968434 |
| TF-IDF character 2-gram + cosine | 0.473543 | 0.572263 | 0.360341 | 0.814604 |
| Always predict positive | 0.470942 | n/a | n/a | 1.000000 |

The TF-IDF F1 is only about 0.00260 above the always-positive baseline. Its selected threshold predicts positive for about 83.41% of tune examples. The score distributions overlap substantially and the ranking signal is weak. These results do not show strong classification performance and must not be described as a successful real customer-service system.

Reviewed, data-free aggregate evidence is collected in [`public_evidence/`](public_evidence/): dataset fingerprints, internal-tune metrics, a QA summary, five reviewed QA notes, and the score-distribution figure. The full error-case note with actual question pairs is deliberately kept local.

## Project status

Stage 0 passed; Stage 1 is conditional pass; Stage 2 passed; Stage 3A and its QA review are conditional pass. The formal primary method and Count baseline are frozen before official Dev. Character 2–3-gram TF-IDF was a preregistered internal exploratory sensitivity run and is excluded from the formal Dev comparison. The report revision remains local; final course packaging and submission require separate review.

## Review and academic integrity

The repository contains code, aggregate results, and data-free audit evidence. Keep the source archive, row-level questions, predictions, and error examples local. Check all calculations and cited sources, document any AI assistance required by course policy, and make the final academic decisions and report your own. No result here establishes production readiness.

## Official Dev recovery results (awaiting user review)

| Method | F1 | ROC-AUC | AP | FPR |
|---|---:|---:|---:|---:|
| count_char2 | 0.474779 | 0.572598 | 0.362293 | 0.967092 |
| tfidf_char2 | 0.479289 | 0.584950 | 0.368599 | 0.814305 |
| always_predict_0 | 0.000000 | n/a | n/a | 0.000000 |
| always_predict_1 | 0.473293 | n/a | n/a | 1.000000 |

TF-IDF exceeds the always-positive F1 by only 0.005996. All 4,316 rows are retained. High false-positive rates and weak separation remain limitations. Report and source-delivery reviews have progressed; final course acceptance remains pending. Only reviewed aggregate evidence is public; questions, predictions and real cases remain local.
