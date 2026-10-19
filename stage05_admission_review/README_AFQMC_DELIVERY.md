# AFQMC源码交付运行说明

这是AFQMC专用的准入及未来交付说明，避免根目录其他实验README/依赖造成混淆。本轮未制作源码ZIP；下列安装/实验命令是未来在独立解压目录的操作说明，未据此重新执行正式实验。

## 目录与环境

保留`stage01_data_audit/`、`stage02_math_protocol/`、`stage03a_internal_experiment/src/`及config/tests、`stage03b_official_validation/`、冻结清单与其保护的scripts相对目录。确切文件白名单见`proposed_package_inventory.json`；不得直接压缩整个项目目录。

使用Python 3.12.10，在未来解压后的**独立交付目录**创建环境，不改当前项目的其他实验环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r stage05_admission_review/requirements-afqmc.lock.txt
.\.venv\Scripts\python.exe -m pip check
```

安装未在本轮新环境中执行；当前环境版本、约束与pip检查已实际通过。报告绘图的matplotlib环境、Word制作工具链与模型环境分离。源码不包含商业中文字体；相关画图函数在缺字体时报错，应合法安装支持字体后再使用。

## 可无AFQMC进行的核验

推荐专用入口在不含AFQMC/Git的临时源码副本中运行49项原有人工合成检查；ZIP成员读取只允许临时合成目录，结果写入本目录，完整日志留在local_only：

```powershell
python stage05_admission_review/run_datafree_tests.py
```

此入口本轮已经实际运行通过。它使用现有解释器，不能代替新环境安装验收。也可分别运行下列三个套件：

```powershell
python -m unittest discover -s stage02_math_protocol/tests -p test_*.py -v
python stage03b_official_validation/test_official_dev_runner.py
python stage03b_official_validation/test_recovery_runner.py
```

这三组使用人工小向量/小语料和临时合成ZIP，不是官方Dev评价。合成ZIP中可以有名为dev.json的人工成员，不能将其误记为官方Dev读取。本轮另在源码副本中运行Stage 3A中13项不需真实Train的检查，合计49项通过。完整`stage03a_internal_experiment/run_tests.py`还包含1项真实Train读取测试，缺数据时不能宣称整套通过。

`test_recovery_runner.py --prove-v1`额外要求Git中存在原冻结提交，普通源码ZIP不含历史，因此不是默认步骤。不要复制`.git`绕过此限制。

## 有外部前提的历史流程

内部fit/tune入口：`stage03a_internal_experiment/run_experiment.py`；默认会写回该目录outputs。正式只读Train-fit预检入口：`stage03b_official_validation/run_official_dev.py`，必须明确指定v2清单；会重新拟合内部fit但不打开官方Dev。两者均要求原始数据和固定Manifest、版本/哈希匹配，**本轮没有运行，也不应作为提交审阅的默认命令**。未来复现应使用隔离副本保存新输出，不能覆盖历史结果。

Stage 3A原README的fit语料和词表输出说明有路径简写，实际保存位置为`outputs/artifacts/fit_corpus.jsonl`和`outputs/artifacts/feature_vocabulary.csv`。

官方Dev运行器的执行开关不在默认命令中提供。它还需要私有事故/授权/发布门禁、精确历史Git状态和不可重复的attempt记录。保存该源码供审核并不构成新评价授权；不能删门禁、复制旧授权、篡改attempt或换阈值来获得“可运行”标志。

历史`qa_review/run_qa_diagnostics.py`会执行char(2,3)探索并读取人工笔记，不作本次正式交付的默认入口。正式主方案仍为char2 TF-IDF、对照char2 Count和两个常量基线。

## 保存结果与图表

公开`stage03b_official_validation/results/dev_metrics.json`、分布摘要和聚合分箱可以核对汇总指标与图形输入。`recheck_saved_results.py`及旧绘图入口还需要本地逐行预测、记录和执行指纹，不能在没有这些文件的公开包中声称可以逐行复算。该复算脚本还以排他方式写`independent_recheck.json`，历史输出已存在时会拒绝覆盖；未来应在副本中操作。

Stage 4A报告生成器还依赖未公开的Stage 1审计、Stage 3A汇总与QA JSON。最终个人Word/PDF是独立报告交付件，不在源码公开包内自动重建；教师原模板和工具需合法另取。

## 不可缺少的说明

程序中fit语料去重只用于词表/IDF；评价不去重。标签不是输入特征，余弦不是概率，阈值各自在tune选择后冻结。高误报、弱排序及F1接近恒预测1的结论不得因打包而改变。AI辅助编程、调试和审查的实际用途应随交付保留；未完成跨机复现不能写成完成。
