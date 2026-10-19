# AFQMC源码候选交付运行说明

本说明对应58项基础清单加本轮无数据验证入口、合成素材和完整平台锁的新版候选；目录结构和文件SHA见`candidate_inventory.v2.json`。本轮仅准备独立暂存副本并实跑，不制作最终ZIP，不提交学习通。不要整体压缩项目根目录或使用其他数学实验的根requirements文件。

## 环境安装

在未来解压后的独立交付目录使用Python 3.12.10/Windows AMD64：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: --require-hashes -r stage05_p1_closure_review/requirements-cp312-win_amd64.lock.txt
.\.venv\Scripts\python.exe -m pip check
```

9包锁含实际下载安装的wheel SHA；不包含wheels，仍需网络或经单独核验的本地依赖来源。其他系统不得套用此平台哈希锁。详细实际安装、传递依赖和核验范围见02号记录。

## 默认无AFQMC验证

```powershell
.\.venv\Scripts\python.exe stage05_p1_closure_review/run_synthetic_validation.py --root . --output local_validation/synthetic_tests.json
.\.venv\Scripts\python.exe stage05_p1_closure_review/demo_cosine.py --root . --output local_validation/demo.json
.\.venv\Scripts\python.exe stage05_p1_closure_review/check_external_inputs.py --root .
```

前两条应成功，第三条在不含原始材料的交付副本中应返回2并说明MISSING_EXTERNAL_PREREQUISITES，不能把预期缺失误写成已经复现正式结果。合成测试49项包含临时人工ZIP，成员可名为dev.json，但不是官方Dev。

## 原程序与外部前提

- 内部Train fit/tune实验、完整Train加载测试和内部输出验证，需要正确原始归档及完整固定Manifest；没有材料时明确停止。本轮不执行这些数据实验。
- 官方运行器保留原门禁，帮助信息可查看；不提供Dev执行开关，不把旧attempt或旧授权当作新复现权限。
- 保存的逐行结果复核/旧画图入口还需要未随包公开的预测、记录和执行指纹；公共汇总只能核验汇总关系，不能恢复逐行分类。
- 冻结发布构建器需要历史Git状态或其他历史输出，仅为审阅源码，不是默认运行入口。2项旧缺陷证明需原提交历史，普通源码包不附`.git`。
- Stage 4A旧报告生成入口已停用；报告正式正文由本轮登记的私有canonical源控制，不属于公开模型源码包。Word/PDF需分别按课程受限方式提交，日期和本人确认尚未完成。

不得通过修改配置、阈值、标签、输入哈希或保护门禁来让缺材料的程序显示成功。数据准备与教师是否接受不随包附数据，见03号记录，当前尚未解决正式结果复现的全部前提。
