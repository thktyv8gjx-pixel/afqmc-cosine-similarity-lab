# AFQMC 课程源码交付入口

题目：基于余弦相似度的中文客服相似问句识别。课程：人工智能数学实验。
源码与合成样例可在无 AFQMC 数据时验证；完整课程交付另需私有正式报告和教师模板。
本目录是 2026-10-26 新增的工程层；Stage 4A 是历史初稿，不能生成正式报告。

## 安装（原生 CPython 3.12.10）

macOS 14+ / Apple Silicon，先用 `uname -m` 确认为 `arm64`：

```sh
python3.12 -m venv .venv
.venv/bin/python -c 'import platform; print(platform.machine(), platform.python_version())'
.venv/bin/python -m pip install --only-binary=:all: --require-hashes -r delivery/requirements-cp312-macos-arm64.lock.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/verify_course_delivery.py
```

Windows x64 PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -c "import platform; print(platform.machine(), platform.python_version())"
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: --require-hashes -r stage05_p1_closure_review/requirements-cp312-win_amd64.lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/verify_course_delivery.py
```

输出必须为 Python **3.12.10**，Mac `arm64` / Windows `AMD64`。`py -3.12` 或
`python3.12` 只是选择器，版本不符请安装准确补丁版本后重建新环境；不要修改版本门禁。
无需激活虚拟环境。数值库与传递依赖共九包固定，Windows wheel SHA 不用于 Mac。
Mac 锁来自实际下载的 ARM64 wheels，其中 NumPy/SciPy 要求 macOS 14+。
包中不携带 wheels 或字体；首次安装需要包索引网络，不能宣称离线安装已齐备。

## A/B/C/D 四类操作

下文 `python` 指上述虚拟环境解释器。任何工作目录均可用脚本绝对路径调用。

|类型|命令|范围与前提|
|---|---|---|
|A 默认|`python scripts/verify_course_delivery.py`|49 项冻结无数据测试、37 项准备测试、新增交付回归、自有余弦演示、pip 检查及冻结结果核验；不下载/训练|
|B 准备|`python scripts/verify_course_delivery.py --mode prepare`|固定官方下载地址、ZIP SHA、3 成员 SHA/CRC、字段/行数及 42,511 行 Manifest；再离线复查幂等性；不拟合或评价模型|
|B 离线|`python scripts/verify_course_delivery.py --mode prepare --offline`|仅使用已存在归档，完整核验并重建候选 Manifest；不联网、不覆盖|
|C Train|`python scripts/verify_course_delivery.py --mode train --output local_delivery/train-001`|需先完成 B；在新的私有源码快照执行 Train 内部 fit/tune；只允许读取 train.json；绝不写回历史输出|
|D 审阅|`python scripts/verify_course_delivery.py --mode frozen`|检查 34 个冻结源码及配置、附加正式结果原字节 SHA，独立按混淆计数复算 F1/FPR；不是逐行预测复算或 Dev 重放|

每次创建新的 `course_validation/<时间-随机后缀>/summary.json` 与本地日志，记录环境、
各组退出码和指纹。也可传入不存在的 `--output`。已有输出包括空目录均拒绝复用；失败材料保留。
C 的 `--output` 必须在当前根目录 `local_delivery/` 下，不能上传这些输出。
禁止绕过包装层直接用旧 Train 入口写入已有实验目录。

C 绘图使用系统已有中文字体；必要时指定 `--font <合法安装的字体路径>`。
适配器仅为原绘图器提供字体路径，不修改冻结文件、数值计算或图表数据。
Mac 的新内部 tune 阈值可能有末位浮点差异；新结果单独保存，不替换正式阈值。
正式 TF-IDF 阈值始终为 `0.05574242073415668`，Count 为 `0.10660035817780521`。

B 读取全部拆分只用于结构审计和完整 Manifest 重建；这与 Dev 模型评价不同。
缺素材或错版本时返回非零码并列明路径/预期 SHA。官方下载失败可按
[来源说明](../DATASET_PROVENANCE.md) 原样取得归档，再使用 `--offline`；不能接受替代版本。
完整归档 SHA 为 `5a4cb1556b833010c329fa2ad2207d9e98fc94071b7e474015e9dd7c385db4dc`，
Manifest SHA 为 `e856f4b9fd42be0ecfe530c002048933e31d971a3ba8d070a0c13daed241d907`。
官方 Dev 运行器保留原授权、事故和 Git 发布门禁，本入口没有 Dev 开关，不复用历史授权。

## 从当前提交制作和验证 ZIP

```sh
python scripts/build_course_package.py
python scripts/build_course_package.py --verify dist/afqmc-course-source.zip --extract local_delivery/unpacked-001
```

仅允许干净、已提交的当前 HEAD；读取新 `package_allowlist.json` 的显式文件列表，
逐文件比对 Git blob 原字节及全部冻结校验。不能直接压缩整个工作区或沿用历史 64 文件清单。
ZIP 带 `PACKAGE_MANIFEST.json`，绑定提交、全部成员大小和 SHA；不含 Git 历史、环境、
缓存、个人报告、教师模板、原始数据或逐行输出。固定顺序/头信息与不压缩格式使两端可
生成相同字节。验证器先检查全包 SHA/CRC、路径穿越/大小写碰撞，再写到不存在的目录。
外部 ZIP SHA 是独立分发指纹，包内 Manifest 不能代替它，也不构成数字签名。

在解压目录**另建新的 venv**，按上面 A 安装和执行。`build_course_package.py` 的构建功能
需要 Git checkout；包内验证/解压和 A/B/C/D 不依赖 Git。源码 ZIP 不能重新生成缺失的个人报告。

## 原生 CI 与安全范围

普通 Push/PR CI 检查 `macos-15` ARM64 与 `windows-2025` x64 的实际运行架构，使用
Python 3.12.10。每端安装独立环境、运行 A/D、构建当前 HEAD 包，再在含中文和空格的新路径
解压并安装第二套环境重跑。平台支持以真实 Actions 结果为准。
手动 `workflow_dispatch` 的 `prepare_real_data=true` 才允许 B；普通 CI 不下载 AFQMC。
上传白名单仅含 `course_validation/ci/*summary.json` 的明确五个文件，不上传整个目录或数据。

运行器标签依据 [GitHub 官方运行器文档](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)；
仍在进程里断言平台，标签不当作测试结果。

## AI、来源和提交边界

Codex 辅助了工程适配、测试、CI、源码包及审计文字；现有冻结数学实现与历史结果保留。
新增自有小样例明确为人工构造，不能算 AFQMC 性能。本人应阅读理解公式、代码和结果，
AI/CI 通过不等于教师评分、本人已审阅或正式课程提交。
AFQMC 来源 CLUE/蚂蚁金服；获取与再分发限制见 DATASET_PROVENANCE。
教师是否接受“源码 ZIP + 官方获取说明”仍需确认；未经确认不将原始问句放入公开包。
完整 Word/PDF、模板、正式正文源和生成器需从原 Windows 私有工作区转移后另验。
