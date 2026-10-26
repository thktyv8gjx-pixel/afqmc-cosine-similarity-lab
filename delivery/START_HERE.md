# 从这里开始

把私有提交包**完整解压**到可写目录，再打开根目录中的启动文件：

- Apple Silicon Mac（ARM64，macOS 14+）：双击 `Run_Mac.command`。
- Windows x64：双击 `Run_Windows.cmd`（系统自带的 64 位 PowerShell）。

第一次需要联网：启动器从官方 GitHub 下载固定 uv 0.13.5，先核对固定 SHA-256，再配置隔离 CPython 3.12.10、pip 和九个精确哈希锁定的项目依赖。无需先装 Python/uv，无需管理员权限。所有运行时和缓存存于解压目录的 `.course_runtime`；不修改 PATH、注册表、用户配置或已有 `.venv`。首次可能需要数分钟，请等待窗口中的阶段提示。

首次成功后，在**同一解压路径**可完全离线再次运行。将整个目录移动到别处或重新解压后，可能需要重新联网配置私有环境。每次启动都会重新核对依赖锁文件、冻结科学文件和实际环境版本/原生导入；成功探测到完整环境时跳过下载与安装。每次执行 uv 前，都重新校验缓存 archive 并从中提取可执行文件。

启动后自动完成：

1. 核对原始 AFQMC ZIP SHA-256、三个成员、全部 42,511 条记录，并重建/核对完整 Manifest。包内已有数据时强制使用离线数据准备；保留原始文件及既有 Manifest。
2. 执行 A：49 项原有合成测试、数据准备/交付回归、自编例子的实际余弦演示，以及冻结文件与保存指标的算术审阅。
3. 将摘要、演示分数和日志写入一个新的 `course_validation/<时间-随机编号>/` 目录。窗口显示结果目录和实际分数；按回车后关闭。

打开新目录的 `READ_ME.txt` 可读到中文摘要；`summary.json` 提供结构化结果，`synthetic/summary.json` 和日志提供细节。`.course_runtime/last_run.json` 是最近一次验证的摘要指针，使用项目相对路径，不包含绝对用户目录。旧结果目录不会被覆盖。

演示分数是**本次实际计算的自编合成例子余弦值**。摘要中正式 AFQMC Dev 指标来自**已保存的冻结结果**，本次只校验文件指纹与混淆计数算术，**没有重跑官方 Dev，也没有自动执行 Train**。

## 离线与自动执行

从终端运行时可追加两个参数；两个系统含义相同：

```text
Run_Mac.command --offline --no-pause
Run_Windows.cmd --offline --no-pause
```

Mac 终端在项目目录中使用 `./Run_Mac.command --offline --no-pause`。`--offline` 禁止下载；`--no-pause` 完成后立即返回退出码，适合 CI。不带 `--no-pause` 时成功或失败都会保留交互窗口。`--help` 显示参数，不创建环境、不联网。

私有包附带原始 ZIP 和 Manifest。公开源码包不含这两个私有文件；首次联网启动会通过既有 `prepare_afqmc.py` 从登记的官方地址取得数据并验证。离线缺原始 ZIP 时明确失败，不使用替代数据。

## 出错时

成功退出码为 `0`，任何失败均为非零。启动器保留已有数据、缓存和历史结果，不自动清理坏缓存。

| 退出码 | 含义与处理 |
| --- | --- |
| 2 | 参数错误，或 uv 引导 Python 失败；阅读窗口具体提示，联网失败可恢复网络后重试。 |
| 10 | 冷启动离线且缺 uv archive；先在此解压目录成功联网运行一次。 |
| 11 | uv archive SHA-256 错误/无法提取；未执行其内容。保留原件，使用可信包重新解压。 |
| 12 | 目录不可写或出现符号链接/junction；在真实的可写目录重新解压。 |
| 13 | 官方 uv 下载失败；检查网络后重试。 |
| 14 | 系统/架构不支持；需要上述原生平台。 |
| 20 | 离线环境不完整、已移动或已损坏；在当前路径先联网运行一次。 |
| 21–22 | 隔离环境安装/版本/原生导入失败；查看提示的 `.course_runtime/setup-*.log`，已有环境保留。 |
| 30 | 原始数据或完整 Manifest 验证失败；检查新摘要，恢复登记版本的数据，不覆盖历史文件。 |
| 31 | 合成/交付回归失败；查看新结果目录的日志。 |
| 32 | 冻结文件/锁文件不符或其他验证错误；恢复可信的完整提交包。 |

macOS 若阻止首次打开下载文件，在确认文件来自可信提交包后，可右键启动文件选择“打开”，或在终端运行 `/bin/bash /实际路径/Run_Mac.command`。不要关闭系统安全检查。若双击文件被文本编辑器打开，也可使用终端方式。

## 可选 C：单独内部 Train

C 仍是显式命令行操作，见 `delivery/README.md` 和 `scripts/verify_course_delivery.py --help`。使用启动器建立的 `environment.json` 所记录的私有环境 Python，调用 `scripts/verify_course_delivery.py --mode train --output local_delivery/<新的目录>`；依需要提供合法安装的中文字体 `--font`。该步骤不是默认双击流程；不得复用既有输出目录。官方 Dev 不作为任何可选启动模式提供。

Bootstrap 参考：[uv 安装 Python](https://docs.astral.sh/uv/guides/install-python/)、[uv 环境变量](https://docs.astral.sh/uv/reference/environment/)。
