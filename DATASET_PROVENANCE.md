# AFQMC获取流程与版本说明

本仓库提供官方数据的**本地获取与完整性检查代码**，不重新托管AFQMC原始归档、解压文件、逐行问句、Manifest或预测。此前许可审查的未决项保留；本次未重新开展许可调研，也不把第三方数据卡的许可证标注替代原权利人的授权。

## 来源和输入版本

- 数据集：AFQMC（Ant Financial Question Matching Corpus）。数据提供者在CLUE目录中列为蚂蚁金服。
- [CLUE官方任务说明](https://github.com/CLUEbenchmark/CLUE#数据集介绍与下载-introduction-of-datasets)。
- 下载入口：<https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip>。
- 原始ZIP SHA-256：`5a4cb1556b833010c329fa2ad2207d9e98fc94071b7e474015e9dd7c385db4dc`。
- 本地完整Manifest SHA-256：`e856f4b9fd42be0ecfe530c002048933e31d971a3ba8d070a0c13daed241d907`。

指纹仅标识实验版本，不赋予再分发许可。使用者应自行确认适用条款。第三方数据与本项目自有源码的权利归属分别判断；本项目不对AFQMC附加MIT或其他软件许可证，也不声称获得数据权利人的背书。

## 本地准备

```powershell
python scripts/prepare_afqmc.py
```

新入口将数据保存到被忽略的`stage01_data_audit/raw/afqmc_public.zip`，经临时文件下载、归档和成员SHA-256、ZIP CRC、三个拆分的行数和字段检查后才接纳文件。直接读取ZIP，不解压问句文件。

Manifest复用未经修改的Stage1分组、固定种子2026、第一折作tune及CSV写出函数。其输入为全部三份原始拆分；只有Train参与内部fit/tune划分，Dev和Test保持原角色。目标SHA不符即停止，不覆盖现有文件、不改写冻结指纹。此过程读取数据进行结构审计，**不进行模型fit、预测、阈值搜索或Dev性能重评**。

## 文件约定

| 本地文件 | 是否公开托管 | 用途 |
| --- | --- | --- |
| stage01_data_audit/raw/afqmc_public.zip | 否 | 使用者从CLUE获取的固定版本归档 |
| stage01_data_audit/split_manifest.csv | 否 | 本地确定性重建的完整42,511行索引 |
| public_evidence/data_preparation/ | 是 | 不含原文的验证汇总和运行边界 |

下载失败时程序返回2，打印官方URL、预期路径和SHA；可手工原样下载后再用`--offline`校验。程序不会自动接受不同版本、改标签或跳过校验。已有文件哈希不符时保留原件并停止，需使用者检查其来源及备份后自行处理。

详细A/B/C/D运行说明见README；完整课程交付是否获教师接受仍待确认，数据准备通过不等于正式实验重评或最终作业验收通过。
