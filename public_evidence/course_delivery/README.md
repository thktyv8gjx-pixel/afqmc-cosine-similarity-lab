# 2026-10-26 跨平台课程交付验证

此目录保存本轮安全证据。最终状态需以具体提交和 Actions URL 对应的实测为准。

本机 macOS 26.6 / ARM64 / 原生 Python 3.12.10 已安装全部冻结九包并通过 pip check。
原 49 项测试首次因 macOS TMPDIR 优先级触发 11 个合成 ZIP 门禁失败；非冻结包装层
设置 TMPDIR/TMP/TEMP 为同一隔离路径后，49 项及自有余弦演示通过。原 37 项数据准备测试通过。
完整原始失败与修复日志保留本地，不公开主机路径。

官方固定 URL 实际下载、归档与 3 成员 SHA/CRC/行数检查通过；完整 Manifest 为 42,511 行，
SHA-256 `e856f4b9fd42be0ecfe530c002048933e31d971a3ba8d070a0c13daed241d907`。
再次 offline 重建候选通过，已发布归档和 Manifest 字节及 mtime 均未改变。

显式私有 Train 快照实跑：27,467 fit / 6,867 tune，54,330 唯一问句、35,242 维，
语料和词表内容指纹匹配。仅打开 train.json，未运行 Dev 评价；四幅内部 tune 图生成。
新 TF-IDF tune 阈值为 `0.055742420734156695`，与固定正式值的差为约 `1.4e-17`；
Count 阈值完全相同。记录这一平台浮点现象，不覆盖正式阈值或改历史结果。
两方法内部 tune 混淆计数和 F1 与既有汇总一致。

新增检查按原字节核对 34 冻结文件，加上正式结果/图表等共 60 个基线文件。
正式 TF-IDF Dev 的 TP/FP/FN/TN=1186/2425/152/553，F1=0.47928874520105075，
FPR=0.8143049026192075。保存指标的算术核验不冒充逐行预测复算或模型重放。

原生 CI 已实跑通过：macOS 15.7.9 ARM64 和 Windows Server 2025 AMD64，均为 Python 3.12.10。
- [普通 Push CI](https://github.com/thktyv8gjx-pixel/afqmc-cosine-similarity-lab/actions/runs/40250018868)：两个平台均完成 49+37+26、demo、pip check、构建及新环境解压验证。
- [手动官方准备 CI](https://github.com/thktyv8gjx-pixel/afqmc-cosine-similarity-lab/actions/runs/40250103987)：两个平台在解压副本准备官方数据，3 成员 SHA/CRC、42,511 行 Manifest 和 offline 字节/mtime 幂等均通过。

以上安全快照绑定 `a69d1e6`，详见 `ci_snapshot.json`。该版本本机/两端 CI 的 ZIP 均为
1,248,137 字节、SHA `0facfccd6a1554488f46d4f5506c7a33cdbdb9de882cf59ee21003eced0cee72`。
后续仅补齐本页、五维映射与脱敏证据；最终包必须按新 HEAD 重建并再次验证，最终 SHA 在包外记录。

独立 fresh Codex reviewer 审查全部新增实现，另对照基线 Git blob 和实际解压运行，
未发现 P0/P1/P2 代码问题；这不是人类审核或教师评分。五维映射见
[TEACHER_EVIDENCE.md](../../delivery/TEACHER_EVIDENCE.md)，最终交付阻断见 `defects.json`。

正式私有 Word/PDF、教师模板、canonical 正文和生成器在当前 Mac 搜索范围内缺失。
历史 15 页、52 原生公式仅为已登记的旧证据，不能宣称本机已取得或逐页验收。
最终课程交付当前为 `CROSS_PLATFORM_COURSE_DELIVERY_BLOCKED`。

使用方法见 [课程入口说明](../../delivery/README.md)。
