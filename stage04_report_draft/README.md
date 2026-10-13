# Stage 4A：报告正文与证据核对

本目录是完整内容初稿，等待本人审阅。正式题目为《基于余弦相似度的中文客服相似问句识别》。没有运行新 AFQMC 实验，没有再次打开原始 Dev，没有制作最终 Word/PDF 或源码 ZIP。

## 文件

- `报告正文初稿.md`：本地完整稿，含从既有记录选取的少量真实句对；已通过 `.gitignore` 排除。
- `报告正文初稿_公开安全版.md`：同一正文，真实案例仅保留机制与局限说明，不公开原文、行 ID 或逐行分数。
- `report_numbers.json`、`数字与引用证据映射.md`：完整浮点数、原始字节摘要、来源字段及计算式。
- `report_body.md.in`、`build_report.py`：报告模板与生成代码。人工手算文本明确标记为合成例子。
- `figures/`：四张中文标注的报告图与生成记录；原 Stage 3B 图没有覆盖。
- `references.json`：已访问的官方资料，访问日期与用途。
- `report_consistency_check.json`：生成时检查；`validate_report.py` 与 `validation_results.json`：独立的报告数据/表格/图形记录一致性检查。
- `Stage04A自检与待确认事项.md`、`Stage03B用户批准与统计口径勘误.md`：内容审核、风险及既有口径更正。
- `local_only/`：选例、绘图依赖、发布临时文件及本地核验资料，不发布。

## 从保存结果生成

需要已保存在本地的 Stage 1 审计、Stage 3A 汇总和 QA JSON；它们不是可公开再分发的原始数据。缺少这些证据时程序报错，不生成臆造结果。

```text
python -m venv <单独的报告环境>
<报告环境的python> -m pip install -r stage04_report_draft/requirements-report.txt
<报告环境的python> stage04_report_draft/build_report.py
<报告环境的python> stage04_report_draft/validate_report.py
```

本机报告绘图依赖仅装入忽略的 `local_only/plot_dependencies`；不改动冻结实验环境。绘图需要 Microsoft YaHei 中文字体。`--local-cases` 额外读取已保存的人工案例 JSON 和本地选择清单，写入被忽略的正文，不从 AFQMC 压缩包重新读取问句。

## 图表与原图检查

原 Stage 3B 四张 PNG 已逐张检查。四方法比较原图只有两个余弦方案，报告图补齐两个常量基线；分布原图没有明确统一纵轴，报告图使用共同24个分箱、四组统一纵轴、组内比例与阈值线；混淆图增加中文标签及数量色标；tune/Dev图增加常量基线。所有比例指标纵轴从0起，避免夸大小幅变化。报告图只使用保存的聚合指标和分箱计数，不重新运行模型。

最终 Word/PDF 字体、分页与公式排版尚未执行。姓名、学号、班级、专业、指导教师、个人核验说明与授权材料范围待本人确认。
