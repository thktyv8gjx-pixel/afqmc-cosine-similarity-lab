# Word与PDF同源及文件元数据审计

## 同源链

上一版Word/PDF全部保留。修订版以原4B正文为内容基础，使用原教师模板构造可编辑Word；最终PDF唯一内容来源为该Word，由Windows Microsoft Word 16.0的ExportAsFixedFormat只读导出。

`local_only/word_export_final.json`记录实际开始/结束UTC时间、页数、版本、导出前后Word SHA-256和PDF SHA-256；UTC换算北京时间为+8小时。导出未保存重写Word，源哈希前后完全一致。最终大小与新旧SHA-256详见`artifact_hashes.json`。本地有一次仅修正app元数据的中间版本，保存在`local_only/before_app_metadata_fix/`，没有更改正文。

全部15页可检索，正文段落逐段在PDF中匹配；六幅嵌入图哈希匹配，PDF也只有六个图像对象，正文与公式未整页栅格化。两个独立150dpi栅格化入口得到每页相同像素，证据在`document_gate.json`。未使用PDF涂白或单独重写封面。

## 文件大小调查

实际旧PDF为950052字节、14页、每页有可提取文本、共六幅图；当前项目未找到15.56 MB文件，不能据聊天描述认定某文件曾被栅格化或被涂白。大幅膨胀可能来自整页图片、图片重采样或字体重复，但没有该文件不能确认根因。本轮最终PDF约0.97 MB，增加一页与示例表，未呈现该异常膨胀。

## 元数据及模板保护例外

原模板SHA-256保持`9977abdebacbf1aad2d46559b35bd28d411758d9834245da85b9aa38de551b8f`。

原保护规则的明确例外：报告副本docProps/core.xml更正当前题目、用户确认作者、最后修改者及实际创建/修改时间；描述记录AI辅助。docProps/app.xml移除旧模板的字数、行数、编辑时长等缓存，页数更新为实际Word导出的15。原模板本身不变。其余18个不透明部件（样式、主题、页眉页脚、编号、脚注尾注及customXml等）逐字节保持，页面几何相同。仅正文、图像关系、内容类型及上述两项属性按合同允许修改。

## 上传限制

已检索超星官方帮助与[平台学生作业说明](https://mooc1-1.chaoxing.com/mooc-ans/nodedetailcontroller/visitnodedetail?courseId=200478574&knowledgeId=956983124)，确认支持文档附件，但未取得本课程具体附件上限。云盘容量、资料上传限制不能代替作业入口限制，故不填“通用上限”。实际课内上传界面需本人确认，本轮未进入或操作学习通。
