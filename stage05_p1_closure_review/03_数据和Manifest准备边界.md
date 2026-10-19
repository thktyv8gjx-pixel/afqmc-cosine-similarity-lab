# 数据和Manifest准备边界

## 两种可交付范围

**自有合成材料验证已经可运行。** 候选包可以包含本轮构造的两句语料和三个例子，来源标为author_constructed_for_delivery_validation；不包含从AFQMC抽取或匿名改写的问句。程序调用原冻结实现函数完成Count/TF-IDF余弦，并对照独立数学期望。它仅验证算法和素材读写，不作为正式数据性能或正式拟合空间的复现。

**正式AFQMC结果复现仍有外部前提。** 原归档、Manifest、真实问句和逐行结果不随当前源码候选包交付。公开可下载、仓库Public、教师认可提交形式，都不能替代适用的数据许可。既有官方来源审查仍未确认原始数据再分发权；教师是否接受不附原始数据仍待本人确认，本轮不代答。

## 官方获取和放置步骤

1. 使用者自行核对[CLUE官方任务说明](https://github.com/CLUEbenchmark/CLUE)及[AFQMC来源页面](https://tianchi.aliyun.com/dataset/106411)中的适用条款。
2. 从CLUE所列官方入口`https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip`自行取得归档；本轮未重新下载或打开该原始归档。
3. 将归档放在交付根目录相对路径`stage01_data_audit/raw/afqmc_public.zip`，在使用前核对原始文件SHA-256：`5a4cb1556b833010c329fa2ad2207d9e98fc94071b7e474015e9dd7c385db4dc`。
4. 另行合法取得既定完整Manifest，放在`stage01_data_audit/split_manifest.csv`，SHA-256必须为`e856f4b9fd42be0ecfe530c002048933e31d971a3ba8d070a0c13daed241d907`。成员指纹及其他版本字段以v2清单为权威。
5. 运行只检查是否存在的`check_external_inputs.py`；存在不等于哈希、来源或许可已核验。该工具没有自动下载、重建、fit或Dev执行功能。

## Manifest的实际可行性

原项目本地保留有既定Manifest；本轮只复核其文件SHA，不复制到暂存副本，也不公开其行内容。已有文件可用于今后获准的本地审查，这不证明任何获取者都能合法取得同一文件。

历史`stage01_data_audit/audit_afqmc.py`会解析Train、Dev、Test并生成覆盖全体来源行的Manifest，其完整来源覆盖和pair_group_id检查依赖相应文本。它不是“只下载Train就无条件重建完整Manifest”的入口。本轮未运行该脚本，未重新读取Dev用于重建；在当前数据访问边界下，无法现场证明一个新使用者从官方下载后立即取得字节完全相同的Manifest。

因此：**合法取得或在另行允许的数据审计范围内重建并核对原Manifest，是正式结果复现的阻断前提。** 不能临时改用Train子集Manifest、补造group字段、改哈希或关闭检查来绕过。即使全部材料就绪，官方Dev模型重放仍须独立明确授权，当前不提供执行操作。

## 不纳入候选包

原始归档、逐行问句/标签/预测和真实案例、教师模板、课堂PPT、往届PDF、姓名学号等个人材料、密钥、虚拟环境、日志中的本机路径、wheels与商业字体均排除。个人Word/PDF仍为本地独立报告文件，日期和学院抬头继续待确认。源码候选清单明确记录每个安全文件和SHA，但本轮不创建冒充最终提交的ZIP。
