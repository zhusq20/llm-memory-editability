# 完整预测的汇总恢复：零追加写入

source-v2正式运行已保存32/32份原始逐题JSON及32/32份目标/A状态。写入和读出全部完成后，复用的TTCD汇总函数要求不存在的`teacher`字段而退出；不是模型、目标拟合或参数求解失败。原运行的所有文件、失败及W&B保留。目标拟合共367次latent更新，32事实各四次down求解；其中最早两事实的目标和状态已在此前尝试完成，恢复仅重算核对它们。

source-v3只加入不要求TTCD教师字段的汇总模块及完整记录恢复分支，不改变评分、条件交集、bootstrap、样本、目标拟合、层、阈值、L2或计算模型。修正`companion_A4_correct`的显示名称为`companion_edited_correct`。新增回归覆盖无teacher记录的汇总。AlphaEdit写入模块及所有32份A字节不变，所有原始JSON/状态SHA256绑定到新执行配置。

恢复阶段加载原300k基础并校验身份，逐份核对32原始记录、目标拟合元数据及A摘要，从已完成预测生成canonical evaluation/result.json，实际新增latent/down写入均0。原367次latent与128次独立事实down提交记在`scientific_latent_updates_including_prior_attempts`和`unique_formal_down_matrix_updates`，避免在新恢复W&B运行中冒充重复写入。原正式进程在汇总处退出，未执行其终点内存θ摘要；恢复及随后独立重载分别核对新进程θ/绑定输入前后哈希，不能改写原失败阶段为已完成终点检查。

随后独立新容器对32个已存目标重算全部down，与A逐值相等，并直接载入A逐字段重放原32份预测；没有重新优化目标。最后独立核对开发、恢复正式及重放W&B的finished状态、实际更新计数和完整canonical JSON下载哈希。

恢复入口`resume_alphaedit_discoloop_v5.py`，新输出为私有卷`/out/attempt-5`及持久私有`output/attempt-5`；原attempt-3/4、source/source-v2保留。开发仅复制原已选结果、保护矩阵和追踪收据，不重做开发。主机权限不改变，仍不重启其它训练。
