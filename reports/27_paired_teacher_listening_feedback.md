# 配对试听：用户完整试听反馈

日期：2026-10-02（Asia/Shanghai）。承接[报告 26](26_paired_teacher_listening_pack.md)。
本记录保存用户直接给出的人工听感反馈和项目用途许可声明，不是正式学生训练
启动授权或学生模型验收。

## 人工反馈与范围

用户先表示：“稍微听了几首，我感觉mel band平均水平明显更好，分轨比较干净”。
随后在当前配对试听页的上下文中确认：“全部听了一遍。没有问题”。

据此记录为：**当前固定 24 首配对试听包，用户自报完整试听，未发现听感问题**。
沿用上一条明确反馈，Mel-Band 是用户偏好的新教师候选；HTDemucs 仍保留为
同输入对照，不以主观偏好取消已封存的配对实验。

这里的“全部”按当前页面对应的 24 首试听包理解，不扩大为 275 首原曲库。
试听包使用预选片段，不能据此断言整首所有时间点均无问题。
没有播放器逐窗口完成记录，也没有新的逐首、逐教师导出评分；不补造播放历史、
HTDemucs 拒绝项、客观平均分或盲测结论。不要求用户仅为重复证明同一句反馈而重听。

## 已核对的证据

用户提供的 `C:/Users/30519/Downloads/paired_listening_draft (1).json` 与原
bundle 的文件 SHA256、24 个固定 ID 的顺序和预选样本窗口一致。
该草稿保留 4 首 Mel-Band `acceptable`；其余 20 首未填写。alignment、
HTDemucs 和 rights 字段仍 pending，`training_authorized=false`。
当前 Downloads 未发现更新的同名导出；本次完整试听结论来自上述直接用户消息，
不是把旧草稿中的空项改成已签署批准。

```text
paired_listening_draft (1).json SHA256:
0e530e402d0d82fd81bb113c83e3979f2f8b3f9e8c6527ed5d0b28c83058b491

试听文件/teacher_pairs_20261002/manifest.json SHA256:
8e74e593c8e08270c76f2b72bc3988e596b2cf9a864110634873ab5bbaf0387a

results/paired_distillation_prepare_20261002/paired_bundle.json SHA256:
713e49996cf8645b2ef8a963cdbea032520ffe80a991008bae9cd59cca001d8e

results/paired_distillation_prepare_20261002/listening_pending.json SHA256:
1ac2e5ce93acc9e01fbdd62cc631187b5f3250fe6ab8b41338eb74f3a801f3b9

docs/teacher_distillation_protocol_v1.json SHA256:
3aa6e2fe5fb2f260f00b0397609e72c7018dad2e54513791a004278e30fdc492
```

## 补充：音乐训练使用许可确认

同日，助手询问：“这些 MP3 是否有允许用于本项目模型训练的许可依据？”
用户直接回答：“有”。据此记录为：**用户确认本批 MP3 有用于本项目模型训练
的许可依据**；当前首次配对实验仍只覆盖原固定 24 首，不扩大实验名单。

这是用户提供的用途许可声明，不把“持有 MP3”推定为许可，也不是助手独立的
法律核验。未提供的许可方、文件、条款或有效期限不补造；不据此推及音乐上传、
再分发、商用发布或其他数据/教师权重的使用资格。

本声明独立保存在此记录。原草稿、试听 manifest、entry 和 snapshot 中的
rights pending 是生成时状态，保持不变；后续批准导入须引用本声明并独立封存，
不能自动把原 `training_eligible=false` 改成 true。
本次“有”回答的是音乐许可问题，不是要求立即开始正式训练。

## 下一步及保留的门槛

人工听感反馈已有，不再将本阶段描述为“无人试听”。原封存文件中的 pending/0
是生成时的历史快照，保持不变；后续用途批准应在独立、绑定原文件的记录中引用
本反馈，不通过改写旧 entry、snapshot 或协议解锁训练。

正式训练仍未授权、未启动。人工听感无问题、音乐项目用途许可已有用户确认；
这两项不替代样本级技术对齐证据、其他数据/教师权重使用资格检查，也不解决
HTDemucs 原退出码未知的完成证据选择。
独立真实分轨开发/验收角色、批准导入、script119 实际评估接入以及 CUDA 恢复
实测等剩余项见[报告 25](25_paired_teacher_cpu_mechanics.md)。
教师分轨的良好听感不是板上小模型已提升的证据；仍须公平训练对照及独立验证。

本次仅保存反馈及许可声明，没有修改原标签、试听文件、页面状态、冻结学生、旧正式协议、
原数据锁、新蒸馏协议、板端镜像、SD 或 `vitis_journal.py`。
未启动训练/教师推理/定时任务，没有音频上传、Git 提交或推送。
