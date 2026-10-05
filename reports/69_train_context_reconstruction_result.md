# 69 — TRAIN上下文/LF44/重建诊断完成与边界

2026-10-03 21:41（Asia/Shanghai）。固定12批72输入、三个模型36模型批216槽的CPU零更新诊断已完整结束，实际run exit0、无worker。183独立复核实际exit0；182首次verify exit1的校验器字段非对称问题和原181范围失败分别完整保留，不能说182.verify通过。新结论仅是固定TRAIN实现检查，不是音质提升、全库因果或发布验收。

## 完成与独立核验

182 run前台session14585实际exit0，终态21:20:47 complete36/error=null、模型更新0/CUDA=false，worker61448和shim67380已退出。diagnostic SHA `75d70fabe35061871d05a91b233c4aaa20951b45a7cf2b2e2c05e61529d01e1f`。原plan/输入/36行/诊断及所有代码/测试/协议不改，不重复模型前向。

首次182.verify exit1；36/36原文件SHA与清单相符，源问题是seal变异后嵌入行包含content_sha256，旧verify只从文件行去掉此字段。新183没有改原产物或放宽门槛：对文件行和嵌入行各自核对完整封存摘要，再类型敏感地对称比较完整文档；原模型/Adam/参数模式/源停止证据/数据锁/输入等价审计/计数角色/评分区间/FP32运行环境/不可变状态/汇总重聚合门槛保留。

183的18新单元exit0，prepare session36806 exit0，新plan SHA `034e0c358acd4dc30ee8f5b13dcbb3609923262e89d615455d082bfaca900c31`；verify session38834真实exit0，review SHA `39e401f08e72508fb1282beeeef4a7996aa44a0d5151bdcc6662cdb9049289c7`，实际无活跃诊断/复核进程。[完成证据](D:/WORKBUDDY/STEM/npu-stem-repo/results/train_context_reconstruction_recovery_monitor_20261003/completion_review.json)、[聚合证据](D:/WORKBUDDY/STEM/npu-stem-repo/results/train_context_reconstruction_recovery_monitor_20261003/aggregation.json)。183/测试/报告68/其封存plan及绑定不改。

## 上下文与重建

同一352帧谱、同一模型/输入、有效mask bands44..127和frames130..349、共同sample区间33280..89344：三个模型全部216槽中，history128-block256和history128-block16相对整窗的mask与共同波形max误差均为0。仅排除本固定短窗、有效区间、这两条隔离实现路径的差异；不证明整首长源、板端层状态/缓存、任意相位或硬件流式一致。

reset16负对照确实不同，三个模型mask平均绝对差约0.153..0.168，最大差约0.99996..0.99999；共同波形平均绝对差约0.0244..0.02495（归一化PCM）。说明检查能检出丢历史的差异，不表示该负对照在每槽都音质更差或可发布。

126整窗pv与110 product_vocal在216/216槽逐位一致、max误差0；STFT/ISTFT输入往返全样本max误差3.5763e-7。模型权重/全部模式/RNG/grad未变。没有本样本中重建核不一致的证据，但不能消除旧试听长源的边缘/拼接或板端问题。

## LF44真参考支持

原legacy_log合成矩阵的强制全零FFT bins为0..5，bin中心0..215.332Hz；不把44个band说成44个FFT bin。谱能量统计使用原frames98..349、单边内部bin权重2、DC/Nyquist权重1。参考比例由输入决定，不把三个模型重复数当独立样本。

| 唯一输入组 | 槽数 | 强制零bins真参考谱能量比例均值 | 范围 |
| --- | ---: | ---: | ---: |
| MUSDB真值弱-12dB | 3 | 2.7646% | 0.0325%..7.1258% |
| MIR真值弱-12dB | 6 | 30.0382% | 0.2216%..71.0949% |
| Mel伪标签，非最终真值 | 36 | 3.2981% | 0.0000103%..13.1309% |

原12个纯器乐零真参考都skip比例，没有除零。两个弱域本固定批都有覆盖，但n=3/6很小、曲目活动和频谱不同，不能推全库或把弱域差归因增益。MIR弱槽较高的低频参考比例是新的定位线索；不能说LF44造成30%波形人声残留、达到质量上限、所有低频都是真正人声基频，或取消保护必然改善。bin0是加窗DC分量，不等于整段直流偏置；现有统计尚未分离DC/各bin/近静音活动贡献。

## 下一步与资源

不机械延长旧3500训练、升lambda/器乐权重或删伴奏保护。按低频线索先选择更小的TRAIN-only参考低频分解，预声明12批72槽、0模型前向/更新，区分bin0与其余强制零bins、参考活动和均值偏移，不重做四路径上下文/36模型前向/旧全库DEV或试听。新计划见报告70；在知道贡献前不直接修改生产LF44。若未来改保护/图/板端状态，须独立有界协议、真实机制门槛及Z7020算子/周期/BRAM/DSP审视。

当前没有新GPU训练或CPU诊断，巡检继续每20分钟ACTIVE。3500已有33个试听不重复导出，人工听审/独立真实验收PENDING；所有结果NONRELEASE、release_selection=NONE，不部署/替换板端、不改推理精度或封存历史文件。
