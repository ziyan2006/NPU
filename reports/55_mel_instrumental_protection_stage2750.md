# 55 — 纯器乐保护对照：首个新增250步阶段

2026-10-03，NONRELEASE。训练仍在原定2501..3000的500共同更新预算内运行，没有更改配方、停止条件或旧绑定。

## 独立复核

2750评分及全状态已完整提交。174 summary2750退出0，复核哈希、模型/Adam暴露、原31首177视图、逐曲聚合、原资格政策及源候选2500/停止证据通过。2500是基线，2750是本轮第一个新增评分，不能称三阶段平台期。

| 证据 | SHA256 |
| --- | --- |
| 2750全状态 | `e2c973536b6039bb76d3cc0379aacd61db5d21108dc7936877761442b586995c` |
| development_step_2750.json | `69759061b61ccb9da7d19d5efeb74882bb54689addd1a4a077db18db5dc7e27a` |
| summary2750/review.json | `cece80316dfeb7f224b78649cd9049b4d6e37aba1a9a69aaa118c2c23634e0c2` |
| listen2750/review.json | `12ff2b970d78723dd9311b69daded141bb75e3878852d36eb8be0c4d1bb705f5` |

两组全部Mel/真值目标相同、辅助λ均.2；唯一差异是器乐TRAIN基础权重1/4、分母固定6。不能用两组加权loss比较音质。

## 器乐保护改善但不均匀

纯器乐伴奏error-SNR越高越好：

| 三首旧开发样本 | 2500起点 | 权重1控制2750 | 权重4候选2750 | 候选相对控制 |
| --- | ---: | ---: | ---: | ---: |
| Beneath_v1 | 63.5284 | 59.2915 | 74.4019 | +15.1103 dB |
| Maldito_(8-Bit)_v1 | 21.3808 | 19.3622 | 18.7928 | -0.5694 dB |
| Wombat_Combat_v1 | 43.8132 | 48.0889 | 53.6152 | +5.5264 dB |
| 逐曲平均 | 42.9075 | 42.2475 | 48.9366 | +6.6891 dB |

候选平均比起点高6.0291dB，但只有2/3首改善。Maldito相对起点还低2.5880dB；不能把平均恢复说成每首都更安全，也不能将三首旧开发结果推广成全库结论。

## 人声仍未解决

候选相对并行控制的平均残留变化为+0.0682dB（略多），平均人声重建变化仅+0.0110dB。相对2500起点，候选平均残留少0.0666dB、重建高0.1345dB，是小且分域不同的变化，并非干净分离。

| 域 | 候选相对控制残留变化（低更好） | 候选相对控制人声重建变化（高更好） | 候选相对冻结残留变化 |
| --- | ---: | ---: | ---: |
| MUSDB native | +0.0600 | -0.0217 | -1.6965 |
| MUSDB weak -12dB | +0.1892 | +0.1304 | -0.4223 |
| MIR native | +0.0733 | -0.0379 | -0.8948 |
| MIR weak -12dB | -0.0498 | -0.0267 | +0.4457 |

候选弱人声剩余投影系数均值MUSDB约0.7805、MIR约0.7681，仍有大量残留。两组eligible=false；候选对冻结的MUSDB弱域重建低0.3797dB，MIR弱域残留多0.4457dB，均触及原开发门槛。平均重建+0.2502dB并不能覆盖这些失败。release_selection=NONE，旧开发不是新盲测。

## 同输入试听与下一步

CPU开发评分结束且worker重新GPU更新后，174 listen2750独立导出并验证33个WAV，退出0。原预选MUSDB NightOwl、MIR abjones_2、器乐Beneath，每源11变体（原音、真参考、冻结、2500起点和两组人声/伴奏），同输入/同音量，不按评分换歌。CPU参考不是整数板端或连续播放验收；MIR原输入拼接事实继续保留。

人声样例（同一输入）：[原音](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_instrumental_protection_review_20261003/listen_step_2750/source_01_mix.wav)、[真伴奏参考](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_instrumental_protection_review_20261003/listen_step_2750/source_01_reference_backing.wav)、[权重1伴奏](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_instrumental_protection_review_20261003/listen_step_2750/source_01_instrumental_weight1_control_backing.wav)、[权重4伴奏](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_instrumental_protection_review_20261003/listen_step_2750/source_01_instrumental_weight4_backing.wav)。不将人工本轮听审自动标完成。

当前状态仍training/error=null，实际worker与本轮命令/创建时间匹配；已观察step2810/additional310、CPU持续增长、stderr空。没有相符正常退出证据，不能声称500步完成。独立monitor已分析[2500,2750]，新增评分数1，并记录各组各指标历史最好；下一完整评分3000。

继续原定预算到3000再核对退出回执、终态/无worker、172 verify和174复核。若器乐恢复但人声回退或仍分域失衡，记录权衡并转查辅助内部目标或LF44表示/重建，不机械增加器乐权重/λ或预算。巡检保持ACTIVE，训练与全部绑定不改，冻结学生/旧镜像不替换，无Git提交推送、音乐上传、SD写入或部署。
