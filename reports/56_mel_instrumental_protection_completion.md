# 56 — 器乐保护500步完成与独立复核

2026-10-03 16:50（Asia/Shanghai）。结论：运行和结果完整性通过，器乐保护明显改善；整体分离质量仍未通过原开发门槛。全程NONRELEASE，未替换冻结学生或板端模型。

## 运行与证据

固定2501..3000的500共同更新全部完成。相符的`detached_exit_20261003_070144_6350378.json`于16:41:12记录真实exit_code=0，launch_receipt与shim PID关联一致；状态complete/step3000/additional500/error=null，worker57764和shim59340已退出，stderr为空。复核期间短暂出现的172 verify进程是只读检查，不是第二次训练。

172 verify、174 summary3000、174 listen3000均退出0。完成清单所有输出哈希、封存审批/代码身份、模型与评分摘要、Adam完整更新暴露、输入游标、原31首177视图、逐曲聚合和原资格政策通过。最终两臂updates=3000、游标=3000、active stop=3000；model/Adam/参数顺序/模式及Python、NumPy、CPU/CUDA RNG均保存在最终状态。源stop=2500及legacy=[2250,2500]保留，新轮legacy=[2750,3000]，没有重置旧成绩或抬高预算。旧schedule配置只是保留的源元数据，不能用其旧maximum_steps替代新轮硬限3000。

2500仅为分叉基线，2750/3000是两个不同完整新评分；没有pending，不称为三次平台期。两臂均为相同Mel/真值目标，唯一差异是器乐TRAIN基础权重1/4；不是HT教师比较，也不按加权训练loss排列音质。

| 证据 | SHA256 |
| --- | --- |
| 最终全状态 | `c2dbda81188164ffbd1f8ca1ae5ed69b94e83e2ab29e83df9913822f989bb171` |
| completion.json | `2eefb002fcaf5fdb74caca249b5699425ec51dd55a60c6d67031a2171cc9eeec` |
| 相符退出回执 | `eff7c1db9ab7879c77179248eb2b737461a572ec708c1f90944a141cabfef34b` |
| summary3000/review.json | `69bcf16b353be024cafe901cd6d38b792a97ce226d57b9729d57a0f6b86a3c3e` |
| listen3000/review.json | `269c22b48d886de7f006b09a5dffb5746ac7d702e14ea6b5e5f0241e2e05f610` |

完整检查及日志哈希记录在[completion_review.json](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_instrumental_protection_monitor_20261003/completion_review.json)，逐域分析在[stage_3000_analysis.json](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_instrumental_protection_monitor_20261003/stage_3000_analysis.json)。

## 器乐保护：对照三首均改善，但未全部恢复至起点

纯器乐伴奏error-SNR越高，误删越少。不是响度增益或听感提升的直接dB换算。

| 旧开发曲目 | 起点2500 | 权重1控制3000 | 权重4候选3000 | 候选相对控制 | 候选相对起点 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Beneath_v1 | 63.5284 | 63.8460 | 91.8233 | +27.9773 | +28.2950 |
| Maldito_(8-Bit)_v1 | 21.3808 | 19.2467 | 20.0148 | +0.7680 | -1.3661 |
| Wombat_Combat_v1 | 43.8132 | 48.7843 | 58.7275 | +9.9432 | +14.9143 |
| 三首平均 | 42.9075 | 43.9590 | 56.8552 | +12.8962 | +13.9477 |

候选相对并行控制3/3首改善，相对起点只有2/3首改善，Maldito仍退步。平均值受到Beneath近零误删的高分影响，不能推广为全库器乐安全或学生整体胜出。

## 人声：残留略降，重建仍有取舍

下表均为相对原冻结学生的原政策聚合；人声重建越高越好，残留变化越低越好。

| 阶段 | 控制重建增益dB | 候选重建增益dB | 控制残留变化dB | 候选残留变化dB |
| --- | ---: | ---: | ---: | ---: |
| 2500基线 | +0.1157 | +0.1157 | -0.5754 | -0.5754 |
| 2750 | +0.2392 | +0.2502 | -0.7102 | -0.6420 |
| 3000 | +0.0507 | +0.0502 | -0.9686 | -0.8275 |

候选终点相对控制平均残留多0.1411dB，平均重建差-0.0005dB，基本相同。相对2500起点，候选平均残留少0.2520dB，但平均重建低0.0655dB。2750到3000仍在降低残留，同时重建下降约0.2000dB；这不是“毫无学习”，也不是加步就能全面改善的证据。

| 域 | 候选相对控制残留变化dB | 候选相对控制重建变化dB | 候选相对冻结残留变化dB | 候选相对冻结重建变化dB |
| --- | ---: | ---: | ---: | ---: |
| MUSDB native | +0.3149 | -0.0626 | -1.6998 | +0.2901 |
| MUSDB weak -12dB | +0.3998 | +0.3368 | -0.4075 | -0.5091 |
| MIR native | -0.0818 | -0.0344 | -1.2923 | +0.5725 |
| MIR weak -12dB | -0.0685 | -0.2417 | +0.0898 | -0.1525 |

候选MUSDB弱域重建相对冻结下降0.5091dB，超过原允许回退范围；平均重建增益仅0.0502dB，也不足。两臂eligible=false、release_selection=NONE。MIR弱域残留增量回到原容差内，不代表已干净分离：弱域剩余人声投影系数均值仍为MUSDB约0.7818、MIR约0.7373。投影系数不能直接解释为能量百分比或主观音量。

## 3000步同输入、同音量试听

原预选3源，每源11变体，共33个WAV已导出并完成哈希核验；包含原音、真实人声/伴奏、冻结学生、2500起点以及两臂人声/伴奏。未按分数换歌，未覆盖2750试听。人工听审仍为PENDING。

| 片段 | 原音 | 真实伴奏 | 权重1控制伴奏 | 权重4候选伴奏 |
| --- | --- | --- | --- | --- |
| NightOwl，约6.8秒 | [原音](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_instrumental_protection_review_20261003/listen_step_3000/source_01_mix.wav) | [参考](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_instrumental_protection_review_20261003/listen_step_3000/source_01_reference_backing.wav) | [控制](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_instrumental_protection_review_20261003/listen_step_3000/source_01_instrumental_weight1_control_backing.wav) | [候选](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_instrumental_protection_review_20261003/listen_step_3000/source_01_instrumental_weight4_backing.wav) |
| MIR abjones_2，20秒 | [原音](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_instrumental_protection_review_20261003/listen_step_3000/source_02_mix.wav) | [参考](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_instrumental_protection_review_20261003/listen_step_3000/source_02_reference_backing.wav) | [控制](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_instrumental_protection_review_20261003/listen_step_3000/source_02_instrumental_weight1_control_backing.wav) | [候选](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_instrumental_protection_review_20261003/listen_step_3000/source_02_instrumental_weight4_backing.wav) |
| Beneath纯器乐，20秒 | [原音](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_instrumental_protection_review_20261003/listen_step_3000/source_03_mix.wav) | [参考](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_instrumental_protection_review_20261003/listen_step_3000/source_03_reference_backing.wav) | [控制](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_instrumental_protection_review_20261003/listen_step_3000/source_03_instrumental_weight1_control_backing.wav) | [候选](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_instrumental_protection_review_20261003/listen_step_3000/source_03_instrumental_weight4_backing.wav) |

MIR片段在样本289296、659034有原源拼接，不能直接把拼接跳变归因于模型。此处是CPU FP32参考推理，不是板端整数连续流验收。31首旧开发集反复使用，不是新盲测或独立最终真值。

## 下一步判断

器乐保护权重这一变量已产生明确作用，但没有解决弱人声与重建的取舍。不机械提高器乐权重、lambda或延长旧3000硬限。优先做零更新TRAIN-only诊断，分开检查辅助项cv²与(ca-1)²的实际梯度及活动覆盖，并核对固定LF44/推理重建是否限制弱人声表现，再决定新的单变量实验。当前尚不能断言教师标签失效、模型容量上限或仅仅训练不足。

本次仅完成复核及最终试听，不启动新训练、不改变历史绑定。巡检未停用；冻结学生、历史实验、协议/审批/数据锁/标签/缓存、板端镜像及vitis_journal.py保持不变。未上传音乐、写SD、部署或Git提交推送。
