# 64 — 伴奏系数项消融500步完成复核

2026-10-03 19:24（Asia/Shanghai）。本轮3001..3500的500共同更新已完整结束，匹配退出回执exit_code=0，worker/shim已退出；178 verify、180 summary3500/listen3500实际退出0。两臂eligible=false，release_selection=NONE，未发布或替换板端。终点移除伴奏项有小幅残留收益，但不是全面胜出。

## 完成证据

- 最终状态complete/step3500/additional500/error=null。对应[真实退出回执](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_component_ablation_launch_20261003/detached_exit_20261003_094454_7514811.json)时间2026-10-03T11:12:29.060095Z；launch_receipt与shim58036匹配。11:15:30Z实际进程检查无worker23356、shim58036或同任务训练进程。
- 178 verify实际exit0，日志SHA256 `94b7b332bdab82a5d823be3f8b0317c779b2510b39f92c1d43d9f0a9524ad9b2`。
- 最终PT SHA256 `62d3e91beeb75ecd06ff5ff204b3c1704819772a7b82458ae2c3b424277bdcfc`。
- 180 summary3500前台session27436真实exit0，review SHA256 `7edd579341e5581b3a29f8f5ab6a34b635f2a1a2cd4b271ac932d8cf77500218`。
- 180 listen3500前台session82675真实exit0，review SHA256 `fbe6f5834fcbf2452914683b2a474c77118624bc968d45791f1b5bed5c297ce0`。33个WAV，一次导出并核验。
- [完整完成复核](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_component_ablation_monitor_20261003/completion_review.json)、[逐域/逐曲分析](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_component_ablation_monitor_20261003/stage_3500_analysis.json)。

哈希、模型/Adam更新暴露、原31首177视图与逐曲聚合、原资格政策、源候选3000与停止证据均通过独立复核。3000是基线，仅3250/3500是两个新评分；旧源stop及legacy停止证据保留，没有提高旧限额或改封存文件。

## 唯一变量与终点权衡

两臂同Mel/真值、六槽基础权重[1,1,4,1,1,1]/固定分母6、cv²系数.2；候选仅将(ca-1)²系数.2变0。兼容键不代表HT教师比较。原图、LF44、输入/日程/FP32精度不变，不用不同训练loss排名。

| 3500候选比较对象 | 人声重建error-SNR变化（高好） | 投影残留变化（低好） | 纯器乐error-SNR均值变化（高好） |
| --- | ---: | ---: | ---: |
| 并行完整辅助控制3500 | -0.0205dB | -0.1776dB | -2.5502dB |
| 分叉起点3000 | +0.2909dB | +0.5496dB | +0.4042dB |
| 原冻结学生 | +0.3412dB | -0.2779dB | +31.4260dB |

平均重建与残留为原政策的四个人声域等域聚合；器乐仅原3首，不能推广全库。候选相对控制四域残留都略少，但伴奏投影系数绝对误差四域都更大，纯器乐3/3首分数都更低。人声重建平均约同，并不消除保护损失。

| 人声域 | 候选相对起点重建dB | 候选相对起点残留dB | 候选相对冻结残留dB | 候选相对控制残留dB |
| --- | ---: | ---: | ---: | ---: |
| MUSDB native | -0.0274 | +0.5142 | -1.1856 | -0.1762 |
| MUSDB weak -12dB | +0.5732 | +0.3847 | -0.0228 | -0.0649 |
| MIR native | -0.0232 | +0.6677 | -0.6246 | -0.2655 |
| MIR weak -12dB | +0.6412 | +0.6318 | +0.7216 | -0.2036 |

终点候选四域残留仍比起点更多；MIR弱人声仍比冻结多残留，是两臂原资格失败原因。弱域残留投影系数约MUSDB0.8172、MIR0.7929，不是残留能量百分比或主观音量。

3250到3500候选平均残留减少0.3919dB，同时重建降低0.0911dB。仍在变化，不能断言训练无效、三阶段平台期、全库因果或LF44表达上限。

## 纯器乐逐曲

| 曲目 | 起点3000 | 控制3500 | 候选3500 | 候选相对起点 |
| --- | ---: | ---: | ---: | ---: |
| Beneath | 91.8233 | 86.9993 | 83.7306 | -8.0928dB |
| Maldito | 20.0148 | 21.9291 | 21.5745 | +1.5597dB |
| Wombat | 58.7275 | 70.5004 | 66.4731 | +7.7456dB |

error-SNR高好。候选相对起点仅2/3改善；平均被Beneath高分影响，不能宣称全曲安全或严重损伤可闻。

## 最新试听与下一阶段

[3500步试听目录](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_component_ablation_review_20261003/listen_step_3500)，固定原NightOwl约6.8秒、MIR abjones_2和Beneath各20秒，每源11变体：原音、真参考、冻结、起点3000、两臂人声/伴奏。同输入同音量，不按分数换歌。候选伴奏：[NightOwl](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_component_ablation_review_20261003/listen_step_3500/source_01_aux_residual_only_backing.wav)、[MIR](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_component_ablation_review_20261003/listen_step_3500/source_02_aux_residual_only_backing.wav)、[器乐](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_component_ablation_review_20261003/listen_step_3500/source_03_aux_residual_only_backing.wav)。

MIR源拼接样本289296、659034仍保留；跳变不直接归因模型。CPU FP32试听不是板端连续流验收。人工听审和独立真实验收PENDING，旧开发集不是新盲测。

本轮不继续删除保护或机械提高lambda/器乐权重，也不延长旧3500硬限。按预声明转入新的有界TRAIN-only、零更新上下文/LF44/重建诊断准备，区分训练/开发短窗与试听长窗实现差异、固定低频掩码抑制和重建边缘，见报告65。当前没有GPU训练worker；巡检保持ACTIVE。

