# LF 边界对照完成与终点复核（4000）

记录时间：2026-10-04 00:53（Asia/Shanghai）。所有工作 NONRELEASE。

## 完成事实

固定共同阶段 3501..4000 已完整完成 500 次更新，不延长旧限。实际目录 `results/mel_lf_boundary_20261003` 的 run_status 为 complete / step4000 / additional500 / error=null；终态文件是 `completion.json`。
与实际启动回执 `detached_launch_20261003_150859_6059935.json` 相符的 `detached_exit_20261003_150859_6059935.json` 在 2026-10-03T16:31:11.9698165Z 保存真实 exit_code=0。状态 worker41728 / venv shim64684 已消失；16:53:11Z 无仓库 Python 训练、诊断或复核进程，stderr 空，磁盘约25.08GiB。
启动/训练完成/资格/发布是不同事实。3500 是基线，3750/4000 仅两个新增完整训练评分，不当三阶段平台期。

源是 component-ablation3500 的 aux_full_control / ARMS[0]，不是残留-only ARMS[1]。原 model、Adam、参数顺序、全部模式、更新暴露、RNG、游标及调度迁移已由既有真实机制证据核对。本轮唯一变量训练/重建/DEV/试听 kill44 vs32；同 Mel/真值、基础权重[1,1,4,1,1,1]/分母6、完整辅助.2及原活动门槛不变。源 stop3500 / legacy[3250,3500] 与新 legacy[3750,4000] 均保留，4000硬终止，不重置旧LR/best/stale/patience_anchor。

## 真实独立复核证据

- 186 final verify：exec chunk317e4b，真实 native exit0；日志 `results/mel_lf_boundary_monitor_20261003/final_verify.log`。原审批、绑定、输出哈希、完整预算/游标/两臂暴露及全状态检查通过，无重训。
- 188 summary4000：前台 session87279，完成 chunk0a36fe，真实 native exit0，内置 verify_output 通过。原31首177视图/逐曲聚合/原政策/完整model-Adam-模式-参数顺序/源ARMS[0]与stop/kill身份复核。没有重新生成3500/3750。
- 188 listen4000：前台 session99240，完成 chunk96532d，真实 native exit0，内置 verify_output 通过。CPU-only；原固定 NightOwl / abjones_2 / Beneath，每源13变体，39个WAV，同输入同共同音量，不按成绩换歌，不重复旧试听。
- 原35单元的CUDA夹具失败、审计顺序错误、3750原summary native退出码捕获缺口均保持原记录；当前新核验不回写历史。此次命令 Tee-Object|Out-Null 完整消费后捕获 LASTEXITCODE。

主要哈希：

| 证据 | SHA256 |
| --- | --- |
| 最终PT | 94323f8a30fb5e5decb65d4c110fd5252d0916de68507f42af555b3f539a8f6d |
| completion.json | af40cb8c7591e70e3302a5fe39e918a6f492c9ef92caca0939dc14d9049d9eaf |
| 实际exit回执 | f276c6a4b9c4f539d36ed1e50fd44ad8537a9fa66c4d6b788471372a27c44a35 |
| summary4000 review | af8bd7c3c6cb04b791e389d7a35f6412b2eb77b31e0f1e954c5a8b12342eaefe |
| listen4000 review | 62668572c7f6c15063207f0733e933fb875c021d9d7d7862bd41a9667c8aee9d |
| listen4000 plan | a98e2412dfbe392d91d99ae6ba431c80986f9df582711b3557860ffd88b0d9f7 |

独立完成证据和计算对照写 `results/mel_lf_boundary_monitor_20261003/completion_review.json` 与 `stage_4000_analysis.json`，不写活训练目录或旧绑定。

## 终点结论

两臂相对冻结均 eligible=false / release=NONE。LF32零更新3500曾在旧DEV政策下 eligible=true，但不能说正式500步保持资格或通过盲验收。

平均残留为投影绝对系数的dB变化（负为少），重建为人声error-SNR变化（正为好）；不是能量百分比：

| LF32终点参照 | 平均残留变化dB | 平均重建变化dB |
| --- | ---: | ---: |
| 本臂3500（相同LF32） | -0.111957 | +0.059541 |
| 并行LF44终点 | -1.281793 | +0.193400 |
| 源3500同权重LF44 | -1.489428 | +0.291015 |
| 冻结LF44 | -1.589716 | +0.652748 |

自身起点参照才去掉零更新阈值直接效应：MUSDB native/weak 残留分别多0.069195/0.311986dB，MIR native/weak 残留少0.786502/0.042506dB；MIR weak重建低0.332704dB。候选平均学习收益有限且明显跨域取舍，不称全域更好。

对并行控制，平均优势主要MIR native（残留少4.354689dB/重建高1.469390dB）；MUSDB weak残留多0.023182dB，MIR weak重建低0.586899dB。伴奏系数绝对误差四域全大于控制；低频伴奏error-SNR sidecar的MIR weak低2.858607dB，其他三域分别MUSDB native -0.435511、MUSDB weak -0.266121、MIR native +0.765199dB。sidecar是描述，不新增资格阈值。

对冻结，LF32的MUSDB weak残留多0.337210dB，为原政策失败原因；MIR weak残留少0.052284dB而重建低0.177759dB。不能说已解决弱人声。

逐曲纯器乐（error-SNR变化，正为好）：

| 曲目 | LF32对并行LF44终点dB | LF32对本臂3500dB |
| --- | ---: | ---: |
| Beneath | +0.666435 | -16.425812 |
| Maldito | -1.415309 | +1.214107 |
| Wombat | +5.256842 | -4.145414 |

对控制2/3改善，Maldito退步；对自身仅Maldito改善。均值对控制+1.502656、对自身-6.452373dB受起点Beneath/Wombat极高SNR回落影响，不用单均值或训练loss选赢家。
3750到4000 LF32残留少0.738887dB而重建低0.140379dB，仍变化；两次新增评分不证明平台期、无学习、LF容量上限或全库因果。

## 下一步与试听

最新39WAV：`results/mel_lf_boundary_review_20261003/listen_step_4000`。人工本轮听审、独立真实验收仍PENDING，不能自动视为满足；没有发布、SD、板端镜像或精度变更。

自主选择报告80的 TRAIN-only 交叉边界零更新诊断：同固定三组权重分别按44/32重建，以分开阈值直接效应和相同边界的学习效应。只准备新工具/协议，不再机械下调LF边界、升lambda/器乐权重或删保护。当前没有新GPU训练。持续20分钟轻量巡检，不暂停。
