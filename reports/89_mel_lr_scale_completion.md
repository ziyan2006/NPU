# 89 — LF32学习率幅度对照完成与4500独立复核

2026-10-04，NONRELEASE。500共同更新4001..4500已完整结束；半学习率没有解决整体取舍，两臂仍eligible=false/release_selection=NONE。基线4000是旧起点，仅4250/4500两个新增训练阶段。

## 实际结束与复核

2026-10-03T21:50:48.6348548Z读取实际run_status：complete/4500/additional500/error=null，completion.json和最终PT/receipt/score齐全。实际launch detached_launch_20261003_193445_5324751.json与同标识detached_exit相符；退出回执在2026-10-03T21:41:07.5629354Z记录真实exit_code=0、launcher_process_id51952及正确launch路径。实际worker72684/venv shim51952已消失，21:55:52Z也无仓库Python训练/诊断/复核进程，stderr0。未把旧running或历史PID当活性，不抬高4500硬限。

194 final verify：exec chunk cce79c，真实native exit0，日志final_verify.log保留；完整500预算/游标/每臂暴露及输出哈希通过。196 summary4500：前台session35194，完成chunk d8ec8b，真实native exit0、内置verify_output通过，0模型前向/无优化器。原31首177视图、逐曲原政策、完整model/Adam/模式/参数顺序/更新暴露/源ARMS[1]及停止证据、本臂kill32与实际LR1/.5独立核对。只新生成4500，不重复4000/4250。

196 listen4500：前台session32117，完成chunk94ce0d，真实native exit0及内置verify通过。固定NightOwl/abjones_2/Beneath各11变体共33WAV一次生成于results/mel_lr_scale_review_20261004/listen_step_4500。原输入PCM、每源共同音量及播放增益1.0，不按分数换歌；没有重导4000基线、4250或任何旧音频。人工听审仍PENDING，不是已自动听过。

| 新证据 | SHA256 |
| --- | --- |
| 最终PT | b3a0450ecd9bba259981bd47f8a49c0fee1a997d7a087a72570f589cc14ed5b3 |
| completion.json | def6529b76af98d6b2e037b29c65f7a6c7b31010b3a9e230fc101d21a1eeead7 |
| 4500 receipt | 829f65b6d101c5b298f374ea1864dbae850f9e1323c77920a4ad1eff037a60e6 |
| 4500 score | 06cec2f938b53c854c3e971e7dcef636c3a47e5103046cfa2cd96442c0b046c6 |
| 匹配actual exit | ae305542f4c5e7287ddde867b3ccafde8b7907b685ac694c645777233c47ded3 |
| summary review | 27c9224538244cb07fdb8a7ee4b784921cf38702a6877a43855a6917506d41a6 |
| listen review | 0a7ccdd9e7eca1900802d8bd3ea63b990440dfc3eb9f14062695022716952755 |
| listen plan | 9ff050d2eefd5e975e48c2c13ab91d8677a4397b51adaebf5039af9d8fd15e2d |

独立monitor写stage_4500_analysis/review、completion_review与逐域逐指标/逐曲器乐historical_best，analyzed=[4000,4250,4500]/listening=[4250,4500]/pending=[]。这些是记录，不是loss或单均值排名，不修改本轮封存代码/协议/审批/训练产物。

## 同边界的新终点比较

源仍完整4000 LF32候选ARMS[1]；两臂固定kill32、完整辅助.2、器乐权重4/六槽分母6、同Mel真值/原采样增益/FP32日程与图。唯一新变量是实际Adam LR原函数乘1/.5，不冒充半LR数值resume。原stop4000/legacy[3750,4000]与新legacy[4250,4500]都保留。

残留为原逐曲绝对投影系数均值比的20log10（负为少），重建为人声error-SNR差（正为高），不是能量百分比：

| 原DEV域 | 半LR对自身4000：残留dB | 重建dB | 半LR对并行LR1：残留dB | 重建dB |
| --- | ---: | ---: | ---: | ---: |
| MUSDB native（19首） | -0.139545 | +0.067868 | -0.839160 | +0.193257 |
| MUSDB weak（19首） | -0.533046 | -0.372588 | -0.142924 | -0.320738 |
| MIR native（9首） | +1.495582 | +0.158723 | -1.418394 | +0.183651 |
| MIR weak（9首） | +0.453847 | +0.474390 | -0.440977 | -0.528863 |

半LR对自身平均残留仍多0.319210dB、重建高0.082098dB；对并行原LR四域残留都少（平均-0.710364），重建平均-0.118173，两弱域更低。对照对自身平均残留多1.029573/重建高0.200271。不能选整体赢家，也不能说没有学习。

对冻结44，两臂MUSDB弱残留现在较少（LR1 -0.052912/半LR -0.195836dB），不是仍然“两弱域残留都增加”。但MIR弱仍增加LR1 +0.842541/半LR +0.401564，为原政策失败原因；半LR对冻结MUSDB弱重建低0.065075dB。旧DEV探索不是新盲验收，平均改善不覆盖失败域。

## 伴奏与逐曲器乐

半LR对并行LR1的伴奏系数绝对误差四域更大，不能把有符号CA增益当绝对误差。低频伴奏sidecar沿旧支持、按arm分别记录，只描述不新增资格门槛：

| 域 | 半LR减LR1：CA绝对误差 | LF伴奏error-SNR dB |
| --- | ---: | ---: |
| MUSDB native | +0.013696 | -0.334064 |
| MUSDB weak | +0.009690 | -0.862476 |
| MIR native | +0.030246 | +0.048364 |
| MIR weak | +0.023738 | -0.644281 |

半LR对自身CA绝对误差在两MUSDB变差、两MIR改善；低频sidecar亦两MUSDB退、两MIR改善。不能概括全部保护同向变化。

| 器乐 | 半LR减自身4000：伴奏SNR dB | 半LR减并行LR1 dB |
| --- | ---: | ---: |
| Beneath_v1 | -12.199826 | +2.212996 |
| Maldito_(8-Bit)_v1 | -2.236489 | -2.727178 |
| Wombat_Combat_v1 | -13.183725 | -2.839711 |

半LR对自身三首均退；对照对自身仅Maldito +0.490689改善，另两退。半LR对并行仅Beneath改善，另两退；均值-1.117964dB不能解释为等量绝对误删。分析JSON同时保留每首removed_energy_relative_mix_db差，不省略保护记录。

4250到4500半LR四域残留都减少（平均-0.472553），但重建平均-0.185992，两弱域重建更低，三器乐均退。仍有变化，只有两个新增阶段，不证明三阶段平台期、LR过大、收敛、容量上限或全库/长源/板端因果。

## 后续选择与持续巡检

不机械进一步降低LR/LF、升lambda/器乐权重或删保护。报告90选择新TRAIN-only、零更新的Adam存量方向对齐诊断准备：用新三批固定输入及三组已完成状态，区分存量动量/预条件方向与即时目标梯度；不是重做169/175的原始梯度对诊断，也不是新GPU训练。当前只有新预声明，工具/测试/plan/worker未生成。

人工4500/4250及历史听审和独立真实验收仍PENDING，releaseNONE；图、NPU周期/BRAM/DSP目标、板端镜像和精度不改。原所有历史绑定、失败记录、vitis_journal.py及未提交修改保留。巡检持续20分钟、同聊天ACTIVE，健康无新可行动结果安静，不因本轮结束自行停用。

依据OpenAI Docs的[同聊天定时巡检文档](https://learn.chatgpt.com/docs/automations)同步原heartbeat实际完成证据与下一准备范围；不新建聊天或巡检。
