# 88 — 半学习率探索：4250阶段独立复核

截至2026-10-03T20:59:27.5461624Z，本轮首个新增阶段4250已完整提交并通过196独立复核（追加250共同更新）；整轮尚未完成。半学习率对并行原LR的残留投影四域均较低，但两弱域重建、伴奏系数保护及部分器乐有退步，不能选作整体赢家。两臂相对冻结44均eligible=false，release_selection=NONE。

## 同LF32并行比较与本臂起点

唯一变量仍是实际Adam LR倍率1.0/0.5；两臂LF32、完整辅助、器乐权重4、六槽原输入/采样/增益/FP32及原日程不变。起点为完整4000 LF32候选ARMS[1]，不是LF44控制或3500。4000为同源旧起点，不计新增训练阶段。

以下候选变化为原逐曲聚合；残留使用绝对投影系数均值比的20log10，负数较少，不是能量百分比；重建error-SNR正数较高。

| 原DEV域 | 相对本臂4000：残留dB | 重建dB | 相对并行LR1：残留dB | 重建dB |
| --- | ---: | ---: | ---: | ---: |
| MUSDB native（19首） | +0.410340 | -0.082851 | -0.288692 | +0.112265 |
| MUSDB weak（19首） | -0.038102 | +0.139783 | -0.100550 | -0.129817 |
| MIR native（9首） | +2.038877 | +0.062043 | -0.538328 | +0.062984 |
| MIR weak（9首） | +0.755937 | +0.953385 | -0.346625 | -0.268647 |

候选对自身起点平均残留增加0.791763dB、重建高0.268090dB，仅MUSDB弱域残留略减；对并行原LR平均残留少0.318549dB、重建低0.055804dB，不能用单均值或加权loss选赢家。对照对自身起点平均残留增加1.110312dB、重建高0.323894dB。候选对冻结44虽平均残留少0.797953dB、重建高0.920838dB，但MUSDB/MIR弱残留分别增加0.299107/0.703653dB；对照两弱残留也增加0.399658/1.050278dB，原资格均失败。

## 伴奏与逐曲器乐保护

候选对并行LR1伴奏系数绝对误差四域均更大（不是把有符号CA增益直接当误差）。低频伴奏sidecar沿原支持、按arm保存，只描述，不新设资格门槛。

| 域 | 候选减对照：CA绝对误差 | 低频伴奏error-SNR dB |
| --- | ---: | ---: |
| MUSDB native | +0.004566 | -0.130701 |
| MUSDB weak | +0.004413 | -0.506295 |
| MIR native | +0.013030 | -0.106019 |
| MIR weak | +0.012676 | -0.202417 |

候选对自身起点CA绝对误差四域均改善；完整伴奏重建native MUSDB退而其余三域改善，低频sidecar MUSDB native略退而其余三域改善。不能概括为所有保护全面变差或已改善弱人声分离。

器乐以完整伴奏error-SNR差描述（高更好），同时在分析JSON保留逐曲removed_energy_relative_mix_db，不省略两种保护记录：

| 器乐 | 候选减起点 dB | 候选减并行对照 dB |
| --- | ---: | ---: |
| Beneath_v1 | -2.862711 | -3.121999 |
| Maldito_(8-Bit)_v1 | +2.424662 | +0.094959 |
| Wombat_Combat_v1 | -5.749449 | -1.586165 |

候选对自身/对照都仅Maldito改善，均值分别-2.062499/-1.537735dB受Beneath/Wombat原高SNR影响；不能解释为三首等量绝对损伤。

## 真实执行与可恢复证据

- 196 summary4250前台session72849，完成chunk ebf6e8，真实native exit0；内置verify_output通过。原31首177视图、逐曲聚合/原政策、权重评分哈希、完整model/Adam/模式/参数顺序/更新暴露、源ARMS[1]/停止与本臂LR/kill身份独立核对，0模型前向，不重生成4000。
- 196 listen4250前台session22582，完成chunk1cd04f，真实native exit0、内置校验通过；固定NightOwl/abjones_2/Beneath原三源各11变体33WAV一次生成于results/mel_lr_scale_review_20261004/listen_step_4250。同输入与每源共同音量（追加播放增益均1.0），无按分数换歌，原4000/3750/3500音频不重导。
- summary SHA d6c3b25a60147d9ba5a2195996b14b8e4814826bb038229940173c04f8fe7827；4250 PT SHA ef072b21ff4f097e8d6ef2d36ad7dacd6c21eccf48a5b63c23f0e1f40d8b9fee；receipt SHA f758c13705dd7fbc7773284ab390cab09fd3b9fee14128918a94307579372a05；score SHA4774089d5c3adccd31d6054b876dafb2ae7eda5497387323be69e7738e809089。
- listen review SHA443da50d766168373ed1013e2860b7aa7db056b379d68fb18ba245747f684d64；plan SHAe671c84b7aa079d8d067cc359556ba993fd9eed7fc86fd851244248c2855807f。试听history128/block256，TRAIN/DEV整352窗warmup96/native98*256..length-2*256，不混称相同上下文。
- 独立monitor保存stage_4250_analysis/review、每域每指标及逐曲器乐历史最佳（非整体排名）、进度及真实观察，analyzed=[4000,4250]，listening=[4250]，pending=[]，next4500。不写运行中训练或封存绑定。

20:59:27Z动态核对一个实际worker72684（创建2026-10-04T03:34:50.171771+08:00，194 train/CUDA/新审批/输出命令匹配）和其shim51952链。状态running/training/4330、追加330/500，CPU6571.125秒较前次20:30增加2185.546875秒，stdout4352bytes增长、stderr0，同PID列为GPU进程（WDDM显存N/A不归属整卡），磁盘24.823334GiB；无复核worker，4500文件与相符detached_exit尚无。不是完成500步或正常退出，不重复启动。

## 后续边界

仅4250一个新增阶段，不证明LR过大、已收敛、无学习、平台期或容量/全库/长源/板端因果。等待4500完整追加500/errornull、匹配实际detached_exit0、无worker后194verify及196summary/listen4500各一次。4500硬止不变，不机械改LF/lambda/器乐权重或删保护。人工4250听审及独立真实验收PENDING，旧DEV不是新盲测、伪标签不是最终真值，全程NONRELEASE。

按OpenAI Docs的[同聊天定时巡检文档](https://learn.chatgpt.com/docs/automations)，仅把本阶段完成证据写入原heartbeat提示；同聊天、20分钟间隔、ACTIVE及重要结果才通知保持不变，没有新建或暂停巡检。
