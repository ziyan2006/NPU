# 63 — 伴奏系数项消融3250步复核

2026-10-03 18:39（Asia/Shanghai）。首个新增完整阶段3250已复核，训练仍运行；不是500步完成。结论是重建改善、残留未改善，移除伴奏系数项尚未呈现有意义的优势。全程NONRELEASE，两臂eligible=false、release_selection=NONE。

## 运行与独立证据

3250评分、完整PT及提交回执相符；180 summary3250前台session47395实际exit_code=0，180 listen3250前台session21143实际exit_code=0。哈希、模型/Adam暴露、原31首177视图、逐曲聚合、原政策、源候选3000与停止证据通过。原3000是基线，3250仅一个新增评分，不能称三阶段平台期。

最终本次观察仍为training/step3320/additional320/error=null；状态PID23356与回执子链、实际创建时间和命令行一致，CPU增至5808.844秒。尚无对应detached_exit，不能声称本轮正常结束。没有重复启动、改旧限或修改任何封存绑定。

- 3250全状态SHA256：`f802cffe4a490c2fb7292d4880c26128836ab005217daa63d7f1bcf76a6eefc1`。
- summary SHA256：`e8fd291805ba1fd423ee4aa1d10e73d6f0d8d7182a2a587877aeb52744633d54`。
- listening review SHA256：`a513dc1caf5a37d343c529a320a5116d5b0404f9ee753c2824997f1ec5ec77d1`。
- [完整阶段证据](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_component_ablation_monitor_20261003/stage_3250_review.json)、[逐域分析](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_component_ablation_monitor_20261003/stage_3250_analysis.json)。

## 仅移除伴奏项：对照差异很小

控制aux_full_control和候选aux_residual_only同Mel/真值目标、基础器乐权重4、残留cv²系数.2；候选仅将伴奏系数项.2变0。不是HT教师比较，不按不同训练loss排列音质。

| 比较 | 人声重建error-SNR变化（高更好） | 投影残留变化（低更好） |
| --- | ---: | ---: |
| 控制3250相对冻结 | +0.4399dB | +0.1340dB |
| 候选3250相对冻结 | +0.4323dB | +0.1140dB |
| 候选相对控制 | -0.0077dB | -0.0200dB |
| 候选相对3000起点 | +0.3820dB | +0.9415dB |

候选虽比控制略少残留，但两臂平均残留均比冻结更多；候选相对起点四个人声域全部增加残留。重建改善不能描述成“人声删得更干净”。

| 域 | 候选相对起点重建dB | 候选相对起点残留dB | 候选相对冻结残留dB | 候选伴奏系数绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| MUSDB native | -0.1808 | +1.4969 | -0.2029 | 0.08476 |
| MUSDB weak -12dB | +1.0481 | +0.8213 | +0.4137 | 0.03250 |
| MIR native | -0.0403 | +0.6993 | -0.5930 | 0.15303 |
| MIR weak -12dB | +0.7010 | +0.7485 | +0.8383 | 0.10860 |

伴奏系数保留相对起点更接近1，但弱域剩余人声投影约0.8594/0.8036，仍是短板。这与“少删”权衡一致，不是全库因果、历史Adam方向或LF44容量上限的证明。投影系数不是能量百分比或主观音量。

## 纯器乐必须逐曲看

| 曲目 | 3000起点error-SNR | 控制3250 | 候选3250 | 候选相对起点 |
| --- | ---: | ---: | ---: | ---: |
| Beneath | 91.8233 | 73.1716 | 72.7275 | -19.0958dB |
| Maldito | 20.0148 | 25.5208 | 25.8942 | +5.8795dB |
| Wombat | 58.7275 | 66.7761 | 67.9521 | +9.2246dB |

候选平均比起点低1.3306dB，受Beneath高分回落影响；其绝对分仍很高，另两首改善。候选相对控制仅2/3首改善，不能用平均分宣称全面安全或损伤已严重可闻。

## 最新固定三源试听与下一步

[3250步33个WAV](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_component_ablation_review_20261003/listen_step_3250)已一次性导出并核验。原预选NightOwl约6.8秒、MIR abjones_2与Beneath各20秒，每源11变体：原音、真参考、冻结学生、3000起点、控制与候选的人声/伴奏；全部同输入同音量，没有按评分换歌。

候选伴奏：[NightOwl](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_component_ablation_review_20261003/listen_step_3250/source_01_aux_residual_only_backing.wav)、[MIR](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_component_ablation_review_20261003/listen_step_3250/source_02_aux_residual_only_backing.wav)、[器乐](D:/WORKBUDDY/STEM/npu-stem-repo/results/mel_component_ablation_review_20261003/listen_step_3250/source_03_aux_residual_only_backing.wav)。

MIR原源拼接样本289296、659034仍保留，跳变不直接归因模型。CPU FP32参考不是板端连续流验收；人工听审PENDING，旧开发集不是新盲测。

不修改活任务；继续既定3500硬限，等待第二个新增完整阶段。若终点仍无弱域残留收益或仅伴奏损伤，按预声明转查LF44、上下文与推理重建，不机械提高lambda、器乐权重或删更多保护。巡检保持ACTIVE。
