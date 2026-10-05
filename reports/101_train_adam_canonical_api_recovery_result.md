# 101 — 新202规范Adam方向诊断完成与独立复核

2026-10-04，NONRELEASE。新202已完整结束并通过独立0前向复核；不是仍4/9运行。新成功不回写197的failed3/9、199的verify exit1或201的prepare exit1，不拼接旧计算。人工听审与独立真实验收仍PENDING，release_selection=NONE。

## 实际完成证据

74新单位session56937/chunka3399d、prepare40672/eddee7、run21127/7181c5、verify14623/5515a7均真实native exit0。run与verify均前台完整消费日志后捕获退出码；非WMI，没有detached_exit，不伪造。历史worker85184及父venv shim92264已退出；shim不是第二诊断。

run_status complete/9模型批/error=null，primary started=completed54、reference started=completed18，Adamfalse/CUDAfalse/updates0。run_status的phase仍保留canonical9_requires_independent_verify；另存的verification.json已经独立通过，不能只按旧phase误判待复核。2026-10-04T05:02:33.3307812Z动态CIM无仓库Python任务，磁盘24.627590GiB。

新202工具/测试/协议/unit_gate/plan/inputs/status/9rows/54slots/18references/9totals/diagnostic/verification/日志保持封存，不再单位、prepare、run、verify或前向。计划403实际绑定；主计算和独立复核均检查父文档与文件/嵌入行、totals各自seal后完整类型敏感对称比较，没有删除单侧seal或放宽阈值。

| 文件 | SHA256 |
| --- | --- |
| plan.json | 657c694ce334958af1b4102847309c1ac8475c45c5fc61c7388cd12eeabb87cf |
| diagnostic.json | 26af9bfbcfadcf45fd69d0bfec30e4d3b530f67580fa1bf02c33b6ad93318d38 |

独立完成归档及所有新文件哈希见[completion_review](D:/WORKBUDDY/STEM/npu-stem-repo/results/train_adam_memory_canonical_api_recovery_monitor_20261004/completion_review.json)，逐批数值及边界见[aggregation](D:/WORKBUDDY/STEM/npu-stem-repo/results/train_adam_memory_canonical_api_recovery_monitor_20261004/aggregation.json)。这次归档只读JSON、0前向/autograd/更新，不是重复完整verify。

## 固定范围与数值门槛

三预定完整状态：共同LF32 4000 ARMS0、4500 control ARMS0、同4500PT half ARMS1；不是按分数挑选。固定原TRAIN counters4500..4502与197封存输入逐槽匹配。仅3唯一批/18唯一槽，9模型批/54主前向加每模型首批6原194参考，共18重复参考/72槽前向；模型重复和参考不算独立样本。MUSDB[+6,0,-12]、MIR[+6,-12,-12]、器乐3、pseudo9；不补缺失MIR native或-6，pseudo不是最终真值。

CPU2、FP32前后向/complex64，detach几何统计FP64。原352/warmup96/native25088..89344/kill32、6microbatch顺序、完整辅助.2、器乐4、分母6及活动门槛不变。源model/全部mode/已有grad/完整Adam映射和flags/游标日程/RNG/输入每批及整体保持只读；没有Adam构造、step、CUDA、更新、PT、音频、DEV或部署。

- 直接原194完整组合标量梯度、原六槽FP32累计为唯一总方向权威。54逐槽+9总量规范base+完整aux与权威的原rtol2e-4/atol2e-7比较全部通过；最大绝对差2.57045030594e-6，符合相对加绝对门槛，但不是逐位一致。
- 18活动主槽完整aux参数梯度与同图完整波形余切VJP全部逐位一致。完整波形余切vs分项和、完整参数vs合并分项余切VJP原strict全通过，最大差分别4.65661287308e-10、1.04308128357e-7。
- 独立分项参数和仍有2/18比较、9元素原strict失败，作为测量完整保留，不当作等价：control/c4500/MIR+6的原8索引[458,467,508,517,817319,817370,823653,823706]；half/c4500/MIR+6新增索引822869。half该失败元素绝对差2.25809344556e-7、原界2.12678373605e-7；整向量最大差3.34111973643e-7发生在别处，不能混同。所有失败元素及抵消记录留在原slot证据，不补误差给组。
- 每模型首批原194独立参考的18逐槽及3总权威比较全部逐位一致，输入PCM/wave/mask/base/full/combine标量身份一致。

## 新方向结果：不均匀，不是Adam有害结论

u由保存的moments、各参数实际step/betas/eps计算；d=-实际保存groupLR*u。它是存量moments最后方向代理，不是舍入后的真实参数delta，也不是加入下一梯度的Adam更新或训练4501。g·d<0仅一阶局部下降指示；g与u正对齐对应这个下降符号。零范数按unavailable处理，不加epsilon造余弦。

| 固定状态 | TRAIN counter | g·u | g·d | cos(g,u) |
| --- | ---: | ---: | ---: | ---: |
| common4000 | 4500 | -5.519180 | +0.000384787 | -0.043360 |
| common4000 | 4501 | +0.687129 | -0.0000479053 | +0.004463 |
| common4000 | 4502 | +24.420626 | -0.001702559 | +0.079798 |
| control4500 | 4500 | -38.920240 | +0.002444743 | -0.126412 |
| control4500 | 4501 | +6.287671 | -0.000394955 | +0.042414 |
| control4500 | 4502 | +17.436746 | -0.001095275 | +0.067168 |
| half4500 | 4500 | -25.000833 | +0.000785203 | -0.096117 |
| half4500 | 4501 | +26.517581 | -0.000832840 | +0.120792 |
| half4500 | 4502 | +30.646602 | -0.000962520 | +0.093930 |

三个状态都在c4500呈局部上升指示、另两批呈下降指示，不能把一个批的反向概括成全程无学习。五组保持独立原梯度和系数[1,4,1,.2,.2]/6：residual_cv2的g·d在9模型批全正、ca保护组全负。器乐组下降指示分别3/3、0/3、1/3；pseudo分别2/3、1/3、3/3，存在状态与批次差异。完整aux原梯度在4000首批为负、随后两批为正；两个4500状态三批全正。组统计不是严格可加权威的替代品，也不自动等于目标质量变化。

u范数分别186.498048/215.764917/264.153783，d范数.013002282/.013553095/.008296301；权重、moments、历史、LR均不同，不能把幅度差归因于LR单因素或据此排名。没有应用后loss/音质收益证据，不推出全库/长源/板端/容量/LR-beta根因；不清空Adam、调beta/LR、删保护或机械改loss权重。

## 下一步与巡检

原90/98的9批诊断目标由新202单独完成并复核；197部分计算仍失败，201只有准备失败。下一步另记报告102的有界EMA影子权重方案：保留原训练更新和保护，只比较权重轨迹平滑，不将方向代理当因果证据。当前无新GPU、无新训练工具/审批/plan/worker。全程NONRELEASE，20分钟同聊巡检ACTIVE，健康无新可行动结果安静，仅重要新结果/失败/确需人工处理时通知。
