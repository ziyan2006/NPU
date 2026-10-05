# TRAIN Adam存量方向诊断：197失败留证（NONRELEASE）

2026-10-04T00:09:01.5350289Z只读检查：新197真实前台session23132/chunk556697 native exit1，worker84684/venv shim83336均已消失，没有仓库Python任务；磁盘24.631947GiB。不是GPU任务，不存在WMI detached_exit。

最终59项草稿单元session92763/chunka9f8cf native exit0；prepare session77047/chunk460f0e native exit0。较早46/56项草稿单元日志全部保留。测试通过不代表真实诊断或音质通过。

results/train_adam_memory_20261004已failed：仅common_lf32_4000的counter4500/4501/4502三个文件行完整提交，即18主槽+6参考槽。第4模型批lr1_4500/counter4500在分拆梯度等价检查失败；未提交批的实际前向数量未完整记录，不能按零或完整六槽计算。没有diagnostic.json/verification.json，绝非完整9模型批、54+18槽成功。

失败地点：197.parameter_probe调用封存175.compare_gradients(go,gr+ga)。原逐元素rtol2e-4/atol2e-7下，824900个参数中8个不满足；最大绝对差4.333778633736074e-7（index508）、该元素相对差0.004891476593911648。这定位了数值等价门槛失败，不证明模型/GPU故障、优化器方向有害或音质上限。FP32运算路径/抵消是待独立验证的解释，不作为已确认根因。

0前向只读复核确认原计划全部源/代码/测试/锁/缓存绑定未变，三个文件行各自验seal，输入目标PCM元数据与inputs.json一致；无CUDA、更新0。失败时readonly finally未报告权重/全模式/已有grad/RNG/moments/输入变化。部分行不做方向排名或质量结论，也不调用197完整verify。

固定三批真实增益为MUSDB[+6,0,-12]、MIR[+6,-12,-12]，器乐3个零参考、pseudo9个单标签非最终真值。缺MIR native/-6及MUSDB -6，明确不补采样。

197、测试、协议、plan、inputs、status、三个row、日志、unit_gate从此封存，不修改、不在旧目录重跑、不调低门槛伪称通过。独立failure_review.json保存真实exit及完整哈希。plan SHA f01972948f18ea1ea5de7e4d02201bb86445a036f79d7fc87712f7f7949fefe8；输入SHA19caf93b84357d46e37e5614cc517fe2a3d8c75bfc27c24e2f22d63e5992cc0f。

后续先新隔离目录、有界只读数值路径诊断：同一个失败权重/同已锁定counter4500六槽，记录原严格门槛失败元素、分项抵消和波形余切/VJP路径差别，保持FP32与旧阈值不变。证明原因及新恢复门槛后才另立恢复协议，不盲重启197，不自动清空Adam/调beta/LR。

20分钟巡检继续；所有历史训练/听审/验证不重做。全程NONRELEASE，人工听审及独立真实验收PENDING。
