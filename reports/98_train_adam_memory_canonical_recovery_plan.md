# 存量Adam方向9模型批：新201规范恢复计划（NONRELEASE）

报告97的新200已经以0前向真实exit0独立核对199现有6+6桥接，原严格分项失败保留、直接原194权威与独立参考逐位相同；报告95的199.verify exit1和报告91的197failed3/9均不修改。持续授权允许继续原TRAIN存量方向问题，不授权调Adam/LR/beta/LF/损失/保护、训练或音质发布。

接续新scripts/201_diagnose_train_adam_memory_canonical_recovery.py、新独立测试/协议、新results/train_adam_memory_canonical_recovery_20261004及独立monitor。此报告只是设计，不是缺工具的封存实验plan：目前无201工具/测试/协议/plan/worker；下一次先查实际存在产物，禁止覆盖或重复启动。真实新单位及草稿门槛通过后才prepare。

## 保持原90固定范围

三个预定完整状态：本LR4000共同基线ARMS0（PT SHA927806c04e71729b1a1d7a5ebb3fccafecb85790979222bdad6966b6e0729e18，等同旧LF32源ARMS1），同本LR4500 PT SHAb3a0450ecd9bba259981bd47f8a49c0fee1a997d7a087a72570f589cc14ed5b3的controlARMS0/halfARMS1。不按旧分数选模型。

固定原TRAIN counters4500..4502，直接使用原stream只读算法与197已封存三输入身份，不改旧164/182范围守卫；跨模型x/v/meta/PCM哈希一致，不重采样缺失弱档。3唯一批/18唯一槽，9模型批/54主槽前向；每模型首批额外6槽原194独立参考，共18重复参考/72槽前向。不与197部分行或199/198模型调用拼接，不把重复模型或参考算独立样本。实际MUSDB[+6,0,-12]/MIR[+6,-12,-12]/器乐3/pseudo9，缺失native/-6不补选；pseudo非最终真值。

CPU2 FP32前后向/complex64，仅detach统计FP64；autograd.grad不写parameter.grad。0Adam构造/step/CUDA/更新/PT/音频/DEV/验收，不加载活训练engine绕开旧stop。全部源model/modes/已有grad/完整Adam ID-name-moments-step-beta-eps-LR-flags、游标日程/RNG及输入逐批只读不变；只读完整Adam原weight_decay0/maximizefalse/amsgradfalse/foreach-fused身份，不猜默认、不清零或迁移更改。u/d=-实际本臂groupLR*u分别方向与幅度，为存量方向代理，不是实际delta或训练4501。

原6microbatch/352窗warmup96/native25088..89344/kill32/完整.2辅助/器乐4/分母6/活动RMS>1e-4与Gram>=1e-3/角色增益顺序、同Mel真值不变。

## 规范方法和硬门槛

每槽一次主前向，直接对原194combine_slot_loss(base,原完整aux,meta,slot,4,info)求总梯度；按原六槽顺序FP32累计，唯一总方向权威。独立测量真值/器乐/pseudo base、cv²、(ca-1)²及原完整辅助梯度；五组系数[1,4,1,.2,.2]/6，组加和差如实记录，不补误差给组、不用完整减某项造梯度。

使用base+原完整辅助的规范组合逐槽及总量与直接权威原rtol2e-4/atol2e-7核对，是strict不是逐位等同。活动槽保留全部原完整辅助vs独立分项参数和失败元素/抵消、不能当同权威；完整辅助vs完整波形余切VJP要求逐位相同，完整余切vs分项和、完整参数vs合并余切VJP仍原strict通过。相关VJP无额外前向，实际前向开始/完成计数和未提交失败暴露都记录。

各模型首批6独立参考以原194完整标量及原操作顺序，同输入/PCM/wave/mask/base/full/组合loss身份且逐槽/总权威参考梯度逐位相同；先规范+同图门槛再提交通过，不用旧197未提交行冒充参考。保留分项strict失败可作为测量，不是放宽原strict或掩盖失败；新硬门槛不通过即失败。

主要新结果是三个状态在同三批的g与u/d点积/余弦及各实际组测量，不重复旧组冲突表或只看loss；零范数unavailable，不加epsilon造余弦。g·d<0仅一阶局部，不推实际应用音质收益、LR/动量因果、全库长源板端/容量或eligible/rank。模型重复非独立样本，不把本LR变化与存量方向作因果结论。

## 新门槛、封存与资源

实现前完整读实际199/200/197/198/194/175/170/159/176/stream/frontend/matrix/锁与cache依赖。新单位覆盖合法[-12,-6,0,+6]/角色/skip、原strict失败保留/同图VJP/权威参考FP32顺序、完整Adam类型敏感映射/独立标量bias correction/零范数符号、无别名、全状态RNG已有grad保护/fresh CPU无Adam不初始化CUDA、9模型批54+18真实计数/已有输出拒绝；端到端序列化必须覆盖父文档以及文件/嵌入的行和totals分别验seal后完整对称比较，防止199单边seal回归。不能只测行未测汇总序列化。新草稿全部实际通过才prepare。

新plan绑定实际201工具/测试/协议/本报告/97/95/96/94/93/92/91/90、197全部失败、198成功、199完整run及失败verify、新200实际28单位/prepare/verify/log/verification/completion_review/aggregation、原LRPTreceipt审批/实际launch-exit/194verify/196summary、输入锁cache/矩阵与所有实际代码依赖，不改历史绑定。源200的379绑定身份保留，不借训练审批。

disk>=12GiB/no duplicate仓库训练诊断复核后前台一次保存真实session/native exit/log；非WMI无detached_exit。不固定历史PID，启动/0批/单次不变不判卡死，核动态PID创建命令/CPU/日志产物；venv shim父子非双任务。非有限/绑定或源状态变异/角色错误/意外CUDA/资源不足失败留全部证据，不盲重启；完整9批54主+18参考/errornull/真实exit0/no worker后在新目录独立0前向verify一次，不覆盖或重复已成功产物。

只有新201全范围真实成功后分析一次并选择后续有界单变量探索。当前没有新GPU任务，不据一个桥接样本启动训练。原197未恢复/人听与独立真实验收PENDING，全程NONRELEASE；20分钟同聊巡检继续，不暂停。
