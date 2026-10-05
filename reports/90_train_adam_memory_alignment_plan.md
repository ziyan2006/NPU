# 90 — TRAIN-only Adam存量方向与即时目标对齐：有界预声明

2026-10-04，NONRELEASE，准备阶段。报告89的500共同更新已完整结束、真实exit0及独立复核通过。当前没有197工具/测试/封存plan/worker；本报告不启动GPU训练或延长历史4500限。

## 新问题与区别

半LR终点对并行LR1四域残留少，但两弱域人声重建更低、CA绝对误差四域更大；对自身三器乐均退。原169/175只诊断2000..2011固定输入的原始参数梯度几何，明确不是含历史Adam动量/二阶矩的方向证据，均已完整完成，不重做。

新问题仅为：当前已存Adam方向在少量未用于本轮更新的原TRAIN输入上，与原各目标的即时梯度是否同向？不同LR历史是否留下不同的存量方向对齐？它不证明LR太大、动量错误或重置Adam有利；不据小样本调beta/epsilon/清空moments，也不据此自动选择模型/训练变量。

只读三预定状态，不按分数换点：
1. 本LR轮4000共同基线ARMS[0]（与源LF32 ARMS[1]完整状态相符），PT SHA927806c04e71729b1a1d7a5ebb3fccafecb85790979222bdad6966b6e0729e18。
2. 本LR轮4500 lr1_control/ARMS[0]。
3. 同4500PT lr_half_candidate/ARMS[1]。
终点PT SHAb3a0450ecd9bba259981bd47f8a49c0fee1a997d7a087a72570f589cc14ed5b3。共同4000不是新训练阶段，两终点历史各500共同更新；全部同LF32/.2完整辅助/器乐4/原配置。保存本臂实际LR和累积状态，不统一LR或用均值挑赢家。

## 固定输入、支持与计算预算

固定源终点cursor4500的下一三个原TRAIN counters4500..4502，共3批18唯一输入/目标/元数据槽；三状态共9模型-批探针、54模型槽检查，不是54独立输入或训练更新。不按活动/开发分数补采样；弱域若缺失明确记录。

使用原Mel六槽TRAIN生成/PCM/目标/角色/增益[-12,-6,0,6]/顺序/窗口算法，独立诊断副本在cursor4500取原下一输入并记录完整x/v/meta/PCM哈希、3模型对同批一致。不调用仅2000..2011的164.collect或固定3000..3011的182.collect_fixed_train去越界，不修改这些工具守卫；使用新封存只读输入适配器验证实际原stream算法与游标。原协议、教师/输入锁不变；源GPU训练已经结束，不恢复它。

原TRAIN整352窗/warmup96/native98*256..length-2*256不变，全部固定kill32（零FFT0..3且原过渡），器乐基础权重4/六槽分母6，真值TRAIN原RMS>1e-4/Gram>=1e-3，伪标签/器乐辅助0。伪标签不是最终真值。保持原六microbatch操作顺序，不把六槽批前向等价假定成逐槽。

CPU2线程FP32前后向/complex64重建。仅detach向量点积/范数/余弦用FP64统计，不改推理精度。用autograd.grad取返回向量，不写parameter.grad；不构造优化器、不调用Adam.step、无CUDA/参数更新/新预测权重/PT/音频/DEV/验收。54主槽前向及梯度分解；每模型首批额外原194完整组合梯度等价检查（同已用输入、18重复槽前向），总72槽前向，仅3唯一输入批。额外执行数须真实单列，不隐瞒为54或新的输入。

## 存量方向定义和解释

逐实际Adam param_group/参数ID/parameter_names顺序读取exp_avg、exp_avg_sq、实际step、betas、eps、LR及所有group标志；全状态只读。核对原weight_decay=0、maximize=false、amsgrad=false及foreach/fused身份，不凭默认值猜；意外配置差异先失败诊断，不改旧状态。

用该参数自身实际t定义偏差修正存量方向：
u = [m/(1-beta1^t)] / [sqrt(v/(1-beta2^t)) + eps]；
d = -实际groupLR*u。
方向数学用FP32副本，FP64只detach统计。d是已存moments所描述的最后一次下降方向代理，不是PyTorch舍入后的实际参数差、不是加入下一批梯度后的虚拟下一Adam更新，也不是新训练步；不把4501当实际已更新。分开记录未乘LR的u和含本臂实际LR的d，避免把半LR幅度差当方向改变。

在对应模型当前参数处拆分原真值基础、器乐基础、伪标签基础、cv²及(ca-1)²即时梯度，保留原系数[1,4,1,.2,.2]/分母6。新主要结果为各梯度g与u、d的点积/余弦及非零/不可用计数，而不是重复输出旧梯度组对冲突表。g·d<0只是一阶局部下降指示，不能声称真正应用后loss/音质必定改善；g与u正对齐对应下降方向，符号解释不得混淆。零梯度/方向余弦记unavailable，不加epsilon伪造有效余弦。分模型/角色/实际增益活动记录，不把重复模型当独立样本。

## 实现、门槛和封存

拟新scripts/197_diagnose_train_adam_memory.py、scripts/_test_train_adam_memory.py、docs/train_adam_memory_protocol_20261004.json；新results/train_adam_memory_20261004及独立monitor目录。先查文件/进程/产物，不覆盖，工具/单元/草稿真实审查完再prepare，不能缺工具反复封存或借193训练审批启动诊断。

测试覆盖所有合法增益/角色、字面CPU/FP32/形状/finite、kill32/matrix身份、原194损失和梯度分解等价、参数-ID-name/moments/step/beta/eps/LR类型敏感映射、bias-correction独立标量算术及零范数、正确g·d符号、逐组系数/归一化、原3输入跨模型哈希、输入目标/权重全部模式/既有grad/RNG/moments/源游标日程不变、CUDA未初始化（fresh无Adam夹具）、准确9/54/额外18计数、已有输出拒绝，以及文件行和嵌入行各验seal后的完整类型敏感对称比较。允许容差的组合梯度必须提前规定并如实记录，标量/波形位一致和梯度数值容差不能混称逐位通过。

新plan绑定197/测试/协议/本报告90/报告89/86/85/84、4500真实completion/匹配launch-exit/194verify日志/196summary与独立completion_review、源4000/4500PTreceipt/审批、实际原输入stream/损失/矩阵/依赖/锁缓存及必要旧175/169代码（只读复用）。不重复旧全轮verify/summary/listen/diagnostic/单元或改任何历史绑定。启动前disk>=12GiB、无实际活跃训练/诊断/复核；前台一次执行保存真实session退出码/log，不伪造WMIdetached_exit。

完整9模型批/54主槽检查+18重复参考槽、updates0/CUDAfalse/errornull、真实native exit0及无worker后独立verify一次，保存新monitor结果。每批核对原输入/角色、权重/所有模式/grad/RNG/Adam/游标不变、有限与CPU身份；非有限/绑定/状态/角色变化或资源不足立刻失败保留证据、只读诊断，不盲重启。单次0批/文件不变不是卡死，须动态PID创建/命令CPU/日志/产物变化核活性。

## 后续与授权边界

本诊断不能证明模型容量上限、LR/动量因果、全库/长源/板端/质量收益，不设eligible/rank，不按loss/单均值选；无需新GPU预算。若存量方向和即时梯度差异未解释取舍，也不编造结论或反复重复探针。之后若选择训练，另记问题/单变量/默认<=1000共同步/预算资源停止条件、新协议审批/完整modelAdam模式参数顺序CPU-CUDARNG游标日程迁移、实际<=3唯一CPU/CUDA机制门槛及WMI启动，不能默认授权清空Adam或抬旧限。

本次不改图/前端/LF网格/板端状态，若后续意外涉及图/NPU周期BRAM/DSP先独立Z7020审查。所有历史失败记录、脚本测试协议审批数据锁快照标签缓存、冻结学生、板端镜像、vitis_journal.py和其他未提交修改保留。NONRELEASE，人听及独立真实验收PENDING。巡检20分钟ACTIVE，资源/权限安全障碍通知但不自行暂停，不为持续无证据重复训练。
