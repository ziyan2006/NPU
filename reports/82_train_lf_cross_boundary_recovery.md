# LF交叉边界诊断191失败与新192有界恢复计划

2026-10-04，NONRELEASE。191 run前台session21731/chunkdcbd79真实native exit1；不是正常完成。终态failed、已提交2/36模型批12模型输入槽24槽-边界重建、error=Original gainmix activity identity required，0更新/CUDAfalse，无diagnostic。18:48:11Z worker71304及shim54516已消失。

失败根因在新191.bucket：错误拒绝任何正vocal_db；而原TRAIN固定计划合法取值-12、-6、0、+6。第一个+6为counter3002 MUSDB Jay Menon - Through My Eyes.stem。原固定12批实际包含24真值声槽，+6在MUSDB3002/3011和MIR3004/3010。原输入/角色均正确，非模型或GPU故障。40合成单元实际通过，但元数据夹具只取counter3000，遗漏增益档覆盖；不能将其成功冒充完整真实支持验证。

191代码/测试/协议/unit_gate/plan/status/输入/日志及row00、01全部保留封存，不修改、不在原目录重跑、不把部分行排名或伪称36完整。独立failure_review.json保存原始SHA与真实前台退出证据，report81仅记录当时真实启动，不回写成成功。

持续优化授权内新建192_diagnose_train_lf_cross_boundary_recovery.py、_test_train_lf_cross_boundary_recovery.py、docs/train_lf_cross_boundary_recovery_protocol_20261004.json，使用新results/train_lf_cross_boundary_recovery_20261004与独立monitor。复用191已审计算的同掩码双边界设计但新工具隔离、不全局override、不改191；唯一修复为正确接纳字面增益[-12,-6,0,6]并在前向前核对全部72元数据。

分组预声明：0=native，-12=weak_minus12，-6=attenuated_minus6_descriptive，+6=boosted_plus6_descriptive；后两类不混称原DEV弱域或native。按原固定输入统计，MUSDB实际0档4、-12档3、-6档3、+6档2；MIR实际0档2、-12档6、-6档2、+6档2；新单元对全部182已提交元数据核对这些计数并封存。器乐12和pseudo36保持单列，低活动保留。不调采样/目标/增益/kill/loss/模型/支持。

恢复必须先补完整原增益档、全72真实元数据分组与不支持类型/增益拒绝单元，完整旧计算回归，实际exit0及真实源/失败证据检查后封存新plan。新plan绑定原失败全部SHA、新代码/测试/本报告82/报告80/81、源完成证据/状态/输入锁/矩阵及实际依赖；不借旧GPU审批。

预算仍固定同12批72唯一输入、三权重36模型批216模型输入槽、432槽-边界重建。新恢复对全部36批重新计算，不合并或隐瞒191部分2批；这是已明确原因和门槛的新零更新恢复，不重复任何已完成历史診断。191第三批已有前向但未提交，失去的预测不当完整结果。新全轮36与旧失败执行分开记。

CPU2线程FP32前向/complex64重建，仅detach统计FP64；无反向/优化器/grad计算/CUDA/更新/音频/DEV/真实验收。权重/全模式/RNG/既有grad/Adam/游标/调度前后不变。边界44/32原native25088..89344支持不变。完整36/216/432、实际前台exit0/无worker后独立verify一次，文件/嵌入seal完整类型敏感对称比较。启动不是完成；再次失败保留证据不盲重启。

磁盘>=12GiB/无重复任务才启动一次；现在准备阶段尚未启动192。巡检ACTIVE每20分钟；失败已通知，保留轻量检查，不暂停或停用。人工4000听审/独立真实验收PENDING，NONRELEASE/不部署。
