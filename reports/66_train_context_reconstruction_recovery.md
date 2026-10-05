# 66 — TRAIN上下文诊断失败与显式恢复协议

2026-10-03 20:04（Asia/Shanghai）。报告65的零更新诊断181已封存后启动一次，但在首个counter3000读取前失败，前台session88459真实退出码1；0/36模型批次、0模型更新，无CUDA、没有输入证据/行/diagnostic。实际worker63772已退出，12:02:24Z检查没有181或178的Python进程。失败不是正常完成或音质结论。

原因已只读确认：181复用164的collect，它明确只允许该历史诊断的2000..2011；本轮计划是3000..3011。旧保护正确拒绝范围，没有缺数据、模型非有限或GPU故障证据。181脚本、绑定测试、plan、状态、日志全部保留且不修改；[失败复核](D:/WORKBUDDY/STEM/npu-stem-repo/results/train_context_reconstruction_monitor_20261003/failure_review.json)记录真实退出、原始证据哈希和原因。

持续优化授权内选择新的182_diagnose_train_context_reconstruction_recovery.py与_test_train_context_reconstruction_recovery.py，使用results/train_context_reconstruction_recovery_20261003、独立monitor。不是重新运行181、覆盖失败证据、改变历史范围或借旧审批训练。

唯一修复是新的固定3000..3011输入读取器，依照177 ComponentStream.next_batch的同一true.crop/seed/counter和三个crop_recipe(counter*3+index)公式读取原六槽。拒绝bool/非整数/范围外counter，不放宽164或任何训练上限。复用不可变181的源核对、四种上下文、LF44、重建与完整结果检查函数；不运行181.run或修改其模块全局值。

先新单元与继承38项冻结181合成回归，再固定12批72槽真实CPU无更新输入审计。新读取器与隔离的原177.next_batch逐位比较x/两个臂的同Mel/真值targets和metadata，同时记录输入/目标PCM哈希、严格角色、CPU RNG不变和未初始化CUDA。只让隔离的审计流游标按原公式移动；源全状态游标/模型/Adam/模式不动。审计不是模型前向成绩、恢复训练证明或音质改进。失败保存证据，不跳门槛。

新plan绑定182/新测试/本报告/冻结181及其测试/报告65/64/36/3500源完成证据/真实依赖和输入锁，并绑定失败复核及原失败plan/status/log、通过的真实输入审计。原失败记录中UTC精度尾零序列化的源预检差异已在181封存前修复，与本次collect范围失败分别保留日志，不混称旧训练失败。

恢复诊断问题和范围不变：固定3000..3011的12批72槽，起点3000、aux_full_control3500、aux_residual_only3500三模型，36模型批216槽。CPU2线程FP32前向/重建，只有detach统计可FP64；无反向/优化器/param.grad/CUDA/模型更新/音频导出/独立验收。保留权重、全部模式、RNG、grad；源Adam/日程/游标不动。

严格沿用报告65的同谱352帧、整窗/history128-block256/history128-block16/reset16负对照，126pv和110 product_vocal一致性、STFT/ISTFT往返及LF44强制零bins频谱能量。共同因果比较区间130*256..length-2*256；原98区间仅描述，伪标签不当最终真值，器乐零参考比例skip，不按结果另选批次。不改生产图/frontend/LF44/精度/评分/输入/标签，不把小样本推广全库或板端。

审计通过且新plan封存后，确认磁盘>=12GiB、无重复活跃训练诊断，前台启动新任务一次，保存真实exec session退出码；不是WMI，不伪造detached_exit。启动不等于完成。已有run/status/rows/diagnostic拒绝覆盖；动态PID/创建时间/命令行/CPU及文件变化检查活性。完整36批216槽、真实exit0、无真实worker后独立182 verify一次；失败保留证据只读诊断，不盲重启。

全程NONRELEASE，release_selection=NONE。3500人工听审及独立真实验收PENDING。没有新GPU训练或板端授权。巡检继续每20分钟ACTIVE，正常无可行动结果安静；后续有界探索仍需新的预算/协议/审批/目录/全状态迁移和真实CPU/CUDA机制门槛，改图先审Z7020算子/周期/BRAM/DSP。不提交推送、不上传音乐、不写SD、不部署、不新聊天或子代理、不改驱动注册表推理精度；保留所有历史绑定及其他未提交修改。
