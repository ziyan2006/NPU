# 76 — LF44/LF32新500步对照实际独立启动

2026-10-03 23:10（Asia/Shanghai）。新轮已通过隔离单元及真实CPU/CUDA机制门槛，WindowsPowerShell5.1/WMI独立启动一次。目前是CPU开发基线评分、step3500/additional0/error=null，不是已完成500步或已有训练音质改进。

原35项单元session29493真实exit1与审计执行顺序错误保留，具体见报告75。新隔离35项session67554真实exit0(chunk4900df)，固定TRAIN3500零更新audit session39590真实exit0、控制LF44损失/梯度/评分wave位一致，LF32梯度差L2=.2736628/wave max=.1076373、2活动。新189审批prepare session60258真实exit0(chunk959ec0)，审批results/mel_lf_boundary_gate_recovery_import_20261003/approval.json SHA97e6dd1d174a9dcde1dcada3b3833ffa1f2fa509c32ae249a549b9bec082e82a。旧185..188/原测试/协议/报告73/74/审计及新189/190/恢复测试/报告75/新审批全部绑定封存，不再修改。

186 CPU机制session85940实际exit0(chunk1f8010)、独立verify_proof exec6ae5a4 exit0；CUDA机制session60675实际exit0(chunk71d72a)、CUDA独立verify_proof与跨设备输入exec659250 exit0。CPU proof SHA9a38ee65e319097322e0b57d0553e2df3081b8b0df50734d750bb25649c23b47，CUDA SHAfae1969a12bb188045afc9b92c67867680cb72f489e09a0832df193bad281cb2。每端仅3唯一输入3501..3503，重复回放不计正式训练；源ARMS[0]完整model/Adam/参数顺序/全部模式/更新暴露/日程/RNG/游标精确迁移，磁盘位一致重放、保存无别名、第二臂包含CUDA RNG事务回滚、实际44/32边界及有效完整.2辅助、器乐权重4非零、输入PCM/metadata跨设备一致。CPU只是显式机制迁移；框架CPU Adam健康检查可能初始化CUDA context，不冒称跨设备数值resume。CUDA原严格FP32 runtime完全一致，没有改精度或驱动。

正式固定500共同更新3501..4000，每250评分/完整状态提交；3500是本臂零更新基线，3750/4000才新增训练阶段。源完整3500 aux_full_control ARMS[0]，不选旧residual-only；新两臂同model/Adam/modes/RNG/cursor/LR/best/stale14/patience_anchor，源stop3500/legacy[3250,3500]保留，新活跃stopNone、旧早停触发另记legacy_stop_events，4000共同硬终止，不抬旧上限或资格。

唯一变量kill44/32贯穿训练重建、开发与试听；htdemucs_waveform_control=lf44_control、kim_melband_waveform_candidate=lf32_candidate，非HT比较。全部六槽同Mel/真值、基础权重[1,1,4,1,1,1]/分母6、两辅助项各.2、活动门槛及原输入/增益/順序/窗口/日程都不变。候选32仍强制零FFT0..3，4/5及连续过渡变化，不是完全关闭保护；候选3500同权重不同输出是零更新直接效应，不当训练收益。图/矩阵/布局/NPU参考计划及周期/BRAM/DSP身份同，新增MAC/DSP/DMA/模型激活权重BRAM为0的结构判断不是实板验收；板端不动。

启动前磁盘约25.54GiB、GPU空闲2445MiB，无重复活跃任务。190新启动器显式传新审批；dispatch exec28df2c实际exit0只表示启动请求成功。实际回执results/mel_lf_boundary_launch_20261003/detached_launch_20261003_150859_6059935.json，UTC2026-10-03T15:09:02.5688865Z，helper39088/WmiPrvSE/in_job_object=false，shim64684、实际worker41728（23:09:03.087899创建）。父子链不是双训练，PID仅关联，以后用状态动态核对创建时间/命令行/CPU/本次Stdout/Stderr。

15:09:57Z初检running/development_baseline/step3500/additional0/error=null，worker CPU74.890625秒，起点完整PT/receipt已保存，stderr空，没有detached_exit。不能把CPU评分或单次步数不变当卡死，也不能把启动当完成。完成必须相符真实detached_exit exit0、完整4000/additional500、无worker，再186 verify/188独立复核。独立monitor保留preparation_review、launch_review、进度/已分析/历史最佳/通知，不写活输出或绑定。

188独立原31首177视图/逐曲聚合/原政策/模型Adam/哈希/源起点与停止/LF身份；比较冻结44、本臂起点44或32、源权重44、并行控制的残留/重建/伴奏及器乐。原政策不变，低频伴奏统计为单独描述sidecar，不用不同loss排名或DEV当新盲测。试听仅原三源同输入同音量，新阶段含起点44/32、冻结及两臂，共13变体39WAV；不重复旧3500的33WAV，不在CPU开发评分期间导出。

全程NONRELEASE，human3500及本轮听审/独立真实验收PENDING，eligible不等于发布。20分钟巡检继续ACTIVE、健康无新结果安静，原失败负证据不删；不部署、不上传音乐、不写SD、不Git提交推送、不改历史/板端/其他未提交修改、不新聊天或子代理。
