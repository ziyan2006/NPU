# Mel LF32更新幅度单变量试验预声明

2026-10-04，NONRELEASE，准备阶段，工具/测试/协议/审批/worker均未生成。持续优化授权内自主选择；不延长历史4000轮或复跑已完成诊断。

## 问题、假说与预算

报告79的3750到4000 LF32残留改善而人声重建回退，两个新增阶段还在变化，不能说平台期。报告84区分边界直接效应和训练权重效果后，显示固定TRAIN弱人声残留/重建与伴奏低频保护存在取舍，未证明LF32胜出或LR过大。用单一实际更新幅度系数检验：较小步长是否在同预算下改善这些取舍；可能无改善或只减慢学习。不得预先挑胜者，不机械再次下调LF/升lambda/器乐权重/删保护。

新独立轮500共同更新4001..4500，4250/4500评分与全状态提交；4000同权重/同32为共同零更新起点，仅两个新增阶段。硬4500、身份/非有限/资源/恢复失败即failed；旧4000终点证据和上限不改，部分/零步不称500完整。第一step4001使用源cursor4000下一原输入。

## 源与唯一变量

源results/mel_lf_boundary_20261003/NONRELEASE_lf_boundary_step_4000.pt SHA94323f8a30fb5e5decb65d4c110fd5252d0916de68507f42af555b3f539a8f6d，只选LF32候选原ARMS[1]（kim_melband_waveform_candidate），不是LF44控制、3000/3500或按部分成绩换点。

两个新臂完整同model/Adam/参数顺序/全部模式/更新暴露/源CPU-CUDA RNG/游标/原配置/best/stale/patience_anchor分叉。源stop4000、legacy[3750,4000]及累计状态实际读取保留，不凭计划猜测。新活跃stop初始None，旧政策事件另记legacy_stop_events，4500硬终止；不重置既有LR配置/最佳/累计stale/anchor。

两臂均固定kill32（训练/DEV/试听一致）、同Mel/真值、基础权重[1,1,4,1,1,1]/分母6、cv²和(ca-1)²同.2、真值TRAIN原RMS>1e-4/Gram>=1e-3；器乐/pseudo无辅助。图/参数/前端/分析合成矩阵/24首/六槽顺序/PCM/增益[-12,-6,0,6]/采样/窗口裁剪/严格FP32运行环境/原学习率日程函数不改。

唯一变量为实际Adam参数组LR：control=原learning_rate(step,source_config)*1.0，candidate=同值*.5。源参数组LR逐位保存/迁移；首次新更新开始才按明确外部系数应用，不能把候选半LR伪称原运行恢复位一致。两臂4000模型和Adam状态应相同，下一输入相同；不同时改beta/epsilon/梯度裁剪/归一化/目标/loss或重置调度。兼容键解释htdemucs_waveform_control=lr1_control、kim_melband_waveform_candidate=lr_half_candidate，两臂同Mel非教师比较。

## 新工具与封存门槛

拟新scripts/193_prepare_mel_lr_scale.py、194_train_mel_lr_scale.py、195_start_mel_lr_scale.ps1、196_review_mel_lr_scale.py、_test_mel_lr_scale.py；docs/mel_lr_scale_protocol_20261004.json及独立results/mel_lr_scale_*_20261004目录。先核对现有文件/任务，禁止覆盖。不能借旧189审批启动，不能改历史186/192的守卫或任何绑定。工具齐备、完整单元/草稿审查后才prepare，不能缺工具反复封存。

单元包括单变量LR应用顺序/4001实际值/无意调度重置、控制与源186同32全辅助损失/梯度/更新逐位一致、同输入/全部增益合法组/活动role/器乐权重、真实有限非零分支、Adam参数顺序/mode/RNG、磁盘位一致/保存无别名/第二臂含CUDA RNG全事务回滚、已存在输出拒绝及完整状态/退出回执/资格政策校验。CPU无CUDA统计单元须fresh无Adam夹具，避免健康检查触发的旧夹具问题。

固定真实下一TRAIN输入零更新审计先验证控制原损失/梯度/wave与旧186 LF32逐位一致；LR系数不应改变本步原梯度或损失，仅影响Adam更新。源非零梯度/实际参数差异必须真实证明，不能合成代替真实。最多3唯一真实输入步4001..4003 CPU/CUDA完整状态机制，严格源FP32runtime、源CUDA RNG恢复、源CPU显式迁移不是跨设备数值resume；各分支自身磁盘位一致重放/无别名/第二臂全事务回滚/PCM元数据跨设备一致。不同LR臂参数本就应不同，不能误要求两臂更新位一致；机制不是正式训练步或音质成绩。

封存新协议/审批绑定全部实际代码测试、报告85/84/79、192成功完成与独立verification/aggregation、191失败证据、源4000完整PT/receipt/审批/匹配launch-exit/真实verify/summary/输入锁/教师缓存和分析合成矩阵等实际依赖。使用已复核结束证据，不重做186/188或192/184/182、旧试听/机制/275首扫描。

## 资源、启动、独立复核与停止

新GPU训练磁盘>=12GiB、GPU空闲显存>=2300MiB、无重复活跃训练/诊断、真实机制全齐后WindowsPowerShell5.1/WMI独立启动一次，保存真实launch及helper实际exit回执，立即更新巡检真实工具/目录/预算。共享GPU高利用率非故障；不终止用户程序，不强开，资源不足通知后巡检保持ACTIVE。

4000两臂应同原源LF32评分，为共同基线，不重复旧4000音频。新4250/4500不同完整checkpoint/score才分析一次，评分先于全状态为pending、不排名。196独立核对原31首177视图/逐曲/原政策/哈希/完整modelAdam暴露/源ARMS[1]/kill32/LR系数/停止；比较冻结44、本臂4000LF32、共同边界并行控制，分开残留投影、人声重建、伴奏保护和逐曲器乐误删，不按loss或单均值选。

固定原预选NightOwl/abjones_2/Beneath、同输入同共同音量，每新增完整阶段一次11变体33WAV：原音/真声伴/冻结44声伴/起点32声伴/两臂32声伴，不按分数换歌、已有先verify不覆盖。不在worker CPU DEV评分时并行导出。源旧4000/3750 39WAV和3500 33WAV不重复。

完成须完整4500/additional500/errornull、相符实际detached_exit0、无worker，再194verify和196独立评分复核。没有退出证据不伪称正常；失败保留最後完整提交/全部失败证据，先只读原因，真实恢复门槛后新协议目录不盲重启。人工新轮听审/独立真实验收PENDING，旧DEV非盲测、伪标签非最终真值、eligible非发布，全程NONRELEASE。

本轮不改图/LF网格/算子或板端，仍实核张量/参数/NPU算子/周期/BRAM/DSP身份；若意外变图/状态先审Z7020不放宽目标。禁止提交推送/上传音乐/SD写入/部署/驱动注册表精度改动/改冻结学生或历史绑定。一次探索失败不证明平台期，不为持续而无证据重复训练；巡检保持20分钟轻量持续。
