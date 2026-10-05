# EMA真实4500源状态接口：单位通过，完整集成仍待完成

2026-10-04。NONRELEASE；本轮不是新训练、质量比较或发布验收。

## 已完成的新进展

独立新增 `scripts/204_mel_ema_source_contract.py`，以及新接口测试和scope。实际入口读取固定LR4500 control ARMS0完整源；它不是训练器，不构造模型/Adam，不推进采样器或日程，也不修改旧4500停止状态。新接口只生成内存中的隔离source-only packet，明确标记 `new_training_authorized=false`、`complete_resume_container=false`。

新70项CPU单位在前台 `exec_command` chunk **3470dc** 真实native exit **0**；日志完整消费并保留。无持续session、无WMI启动，因而没有也不伪造detached_exit。没有重跑203的46单位、202或任何已关闭训练/诊断/verify。

本轮前置元数据检查一次CPU PT反序列化，随后新单位实际入口另外一次CPU PT反序列化；整个单位套件只反序列化源一次。这里的PT读取不等于活学生Module加载或模型前向。

## 真实源结论

固定完整PT：`results/mel_lr_scale_20261004/NONRELEASE_lr_scale_step_4500.pt`，SHA256 **b3a0450ecd9bba259981bd47f8a49c0fee1a997d7a087a72570f589cc14ed5b3**。

- 所选臂仍为预定 `htdemucs_waveform_control` / LR1 control；不按分数选择赢家。
- 原22参数名字、顺序、形状及FP32位、27模块mode均保存；匹配Adam22个ID的m/v/step，以及完整类型敏感group/options全部检查。每个参数step=4500，实际group LR=**6.281416799501188e-5**。
- 实际源累计stale=**18**，best/anchor均null；不能沿用4000旧源stale16或重置成绩资格。原sampler cursor4500/seed20261002、完整config、parent limit4500/stopped_at4500、source legacy[3750,4000]与本轮legacy[4250,4500]分别保留。
- 原保存的Python/NumPy/CPU/CUDA RNG完整检查和复制，不初始化CUDA，不在source-only reader里清除CUDA RNG作隐式CPU迁移。
- 两个原源model均匹配旧DEV原始模型digest；control digest **64285036d80e7a8f1c178b2a463fd7dcffb1aec32d54a10b042b197476516a6d**。这是身份核对，不是新DEV评分。
- actual receipt与completion嵌入receipt分别验自身seal，再完整对称、类型敏感比较，不单边去seal、不放宽梯度阈值。检查完整实际源/代码/审批绑定与原匹配launch-exit，source-only packet全位/类型一致，无源或内部storage别名。

旧PowerShell完成复核的嵌入receipt曾将LR scale `1.0`规范化为整数 `1`；原PT/实际receipt/completion均为float。旧复核文件仅按已有固定文件SHA保留其退出证据，不把规范化的嵌入副本替代真正类型权威，不回写旧复核，也不放宽新比较。

测试覆盖合法入口、所有停止/游标/模式/角色/类型、完整Adam flags和映射、非有限/负v/缺字段、参数位变化、别名、文件变异在反序列化前拒绝、旧单边seal回归和浮点/整数不能互换。禁止API测试在构造或计算前拦截；实际执行的活Module/前向/autograd engine/Adam构造或step/CUDA初始化/学生更新/真实音频输入全部为0。CPU/Python/NumPy RNG不变，所有15个源接口固定文件在入口前后及单位结束时SHA一致。

## 封存

- 新工具SHA：**e36f6045ea3eec18bc28dd89c5b521bbc887a55a900425b7b55d3dec4ebf2ea9**。
- 新测试SHA：**26d7bfccba88c218d3bf83bce2e3c08bb83d53ef9c5d4687ec5113b3a3a4c959**。
- scope SHA：**846771f0d9ed4ead5bc320ffd037556be7f7c01286ee7c7e50b7871967b6d9fb**。
- `results/mel_ema_source_contract_monitor_20261004/unit_tests_attempt01.log` SHA：**019190b286dc35cf45094c628d905aa093bc0104cb80d360d36f8bf8a22d39c0**。
- 同新monitor `unit_gate.json` content seal：**4e241d682e3dceb573db4831af9464760155f13730d103d5d2e79546f549e3f9**。新工具/测试/scope/log/gate不回写，不重复70单位；后续新集成可调用这一已封存接口，但不能把本gate当完整训练审批。

05:58:40.7056167Z动态CIM无仓库Python任务，磁盘**24.62729263305664GiB**。本轮未启动新GPU、未修改旧脚本/协议/审批/输入锁/cache/学生/板端或用户其他改动。

## 尚未证明及下一步

已完整读193/194/143/147/150及136/149/125/126/159/170/176和实际源审批；frontend/损失/采样cache的其余完整文件阅读与新训练使用时的全部实际绑定仍需完成，不能声称全依赖已就绪。

继续实现新的完整raw+shadow恢复容器与事务：单真实raw model、单原Adam、独立203影子，不套194双真实训练臂、不把EMA权重塞回旧Adam。全模式/已有grad/完整Adam/RNG/原采样日程、原累计best-stale-anchor和单独shadow递推计数应共同回滚。source-only packet目前既没有活Module恢复，也没有采样器/日程执行、完整磁盘恢复或失败注入事务证据。

新的完整集成单位、真实零更新审计、最多3唯一4501..4503 CPU-CUDA下一输入机制、原194首步raw位一致、影子非零/noalias、每个完整容器磁盘位重放、含shadow计数和CUDARNG全事务回滚、跨设备PCMmetadata仍PENDING。完整训练工具/协议/审批/封存trainingplan/worker不存在；不提前封存缺工具计划、不启动GPU。

报告102的500共同更新4501..5000/4750和5000/硬5000、唯一EMA评估变量、原图/精度/损失/LR1/kill32不变。202已成功归档不重跑；197失败、199.verify1、201.prepare1永久保留。仍无新的音质结论，人工听审与独立真实验收PENDING，release NONE。20分钟同聊巡检ACTIVE不暂停，健康无新可行动结果安静。
