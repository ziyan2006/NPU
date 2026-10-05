# 伴奏系数项消融：500 步对照已独立启动

2026-10-03 17:44（Asia/Shanghai），依据报告 60/61 和持续优化授权启动。它是 NONRELEASE 探索，不是发布，也不是已训练完成。

## 固定实验身份

运行目录 `results/mel_component_ablation_20261003`；新工具 177 导入、178 训练、179 Windows PowerShell 5.1/WMI 独立启动、180 独立复核，损失核 176。审批 `results/mel_component_ablation_import_20261003/approval.json` SHA256 `8a6c320065c6c6057e57d6fe636f602114f71809faf005e0c86a6465c241d027`。代码、测试、协议、审批与全部绑定已封存，不再修改。

固定 500 共同更新 3001..3500，3250/3500 评分并提交完整恢复状态；3000 是起点基线，不是新增更新。两臂均从保护轮完整 3000 的权重 4 候选 ARMS[1] 分叉，同 model/Adam/模式/全部 RNG/游标/日程与源停止证据。源停止 3000、legacy=[2750,3000] 保留；新阶段活跃 stop=None，不重置旧 LR/best/stale/patience_anchor，旧早停另记 legacy，新 3500 硬终止。

六槽标签全同 Mel/真值，基础权重 `[1,1,4,1,1,1]`、分母 6；cv² 系数均 .2。控制保留 .2*(ca-1)²，候选系数为 0。内部兼容键对应 `aux_full_control` / `aux_residual_only`，不是 HT 教师比较。原图、LF44、24 首、3 真值/3 伪标签、活动门槛、输入增益顺序、严格 FP32 与推理计算不变。去掉此项可能损伤有歌声歌曲中的伴奏，器乐权重 4 不保证补偿。

## 启动前证据

30 项新分叉单元通过；176 的 12 项合成核单元已有证据。固定真实 TRAIN counter3000 零更新审计退出 0：控制旧 172 的损失及参数梯度逐位一致；候选梯度差与移除的伴奏项容差一致，max error 8.94e-7，移除项梯度 L2=.1232、贡献=.005553，两个活动槽，权重与模式未变。单批机制不是音质或全库因果证明。

CPU/CUDA 各三个唯一真实输入 3001..3003：完整源状态与日程迁移、源 RNG、磁盘逐位重放、保存无别名、第二臂异常含 CUDA RNG 整组事务回滚均通过；两臂均实际非零残留辅助/器乐加权，控制伴奏项非零、候选为零。独立 verify_proof 与跨设备输入 PCM/元数据核对退出 0。CPU 是显式机制迁移，不冒充跨设备数值 resume；回放不计正式 500 步。

CPU proof SHA256 `4b5f4919f40de1e5d48c2d8a91b533b207d9a0dff0826f67c0d5fda48e0b80ba`；CUDA proof SHA256 `8e4d1d0a44b28a9048e113f84da03f8d676eed674d2a61057291c11922d14618`。审计 SHA256 `484f84b741792509062cb3ade91a223e5d7674f1b54186c0fe2299843a8b4e12`。独立准备核验及真实前台退出码在 `results/mel_component_ablation_monitor_20261003/preparation_review.json`。

启动前磁盘空闲 27,946,430,464 bytes（约 26.0 GiB），RTX4070 Laptop 空闲显存 2490 MiB，无重复活跃训练/诊断。共享 GPU 不终止用户应用。

## 实际启动与初始活性

回执 `results/mel_component_ablation_launch_20261003/detached_launch_20261003_094454_7514811.json`，helper65464 的父进程为 WmiPrvSE，in_job_object=false。shim58036 / 实际 worker23356 是父子链，不是双训练。实际 worker 创建于 17:44:57，17:45:18 CPU=31.234375 秒，stdout 写入 TRAIN_AUDIT VERIFIED，stderr 空；此时尚无 run_status，仍处初始化，不是训练完成或卡死。以后从状态/回执动态核对 PID、创建时间、命令行及 CPU/文件变化，不固定依赖上述 PID。

日志仅使用本回执的 Stdout/Stderr。对应 detached_exit 尚未产生；只有相符真实 exit0、终态3500/additional500、无 worker 后，才作正常完成及独立训练/评分复核。

独立监测目录 `results/mel_component_ablation_monitor_20261003` 保存已分析完整阶段、历史最佳与通知，不写活训练输出。巡检保持当前聊天每20分钟 ACTIVE，正常无可行动变化时保持安静。比较冻结、3000 起点、并行控制的各域残留/重建/伴奏/逐曲器乐误删；不按训练 loss 排名，不把旧开发当新盲测。只导出原预选三源同输入同音量，每新增完整阶段一次，CPU 开发评分期间不导出。

旧 3000 的33个试听 WAV 已存在，本轮没有重复导出。若仅增加伴奏损傷或弱域仍无改善，优先检查 LF44/上下文/推理重建，不机械删更多保护或加权。保留历史实验、冻结模型、板端镜像及所有无关未提交修改；不提交推送、上传音乐、写 SD、部署或修改推理精度。所有人工听审/独立真实验收仍未自动满足。

## 17:50 后追加核验

CPU 开发基线已完成并进入 CUDA training。180 summary3000 的前台 session71596 真实退出0，原31首/177视图、哈希、Adam暴露、逐曲聚合/原政策、起点候选及两臂身份一致性和停止证据通过。summary SHA256 `f939bd2b361606dda277d26d955bdc70f65191fb1ab1e506fb14126fd3d67be6`；独立monitor已记 analyzed_steps=[3000]、pending=[]，这是旧起点成绩，不计新增质量评分，双方 eligible=false。

17:50 的前20个成功共同更新3001..3020平均7.888秒（3.558..11.039秒，共享GPU波动），不是20个评分阶段或完整恢复提交。nvidia-smi 列出实际worker23356，单进程显存因WDDM为N/A，不能把整卡占用全部归因于本训练。进程CPU已从31.234增到717.781秒，stderr空；仍无detached_exit，训练未结束。下一新增完整评分3250；没有新的学生音质结论，不重复基线试听。
