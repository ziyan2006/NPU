# 探索蒸馏的评分复核与学生试听准备

阶段记录：2026-10-03（北京时间）。本工作不改变正在运行的训练及其绑定。

## 当前结果与边界

01:45 检查时，两臂共同训练已超过 210/1000 步，实际 worker PID 29824 的命令、
父进程和启动时间与本次 WMI 回执相符；状态仍为 training，error=null，对应 stderr
为空。尚无第 250 步完整评分/检查点组合，不能据此声称学生音质已改善。
检查点保存和真值开发评估仍每 250 步一次；不重复启动或扩充训练预算。

新增 `scripts/152_review_paired_exploration.py` 为独立只读工具：

- 仅读取完整的 checkpoint receipt、哈希绑定的模型状态和同一步开发评分。
  评分先落盘、完整状态尚未提交时，仅记为 pending，不产生质量结论。
- 核对原 31 首/177 视图清单、逐视图描述、模型权重摘要、原冻结基线和原选择策略。
  从逐曲指标重新聚合并重算 assessment，不只相信已写好的均值或 eligible。
- 展示各域人声重建、人声残留投影、伴奏保留及纯器乐误删变化；
  两臂教师目标不同，不能用不同目标的训练 loss 大小排名音质。
- 不连接活模型、不构造优化器、不初始化 CUDA、不改停止步数或 schedule、
  不将旧开发 eligible 当盲测通过，不选发布模型，不替换板端。

## 学生试听输出

沿用训练开始前 script148 已封存的三个 DEVELOPMENT 来源，不根据新学生分数挑歌。
各取固定首 min(20 秒, 原长度)，使用同一 FLOAT32 输入、同一参考与相同 LF44 路径。

| 来源 | 实际试听长度 | 注意事项 |
| --- | --- | --- |
| MUSDB / A Classic Education - NightOwl.stem | 6.8 秒 | 原始资料本身是节选 |
| MIR-1K / abjones_2 | 20 秒 | 原配方拼接卡拉 OK 小片段，拼接点单独注明 |
| instrumental / Beneath_v1 | 20 秒 | 检查误删乐器，不作为人声削弱样本 |

每个来源导出 9 路：原混音、参考人声/伴奏、冻结学生人声/伴奏，以及两臂新学生
各自的人声/伴奏。若某一路幅度较大，对全部九路共同衰减到峰值不超过 0.95；
不单独归一化学生、不裁波。保存 FLOAT WAV、输入/输出哈希和模型状态摘要。

这是 CPU 浮点试听，不是板端整数前端录音或卡顿改善证明，也不是新增盲测。
已有冻结学生、标签、数据锁、旧协议与板端产物仍不变。

## 使用方式（无需再次启动训练）

```powershell
$taskPython = 'C:\Users\30519\.workbuddy\binaries\python\envs\stem-npu\Scripts\python.exe'
# 对已完整提交的阶段进行复核；尚无完整组合时只显示 PENDING，不创建输出。
& $taskPython scripts/152_review_paired_exploration.py summary
# 首次 250 步完全提交后生成试听；500/750/1000 可指定对应步数。
& $taskPython scripts/152_review_paired_exploration.py listen --step 250
# 核验已生成结果（路径与默认对应）
& $taskPython scripts/152_review_paired_exploration.py verify --out results/paired_exploration_review_20261003/listen_step_0250
```

默认各阶段写入新的 `results/paired_exploration_review_20261003/summary_step_XXXX`
或 `listen_step_XXXX`。现有输出一律不覆盖；如果已有同一阶段结果，先 verify。
磁盘空间须保留正在运行的训练所需 12 GiB 余量。CPU 使用 2 个线程，不占显卡。

## 验证记录

- 新增 13 项测试：半提交、错步、缺臂、模型哈希变化、篡改汇总、非有限指标、
  原策略重算、同输入九路对齐/共同音量、真实前端 CPU 推理、模式/RNG/权重恢复、
  异常恢复、无 CUDA/优化器、证据不覆盖和磁盘保留。
- 联合原开发评估、全源诊断、探索训练和设备机制测试，合计 68 项通过。
- 对实际三个来源完整 WAV 哈希与采样格式预检通过；只解码固定试听前缀，
  实际尺寸分别为 [2,299880]、[2,882000]、[2,882000]，CUDA 未初始化。
- 实际 summary 检查正确报告 pending，没有伪造第 250 步结果，也未创建试听音频。

下一步：等待正在运行的训练提交首次完整阶段，再复核指标并导出同音量学生试听。
如指标与听感冲突，保留每域差异而不把单一均值视为升级许可。1000 步后按原
批准预算退出，是否继续训练另行决定。未写 SD、上传音乐、提交/推送或新建定时任务。
