# 配对教师标签完成与 CPU 训练机制验证

日期：2026-10-02（Asia/Shanghai）。承接[报告 24](24_teacher_completion_and_matched_controls.md)。
**安全准备已完成，正式学生训练仍关闭；没有新音质或上板验收结论。**

## 标签与完成证据

原 Mel-Band 275 首仍为 `complete`，动态 worker PID 11500 不存在，本次日志以
`TEACHER_BULK COMPLETE verified=275/275` 结束，stderr 为空。沿用报告 24 保存的
完整逐首核验证据；没有重复扫描、推理或改动原批标签。

匹配同一原始 FLOAT32 PCM 的 HTDemucs 固定 24 首在 04:31:38 完成；本次
worker PID 48572、venv launcher 59436 和独立 helper 55484 均已退出。
回执对应的 stdout 为 `MATCHED_HT COMPLETE verified=24/24`，stderr 为空。
没有换歌曲、复用旧 PCM16 小试标签或重新运行教师推理。

本轮单独执行 script139 **verify**，进程退出码 0：24 首计划/entry/原 PCM/
文件哈希、FLOAT32 重载及伴奏逐样本等于输入减人声均通过。
复核是只读诊断，**不能反推已经退出的原推理进程退出码为 0**。

```text
results/teacher_pairs_htdemucs_20261002/post_complete_verify_20261002.log
SHA256: 451fdcacfd4ffb39b0bbc5079fce5cb8de28cd526455a2f50953408c4b5f3069
HTDemucs plan.json SHA256:
9e507510d0b8f15577f479513bf8f5b7e07beb3877cdda8ae4d657098f32ac0f
```

## 原退出码未知，以及启动器修复的边界

原 `detached_exit_20261001_202633_6229756.json` 的 `exit_code` 是 **null**。
该原始回执保留不动，不能将 null 当作 0，也不能补写不存在的历史退出证据。
因此原要求的数值退出码门槛尚未满足，后续须明确决定是否接受独立完整复核作为
替代完成证据，或另行规定补证方式；不为补回执而自动重复 GPU 分轨。

在全部 worker 退出后，新增 script141 的进程句柄保留/退出码采集函数和
Windows PowerShell 5.1 CPU-only 探针。对明确 `exit 0`、`exit 7` 的隐藏子进程：

| 子进程预定退出码 | 不预先保留句柄 | 预先保留句柄 |
|---|---|---|
| 0 | null | 0 |
| 7 | null | 7 |

同样的 `Start-Process` 重定向路径复现了退出码采集缺陷；这解释了为何旧回执
不能证明数值成功，**不是 GPU 或教师推理失败的证据**。
script140 现于启动后立即保留句柄，等待后拒绝缺失退出码，并记录采集模块哈希。
修复只对将来启动生效；尚未再跑完整 GPU 队列验证新回执。PowerShell Core 的
启动环境保护继续保留，不改驱动、注册表或分轨配方。

```text
results/process_exit_capture_probe_20261002/probe.json
powershell_version: 5.1.26100.9444
zero_and_nonzero_captured: true
missing_handle_rejected: true
teacher_inference: false
```

## 技术审查、快照与待试听清单

新增 script142：对固定 24 首的两套标签检查格式/有限性/长度/技术指标和哈希，
封存两份 CPU 审查用途快照。24 对均技术通过；HTDemucs 13 首、Mel-Band 14 首
有试听提示。这些提示不是人声残留评分，不等于质量失败或已审核。

两份快照的歌曲顺序、输入、采样参数和增益规则一致，只有伪标签目标不同。
绑定 60 个计划/协议/原数据锁/冻结模型/代码/entry/复核日志/原退出回执文件；
末次重新检查全部绑定和两份快照哈希通过。没有调整标签增益、削波或改变审核状态。

```text
results/paired_distillation_prepare_20261002/paired_bundle.json
SHA256: 713e49996cf8645b2ef8a963cdbea032520ffe80a991008bae9cd59cca001d8e
results/paired_distillation_prepare_20261002/listening_pending.json
SHA256: 1ac2e5ce93acc9e01fbdd62cc631187b5f3250fe6ab8b41338eb74f3a801f3b9
htdemucs_snapshot.json SHA256:
cfff1958067412c701ce6aced152a6c740a44b64240cdf0a3a8777833d329c70
kim_melband_snapshot.json SHA256:
55f6c74860f25aba488918eec9947f0ed5bbad7679a9b3c861638457b5234ae5
```

当前人工审核 **0/24**、24 首使用资格均 pending、`training_authorized=false`。
清单记录同样本窗口、原曲路径和两教师输出目录；先按固定 24 首试听，随后查看
全库提示项。每首建议 10%/50%/90% 三个 18 s 窗口，同时听原曲/两路人声/两路
伴奏，检查人声残留、乐器误删、瞬态和段间异常。播放应采用各路共用的安全增益，
不能逐路响度归一化后比较。未生成新试听音频，未上传音乐。

## 同输入、共享预算与恢复原型

新增 script143 和 15 项配对机制测试；目前是**硬限最多 3 个逻辑步骤的 CPU
机制入口**，不是已获准的正式 CUDA 训练器。默认训练用途、CUDA 实例化和
`--formal-train` 均拒绝；未审核快照仍不能用于正式训练导入。

- 两臂同一冻结 bott2 初始化、128 带 legacy_log/linear/LF44 和产品重建损失。
- 每批 6 段：锁定 TRAIN 的 MUSDB/MIR-1K/器乐各一，另加 3 段相同伪标签输入。
  不读取 OnAir 代理或开发/验收曲目做训练，不按教师输出给两臂生成不同混音。
- 256 计分帧、96 预热帧和原裁剪边界不变；microbatch=1，两臂只有伪目标不同。
- 共享逻辑步数、输入曝光、LR 和停止预算；第二臂失败时整体恢复两臂模型、Adam、
  RNG、采样游标与调度状态，不能留下半更新配对。
- 实现原协议的 warm-up/余弦、共同早停和无合格候选 `NONE` 状态机；正式 10,000
  步预算仅作为合同测试，未执行。script119 实际开发评估尚未接入/执行；状态机
  测试用的合格分数不能算作真实模型门槛通过。
- 磁盘检查点保存两臂 model/Adam、Python/NumPy/CPU RNG、输入游标和共同预算；
  保存后校验哈希，重载拒绝绑定、后端、预算或两臂曝光不同。
- CUDA RNG 捕获/恢复接口已实现，并用模拟测试检查；**未实测 CUDA 恢复**，
  不能由 CPU 重放成功推断 GPU 位精确恢复。

## 真实 CPU 机制检查与回归

使用当前固定快照、真实 TRAIN 音频和冻结学生，在 CPU 上完成两臂各 3 个逻辑
更新；从磁盘第 1 步检查点新建实例恢复，再重放第 2、3 步。
输入抽样及整个 model/Adam/RNG/游标/预算状态逐项完全相同。
每臂实际优化器执行次数是 **3 次参考运行 + 2 次恢复重放**；不是 5 个训练逻辑
步骤，更不是收敛或教师优劣实验。两臂损失目标本来不同，不据这些损失比较音质。

```text
PAIRED_CPU_MECHANISM PASS updates_per_arm=3 disk_resume=identical;
selection=NONE; no formal training
results/paired_distillation_cpu_20261002/cpu_mechanism.json
SHA256: 3dcacf0abfb87b291d081486fd0aaeb90df622fd4df84109f35104d5b9ca3086
```

第 0～3 步检查点均以 `NONRELEASE_` 命名，约 66 MB，只作机制证据。
`checkpoint_selected=NONE`、`formal_student_training=false`、`deployment=false`、
`cuda_used=false`。没有以技术检查代替人工听审，也没有输出可发布新模型。

最终 CPU 回归通过：配对机制 15、数据入口 16、技术审查 6、配对标签 7、批量
合同 10、pilot 12、原训练机制 10，合计 **76 项**。Python/PowerShell 语法和
`git diff --check` 通过；6 个冻结模型/旧新协议/原数据锁哈希未变。
末次检查无活跃匹配 worker 或 CPU smoke，D 盘余量 33,524,826,112 B。

## 下一步与停止条件

本阶段安全准备结束，按本聊天预授权停止定时检查，避免重复处理已有输出。
OpenAI Docs 技能仅用于核对并管理聊天定时任务，依据[官方定时任务说明](https://learn.chatgpt.com/docs/automations)，
不改变训练协议。原报告中的“队列进行中/定时检查继续”是历史阶段记录，当前状态
以本报告为准。

正式训练前仍须：

1. 完成配对清单的逐首人工听审，记录接受/排除及原因；不能用之前几首试听的
   良好反馈自动批准全部 24 或 275 首。另确认音乐、权重和数据的使用资格。
2. 明确原未知退出码的证据处理，不自动放宽门槛或伪造历史回执。
3. 补齐独立真实分轨开发/验收角色；至少 10 首独立歌曲、目标 >=5 位艺人的
   新盲测条件仍未满足，MP3 教师伪标签不能当最终真值。
4. 满足资格并得到正式训练授权后，再实现独立批准导入与 CUDA 训练/恢复实测，
   接入 script119 实际门槛，仍按两臂共享预算运行；无合格 checkpoint 继续 NONE。

冻结学生、旧缓存、板端镜像、SD 和 `vitis_journal.py` 未改；保留其他未提交改动。
没有 Git 提交/推送、部署、音频上传、驱动/注册表修改、新聊天或子代理。
