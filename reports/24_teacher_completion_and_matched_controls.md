# 全库教师标签完成、技术审查与同输入对照

日期：2026-10-02（Asia/Shanghai）。本阶段是标签准备，不是学生重训或上板升级。

## 已完成的全库检查

原 Mel-Band 队列在 04:00:38 写入 `complete`，275/275 首；worker PID 11500
及其 venv 父进程已退出，本次 stdout 以 `TEACHER_BULK COMPLETE` 结束，stderr 为空。
04:07 起另行执行了 script134 的 **verify**，退出码 0；不是重复运行教师推理。

逐首检查了封存计划/依赖/权重绑定、原 MP3 完整解码 PCM 指纹、两路 stem 文件
哈希与 FLOAT32 重载，以及伴奏逐样本等于同一输入减人声。最终：

```text
TEACHER_BULK ALL VERIFIED songs=275; per-song listening remains pending
```

本地证据：

```text
results/teacher_library_melband_20261002/post_complete_verify_20261002.log
SHA256: 48fc968b5a9e64d58d3dbf0b2d2a09a4f22c67509b59eff3366d1c3607d525ad
原 plan.json: aa351fbcb9ac489c6ea93fa436a8842e580390141de766f539dca4eabf6d41f5
```

新增 `138_audit_teacher_labels.py` 做有界内存的整首流式技术检查：采样率/声道/
FLOAT subtype/长度、有限性、RMS/DC/峰值/相邻跳变、活跃混音中估计人声很弱的
比例；重新核对 entry/label 哈希，保留完成复核日志绑定。不改变原标签或 entry。

275 首技术检查全通过，161 首附有**人工试听提示**，提示可重叠：

- 148 首伴奏有较多绝对值 >1 的 FLOAT 样本；1 首人声有该提示。
- 24 首在多数活跃窗口中估计人声很弱，可能本来就是器乐。
- 4 首伴奏平均 DC 偏移需要检查。

这些不是分离质量分数或“失败歌曲”统计。FLOAT >1 不等于保存时削波，估计
人声很弱也不能证明教师漏分；试听前须按同一输入/目标统一播放增益，不能分别
响度归一化后评价。波形跳变仅记录指标，不能把电子乐瞬态自动当作拼接错误。

```text
results/teacher_label_audit_20261002/technical_audit.json
SHA256: 2048d4f33ad08824d8722bd4c496d8f38c72567de8e91ef44f9c665b6c8ed171
results/teacher_label_audit_20261002/listening_pending.json
SHA256: a6e422b4075dacbb839c03d1f1653ca0265341fb6d43585317ddf2cee89c6967
```

清单含全部 275 首的输入/输出路径、样本对齐的 10%/50%/90% 位置建议试听窗、
对齐/残留/乐器损伤/瞬态/使用资格问题。**优先固定配对 24 首，再看带提示的其他
歌曲**；没有因为教师结果挑换配对名单。人工审核数仍为 0，rights/listening 均为
pending，`quality_approved=false`、`training_authorized=false`。

## 匹配的 HTDemucs 对照已开始，不等于完成

新增 `139_generate_paired_htdemucs.py`，独立新输出目录，不使用旧频带掩码缓存，
也不复用经过 PCM16 的旧 MP3 小试标签。它读取原来提前封存的 24 首名单，
按同一原 MP3 完整解码，并在推理前核对同一 FLOAT32 PCM 指纹。

新增计划绑定了技术审查、提前封存名单、协议、原教师计划/entry、原正式协议/
数据锁、冻结学生、脚本和现有教师权重/运行时。使用已验证的 script11 HTDemucs
波形推理路径：CUDA float32、shifts=1、overlap=0.25，保留其输入归一化与逆变换，
标签不独立增益/削波；伴奏仍为**原输入减估计人声**。整首重载后发布 completion
entry；已完成项复核再跳过，部分目录保留并拒绝覆盖，原生 worker lock 防并发。

- 固定 24 首合计 5,417.217 s / 1.505 h。
- 两路新增标签含 10% 余量：4,204,627,487 B（约 3.916 GiB）。
- 启动前 D 盘可用约 37.4 GB；要求生成后仍留 12 GiB。
- `models/htdemucs.th` 为 84,141,911 B，SHA256：
  `8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4`，与 pilot 绑定相同。
- 未发现工作区其他教师/训练 Python 进程；GPU 当时约 4.8 GiB 可用。
  WDDM 清单包含桌面程序，不宣称 GPU 完全空闲或未来不会被其他程序使用。
  推理启动另外要求至少 3 GiB GPU 可用，不终止其他程序。

计划与状态：

```text
results/teacher_pairs_htdemucs_20261002/plan.json
SHA256: 9e507510d0b8f15577f479513bf8f5b7e07beb3877cdda8ae4d657098f32ac0f
results/teacher_pairs_htdemucs_20261002/run_status.json
```

04:27 的首轮观察为 2/24 完成，active `song_0028`，worker PID 48572、venv 父进程
59436；前两首 RTF 0.0417/0.0292，stderr 为空。这是启动/初始输出证据，**不是
24 首完成或最终复核证据**。后续需脚本139的独立 verify，不能只看计数。

## 独立启动入口与诊断边界

新增 `140_start_paired_teacher_detached.ps1`，不改 script137/135 或原教师数值配方。
WMI service 创建隐藏 Windows PowerShell helper，验证 parent `WmiPrvSE.exe` 和
helper 不在 job object，启动后写唯一 launch receipt。helper 等待 venv launcher，
另写唯一 exit receipt/退出码。未安装 Windows 定时任务、启动项或自动重试循环。

新入口第一次从 PowerShell 7 调用时，WMI 返回创建 PID，但没有回执/日志/worker。
CPU-only probe 在同一路径复现；Windows PowerShell 5.1 的 CPU probe 正常。
两者导出的模块/运行时环境不同，但尚未定位底层退出异常，**不归因为 GPU 崩溃**。
现在入口明确拒绝 PowerShell Core，在创建前提示使用已验证的 5.1 路径；这是
启动环境保护，不改变驱动、注册表或教师推理精度。未启动过两个推理 worker。

5.1 probe 的 helper PID 52364，parent 9356/WmiPrvSE，job=false，probe_passed。
随后 04:26 的实际启动回执：

```text
results/teacher_pairs_htdemucs_20261002/detached_launch_20261001_202633_6229756.json
worker_20261001_202633_6229756.stdout.log
worker_20261001_202633_6229756.stderr.log
detached_exit_20261001_202633_6229756.json  # 退出后才应出现
```

实际 helper PID 55484、parent 9356/WmiPrvSE、job=false。venv 自己创建子进程 job
仍可能正常，不能据此认定属于 Codex。未进行重启 Codex/关机的存活实测。

使用 Windows PowerShell 5.1：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/140_start_paired_teacher_detached.ps1
```

**已在运行，不要再次执行上面的启动命令。**监测应读取最新非 probe 的 worker
回执、动态 PID 与对应日志；重复启动保护不是授权重复尝试。

## 测试、保留项与下一阶段

新增质量审查 6 项、匹配标签合同 7 项 CPU 测试全部通过；数据入口 16、批量合同
10、pilot 12 项回归通过，共 51 项。Python 语法和新启动器语法通过；5.1 独立
CPU probe 与 Core fail-fast 检查通过。没有把启动器保护的代码存在当作完整
故障恢复验证，也尚未验证新 exit receipt 在推理完成后的退出码。

冻结学生、原正式协议、数据锁、旧教师/批量协议和原 batch plan 哈希均与此前
封存值相同。未触碰旧缓存/板端镜像/SD/`vitis_journal.py`；保留其他 dirty tree，
没有 Git 提交、推送或音频上传。

后续依次：

1. 24 首退出后独立完整复核；仍保留两教师的 listening/rights pending。
2. 实现固定同图/同输入批次、两臂共享曝光预算/停止步数，以及完整 optimizer/RNG/
   输入 cursor 恢复。先单元测试和小规模 CPU 机制验证，不启动正式学生训练。
3. 人工配对试听/使用资格批准、独立真实分轨开发/验收角色满足后，才讨论正式训练。
   新盲测至少 10 首独立歌曲、目标 >=5 位艺人的真值条件仍未满足。

原每 10 分钟聊天检查继续用于该有限准备工作；完成安全准备或确实需要人工决定
时汇总并停止，不无限重复检查已经完成的 275 首。
