# 24 首配对蒸馏探索：入口验证与启动

日期：2026-10-03（Asia/Shanghai）。依据报告 31 的用户授权和后续“可以并行训练”。
**本地探索训练已经实际开始，不等待 MoisesDB 整包；未替换冻结学生或板端。**

## 本轮范围与实现

- 新协议 `docs/teacher_exploratory_protocol_20261003.json` 和独立批准导入（script149），
  使用原封存、已听审的 24 首配对标签。不改原 `training_eligible=false` 快照。
- script150 独立真实输入训练器：同图、冻结初始化、3 真值 TRAIN + 3 相同伪标签
  输入、同增益/损失/LF44；只改变教师目标。两臂不采用独立停止或不同输入预算。
- 保留原 10000 步余弦配方，本次授权探索阶段限 **1000 个共同逻辑步骤**。
  每 250 步执行旧 31 首/177 视图真值开发评估，保存两臂 model/Adam/模式/
  CPU-CUDA RNG、输入游标、验证/停止状态及绑定。运行日志保留每步输入摘要和损失。
- 第二臂失败整体回滚；恢复时拒绝代码、标签/角色、设备/数值设置和预算改变。
  已写但未生成回执的完整检查点，只在状态逐项相同的重放时复用，不覆盖冲突文件。
- 用户允许共享 GPU；新入口保留单 GPU / 最低 2300 MiB 空闲显存检查，不因
  利用率高单独拒绝。旧 script147 的 CLI 保守策略未改，FP32 精度未降低。
- script151 通过 Windows PowerShell 5.1 / WMI 服务独立启动；拒绝重复 worker，
  保留进程句柄并在退出后记录真实数值退出码，不把创建 PID 当成功完成。

本轮所有检查点均为 `NONRELEASE_PAIRED_EXPLORATION`。旧开发评分不是新增盲测；
即便旧门槛通过，正式 `release_selection=NONE`，不自动进入量化/SD 部署。
HTDemucs 历史退出码仍为 null；独立完整波形核验只作为探索标签可用证据。

## 实测与修复

新探索单元测试 16 项通过；原设备 16、配对机制 15、开发评估 14、数据入口 16、
模型选择 13 项回归通过，共 **90 项 Python 测试**。
Windows 退出码探针重现无保留句柄的 null，保留句柄可正确捕获 0/7；
新版 WMI 探针确认父进程为 WmiPrvSE、没有继承应用 job object，未启动推理。

第一次 CPU 真实输入验证完成三步，但重复载入同一个内存检查点时校验拒绝：
Adam 的 CPU `load_state_dict` 可能复用输入张量，恢复后更新使内存中的旧 Adam
快照也发生变化。磁盘文件未损坏，第一次磁盘重放其实已经一致；不是音乐标签问题。
现已在独立 engine 中复制优化器状态后载入，并新增外部快照不变测试。
旧 script143/147 未改。第一轮 approval/检查点保留为未完成历史尝试，不冒充当前
有效证据；代码修订和资源授权后独立生成 r2 导入与验证，不覆盖旧目录。

以下三项在实际冻结 bott2 图上通过：

1. CPU 真实 TRAIN/配对输入：三步更新、磁盘恢复重放及第二臂异常回滚逐项一致。
2. CUDA 合成输入：同样三步、恢复/回滚一致，峰值 allocated 218,498,048 B（约
   208 MiB）；这是本机制的分配峰值，不是 WDDM 总显存或长训资源预留。
3. CUDA 真实 TRAIN/配对输入：同样三步、恢复/回滚一致。

每个机制检查两臂各做三次参考更新、两次恢复重放；注入失败时另有第一臂一次
随后回滚的更新。它们不是学生质量/收敛实验，也不计作后台正式探索的共同步数。

```text
results/paired_exploratory_import_20261003_r2/approval.json
SHA256 aa2eb50b82635cefb46fd4fc2d3cae611154eb064b26f2568968b756b139b7c6
results/paired_exploratory_cpu_20261003_r2/real_mechanism.json
SHA256 8bff0242a16eb01eda4cc486fd19000f2e8ad224f0f17b81adbd5d43fe680b63
results/paired_device_cuda_20261003_r2/device_mechanism.json
SHA256 9521f5975cead20ee9970c82d71ae1944e934306c7dbd837d898d1d697c7f7df
results/paired_exploratory_cuda_20261003_r2/real_mechanism.json
SHA256 74ba44dfd8d324358db8ef96673294bae29ac7bc6b7c4a9b4a04ada41d91cb1a
```

## 实际运行与下一次读取

启动回执：
`results/paired_exploratory_launch_20261003/detached_launch_20261002_171204_5734319.json`。
独立 helper PID 5380，venv launcher PID 35484；真实 worker PID 在状态文件中动态读取，
启动时为 29824。命令与进程创建时间已核对；两条 Python 进程是 launcher/worker，
不是重复训练。

```text
results/paired_exploration_20261003/run_status.json
results/paired_exploratory_launch_20261003/worker_20261002_171204_5734319.stdout.log
results/paired_exploratory_launch_20261003/worker_20261002_171204_5734319.stderr.log
```

本报告首次记录时，原真值开发基线已建立，实际两臂已提交共同第 1 步，
`phase=training/error=null`、stderr 为空。首步含整曲解码等约 10.07 s，
不能拿冷启动首步保证总耗时；并行应用负载会影响吞吐。两臂目标不同，不能比较
训练损失数值来判定教师/学生音质优劣。首次真实趋势评估在共同第 250 步。

后续先读最新非 probe 启动回执和对应日志，再用状态动态 PID 核对活性；不要只信
`running`，不要因一次步数未变判卡死。失败先保留结果和诊断，不自动重启或删文件。
完整数值状态每 250 步保存，状态日志每 10 步更新；崩溃最多需从最近提交点重放。
1000 步后退出，不自动扩大到 275 首或续跑原 10000 步。

```powershell
$taskPython = 'C:\Users\30519\.workbuddy\binaries\python\envs\stem-npu\Scripts\python.exe'
# 已启动，不再执行 train/launcher 来重复启动。
Get-Content results/paired_exploration_20261003/run_status.json
# 完整探索完成且无 worker 后，再核验：
# & $taskPython scripts/150_train_paired_exploration.py verify
# 只有需要且另行明确指示时，未完成运行才可用 -Resume / --resume；不自动重启。
```

冻结学生、原正式/蒸馏协议、数据锁/旧快照/教师标签、板端产物及 vitis_journal.py
保持不变；保护文件哈希和 `git diff --check` 已核对。未上传音乐、写 SD、提交/推送、
改驱动/注册表、终止用户程序、重启下载、创建定时任务/新聊天或子代理。
