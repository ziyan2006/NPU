# 分轨独立后台续跑与定时检查

日期：2026-10-02。仅改变启动方式，不改分轨数值配方、源文件、标签或学生模型。

## 中断线索与边界

第二次中断前最后状态更新为北京时间 02:23:25；当前 Codex 主进程创建于
02:23:31，同期有 `nvlddmkm` 153 事件。结合用户说明刚重启了 Codex，应用重启
回收工具进程树是重要候选原因。只有时间关联，**不能直接确定为显卡驱动故障**，
也没有通过人为再次重启应用复现；没有安装驱动、修改 TDR 注册表或关闭保护机制。

已保存 100 首，均有完成记录、无半成品目录。续跑仍由原 script134 全量重读
已保存歌曲，检查封存输入 PCM、分轨哈希及 `accompaniment = mix - vocals`。
封存计划的 41 个输入/代码绑定复核通过。复核阶段的进度暂为零，不代表旧结果丢失。

本轮复核已完成，随后 `song_0100`、`song_0101`、`song_0102` 三首完整生成并回读
通过；北京时间 02:45:51 检查为 **103/275**、正在 `song_0103`，stderr 为空、
实际进程存活。首首旧回执的哈希保持不变。这仍只是技术完成进度，不是整批完成、
人工音质审核或训练资格批准。

## 新入口

新增 `scripts/137_start_teacher_library_detached.ps1`：通过本机 Windows WMI
服务创建隐藏的 PowerShell 启动器，再调用原 script135/script134。
启动使用 `CREATE_BREAKAWAY_FROM_JOB`、Unicode 环境和隐藏的新控制台；本机
Windows PowerShell 5.1 的无控制台探针不能正常执行，故未用于正式启动。
没有安装 Windows 定时任务、开机启动项或自动反复重启逻辑。
这种 WMI 启动/标志的定义见 [Microsoft Win32_Process.Create](https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/create-method-in-class-win32-process)
和 [Win32_ProcessStartup](https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-processstartup)。

实际独立探针通过，启动器父进程为 `WmiPrvSE.exe`、`in_job_object=false`；
实际 venv 入口 PID 10776 也不在 Job Object 中。真正的 Python 子进程 PID 11500
在 Job Object 中：这与 [CPython 3.13 venv 启动器](https://github.com/python/cpython/blob/3.13/PC/venvlauncher.c)
自己创建、分配子进程并等待的实现一致。**不能要求所有 Python 子进程都无 Job，
也不能把“有 Job”直接等同于属于 Codex。** 独立入口和实际父子关系已检查，
但尚未实际重启 Codex 验证存活，不承诺能抵御关机、注销或 GPU 故障。

本次启动记录（忽略目录，不提交）：

```text
results/teacher_library_melband_20261002/detached_launch_20261001_183837_3002208.json  # CPU 探针
results/teacher_library_melband_20261002/detached_launch_20261001_183904_8060979.json  # 真正启动
results/teacher_library_melband_20261002/worker_20261001_183907_5697326.stdout.log
results/teacher_library_melband_20261002/worker_20261001_183907_5697326.stderr.log
```

启动器以 CreateNew 保存独立回执，不覆盖日志；真实启动会拒绝已有同批 worker
或 launcher；拒绝 results 以外的输出。独立探针、直接会话启动被拒绝、重复启动
被拒绝、输出范围拒绝与 PowerShell 语法检查已执行。原推理脚本没有修改。

## 定时检查

按 [OpenAI Docs 定时任务说明](https://learn.chatgpt.com/docs/automations)，继续使用
当前聊天内现有的每 10 分钟检查，而非新建另一个任务。提示已补充从最新非探针
启动回执选择 worker 日志，结合动态 PID 判断活性。正常时保持安静，完成后继续
标签核验及既定的配对训练准备；异常时先诊断，不盲目重启。

电脑需保持开机；检查及完成后的 Codex 续做仍需应用运行。后台分轨是单独的进程，
检查不是另一个推理任务。正式学生训练、标签人工审核、板端部署门槛不改变。

手工入口（运行前先确认无同批任务；探针不碰 GPU）：

```powershell
& scripts/137_start_teacher_library_detached.ps1 -ProbeOnly
# 阅读其 Receipt 并确认 probe_passed 后：
& scripts/137_start_teacher_library_detached.ps1
```
