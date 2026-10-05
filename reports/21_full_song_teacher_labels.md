# 私人曲库全量教师分轨：清单与启动记录

日期：2026-10-02。用户对提供的教师试听反馈“效果很好”，随后明确要求“开始”
全量分轨。授权的是**本地完整歌曲标签生成**，不是学生训练、部署、SD 写入或音频上传。
试听反馈不等于逐首审核完整曲库。试验教师固定为报告 20 的 Kim Mel-Band RoFormer。

## 封存的批次

| 项目 | 数量 |
|---|---:|
| 原始 MP3 文件 | 329 |
| 已知开发/留出/回归或其版本：隔离，不解码 | 27 文件 |
| 同名/版本组不同完整 PCM：隔离待审 | 4 组，9 文件 |
| 完整 PCM 相同：合并重复副本 | 18 文件 |
| 本轮待生成完整歌曲 | **275** |
| 去重后音乐时长 | **66,816.705 s / 18.560 h** |

数量关系：329 − 27 − 9 − 18 = 275。不能把 329 个文件说成 329 首独立训练歌。
记录 136 个保护 ID，来自原数据锁 DEVELOPMENT/回归/盲测清单、新 Cambridge
开发及验收候选、历史顶层报告中的所有显式 `holdout_tracks`，保留旧 `chills` 排除。
这些来源文件都有哈希绑定，不只检查最新模型报告的一个 holdout 列表。

先通过规范标题/版本别名隔离，再核对保留曲目的完整解码指纹；保护音频没有送入
解码器或教师。保护曲的字节相同但改名副本也排除。完整 FLOAT32 PCM 一致的保留
副本只生成一次，保留全部原路径和原文件哈希。标题或版本别名相同而 PCM 不同的组
不会自动合并、择优或算成独立歌曲：

- Tinlicker / Julia Church — Glasshouse，普通版与 Extended Version。
- Jerro — Forever，普通版与 Extended Mix。
- Sultan + Shepard / Nathan Nicholson — Under The Surface，普通版与 Extended Mix。
- Dimension — DJ Turn It Up，两份普通版文件和 Ben Nicky Remix。

这些可在后续审核中明确选择一个来源；本轮不为凑数量自动放行。
没有发现解码失败或超过 30 分钟上限的曲目。
标题和精确 PCM 指纹仍不能保证检出不同编码、剪辑、变速或未知改名的所有近重复；
也不保证歌手隔离、教师预训练独立性或补齐新电子乐盲测。

协议见 [teacher_bulk_protocol_20261002.json](../docs/teacher_bulk_protocol_20261002.json)。
完整私人清单及隔离原因位于忽略目录：

```text
results/teacher_library_melband_20261002/plan.json
```

计划文件 SHA-256：

```text
aa351fbcb9ac489c6ea93fa436a8842e580390141de766f539dca4eabf6d41f5
```

## 标签、预算和质量边界

每首取完整第一音频流，统一解码为 44.1 kHz 立体声 FLOAT32，不做幅度归一化。
8 s 教师窗口和 overlap=2 延用试验配置；保存**整首人声估计**及
**完整解码混音 − 人声估计**。不裁成 45 s、不各自峰值归一化、不压成旧频带掩码。
标签可超过满幅；这不是播放文件削波的理由，也不能因此裁剪训练目标。

每首一个 `song_NNNN/`，内含 `vocals.wav`、`accompaniment.wav` 和封存 `entry.json`。
两个 WAV 使用 FLOAT32。每首记录原 MP3 哈希、完整解码 PCM 指纹、样本数和 decoder
版本；**不额外保存整首混音 WAV**，避免再占约 22 GiB。未来训练须用绑定的原 MP3
重新解码并核对 PCM，不能使用播放器输出或未经核对的其他版本混音。

两路标签理论 PCM 总量约 **47.146 GB / 43.91 GiB**，加 10% 余量的预算
**51.860 GB / 48.30 GiB**，另保留 **12 GiB** 最低空闲空间；准备时 D 盘空闲
84.656 GB / 78.846 GiB，预算通过。原 MP3、旧教师缓存和试验输出保留。
运行前以及逐首生成前重新检查空闲空间；不自动清理历史文件。

标签完成要通过哈希、有限值、声道/长度、FLOAT 文件逐样本重载，以及
精确 `accompaniment = input − vocal` 核对。`entry.json` 是逐首完成标志；
没有回执的半成品目录被保留并报错，不会靠覆盖来假装恢复成功。
进程退出时自动释放的原生排他锁防止两个 worker 同写。

每首标为 `pseudo_label_train_candidate`，`training_eligible=false`、
`per_song_listening_review=pending`。技术检查通过不代表人声标签无器乐泄漏。
仍需抽查低人声能量、强合成器、声乐采样及教师异常情况，并版本化后续训练清单。
用户对一组试听满意不自动批准全部歌曲，也不允许把伪标签作最终真实盲测参考。

## 当前执行状态

首首完整歌曲（268.56 s）已实际生成并通过逐首重载：教师推理 **32.013 s**，
RTF **0.11920**，峰值 CUDA allocated **1,742,892,032 B / 1.623 GiB**；
混音重组最大 FLOAT 误差 5.96e-8。先以 `--limit 1` 保存后退出，再启动完整
后台队列恢复；验证第一首的既有回执并跳过，不重新生成它。后台不是学生训练。

计划已封存，实际 worker 进度以本地 `run_status.json` 和日志为准。
本报告不将“启动”写作“275 首已全部完成”。运行状态中 `verified_completed` 是
已生成并逐首核验的数量；`complete` 才表示这份清单全部处理完。

启动/恢复不会自动训练学生。冻结学生、正式训练协议 v1、原数据锁、旧教师试验
协议的 SHA-256 均保持原值。没有修改 SD，也没有提交或推送。
新标签写入 Git 忽略目录，原曲不上传；本地研究使用不认证商业训练或再分发授权。

## 运行和查看

```powershell
$py = 'C:\Users\30519\.workbuddy\binaries\python\envs\stem-npu\Scripts\python.exe'
# 现有 plan 已封存，不重复 prepare、不要覆盖它。
& $py scripts/134_generate_teacher_library.py status
Get-Content -Raw results/teacher_library_melband_20261002/run_status.json
# 只在没有活跃 worker 时使用；隐藏后台进程、新建 stdout/stderr 日志。
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/135_start_teacher_library.ps1
# 同一目录的 run 会重新验证既有输出并跳过，不覆盖半成品。
# 整批完成后进行独立重解码/全标签核验：
& $py scripts/134_generate_teacher_library.py verify
```

后台处理需要电脑开机且不睡眠；没有擅自改变系统电源设置。`run_status.json` 中
记录真实 worker PID，Python venv 启动器的父 PID 可能不同。`worker_*.stdout.log`
记录每首耗时/RTF/显存，stderr 单独保存。读取状态不启动第二个 worker。
以首首推理速度外推，本批仅教师推理约 2.21 小时，另有加载/解码/保存/核验；
初步给出 2–3 小时的粗估，不承诺完成时刻，速度会受散热和其他 GPU 任务影响。

**运行中不要修改绑定的生成/推理脚本、协议、源 MP3 或原数据锁**。
若需修改生成方式，要明确保存旧证据并建立新版本输出；不能改 seal 来绕过绑定检查。
报告和 README 可更新，不影响本批生成配方。

## 本轮验证和后续

新增 10 项批量合同测试通过：保护曲版本隔离、同 PCM 去重、不同版本隔离、传递
别名组、磁盘预算、输出边界、不可覆盖、排他锁/退出后复用、状态非部署授权、标签
精确关系/篡改拒绝。原教师/试听 12 项、训练机制 10 项及资料导入 11 项均通过，
合计 **43 项**；原 198 条数据锁重新核验通过，语法及 `git diff --check` 通过。

完整分轨完成后先汇总并抽查标签，再单独确定旧/新教师的同图配对训练协议和正式
启动条件。正式 v1 当前不含教师损失，不能把它的真值-only 结果直接当本批蒸馏对照。
学生学习收益、量化损失和板端连续音频质量都仍须独立验证。
