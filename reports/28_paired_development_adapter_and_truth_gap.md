# 真实开发评估接入与独立真分轨缺口

日期：2026-10-02（Asia/Shanghai）。承接[用户听审与许可声明](27_paired_teacher_listening_feedback.md)。
本轮完成评估适配器并评估已有 CPU 机制检查点，**没有新增优化器更新、正式训练或部署**。

## 本轮实际补齐什么

新增 `scripts/146_evaluate_paired_development.py`，将报告 25 的配对机制连接到
**原 script119 的真实波形评估、聚合及选择标准**，不是继续使用模拟排名。
没有修改 script119、script143、原数据锁、冻结学生或封存蒸馏协议。

- 从原数据锁只读取 DEVELOPMENT 的 19 首 MUSDB、9 首 MIR-1K、3 首纯器乐。
  不解码训练集、旧最终留出或新验收候选；核对实际选用原音频的哈希、大小及角色。
- 保持 256 计分帧 / 96 预热、LF44、同一前端和原 31 首 / 177 视图。
  逐条原波形指纹、窗口及完整 manifest 与之前保存的开发清单相同，未改门槛。
- 两臂共享评估输入和冻结基线；真实调用 script119 `evaluate`、`aggregate`、`assess`。
  检查人声重建、弱人声、投影残留人声及伴奏保留，纯器乐改善不能单独赢得选择。
- 评估保持模型权重/缓冲区、训练模式、Python/NumPy/CPU RNG；异常也恢复。
  同时检验输入指纹、评分行覆盖与聚合，拒绝删行、改输入或伪造汇总。
- `observe_pair` 在共同验证步数边界才向共享调度器提交两臂结果；第二臂评估失败
  或提交异常整体回滚，不留下第一臂已选、第二臂未评估的状态。

该适配器可以供将来获准的训练入口使用，**不等于正式 CUDA 训练循环已经集成完毕**。
当前 CLI 仅 CPU 只读评估、拒绝 `--formal-train`，不创建 optimizer，不解锁训练。

## 实际 CPU 评估结果

读取报告 25 已保存且哈希绑定的第 0、3 步 `NONRELEASE` 检查点，不重复运行
CPU 更新。第 0 步两臂权重与冻结学生相同；实际评分运行冻结基线及两臂第 3 步，
共 **3 × 177 = 531** 个开发推理视图。新冻结评分与旧记录的汇总指标最大绝对
差为 **2.462533e-6**（不同指标单位不混作一个 dB），输入清单指纹完全相同。

| 既有 3 步机制权重 | 四域平均人声 error-SNR 增益 | 平均残留人声变化 | 原标准合格 |
|---|---:|---:|---|
| HTDemucs 对照臂 | +0.012726 dB | +0.027733 dB | 否 |
| Mel-Band 候选臂 | +0.012146 dB | +0.025523 dB | 否 |

两者都未达到至少 +0.15 dB 的平均人声重建提升，且平均投影残留人声没有下降。
因此选择仍为 **NONE**。这只证明真实评估和拒绝机制能运行；3 次机制更新不是
收敛训练，也不能据此判断哪位教师蒸馏更好，或推翻用户对大教师分轨的听感反馈。
未推进原共享训练步数/早停状态、未改旧检查点或 CPU 机制记录。

```text
results/paired_development_eval_20261002/
  selection_suite.json
  frozen_scores.json
  paired_scores_step_0003.json
  development_evaluation.json

suite inputs SHA256:
e5990cdd2faf71e83469cbef7f94dea6b80d0b376ba49d5b81bca69fe442fb07
development_evaluation.json file SHA256:
c18af6522c9fd81289183e178280b3126e4d976cf309fad298a6816a13565586
```

独立 `--verify` 已通过，校验原依赖、选用源音频、两个 CPU 检查点、评分输出及
不训练/不验收/不部署的状态。已有输出拒绝覆盖，后续只核验，不重复生成。

## 新验收资料：再次查重和可用性检查

旧 6 首 Cambridge 候选的原角色不变：4 训练候选、1 开发候选、**1 验收候选**。
源轨听查仍未完成，合格验收曲目为 0；距至少 10 首的新真实分轨目标，候选层面
至少还差 9 首。那 24 首教师伪标签不是独立真值，不补入验收计数。

沿用下载前封存的 catalogue、原 `assignment.json` 的已知曲目名单及作者固定
提交 `0781bc32bcd246898b5fa1646e8b1828ae8be631` 的分轨元数据，找到以下**资料候选**：

| 原目录曲目 | 目录风格 | 作者元数据线索 | 本次状态 |
|---|---|---|---|
| Carol Dant — Do Not Stand | Electronica | `07_LeadVox.wav` | 未见已知名单名称匹配，未下载/听查 |
| FIN — Echoes | Experimental Electronica | `17_Vox.wav` | 未见已知名单名称匹配，未下载/听查 |
| Karl Hungus — Got Your Love | Trap & Future Bass | Vox/SFX 轨 | 未见已知名单名称匹配，未下载/听查 |

这仍是名称/元数据筛查，不证明新歌、足量活跃人声、典型电子乐覆盖或干净标签。
元数据主要对应完整版，不能直接当节选的 vocal 文件映射。
Georgia Wonder — Siren、Juliet's Rescue — Heartbeats、Tim Taler — Stalker 和
Mu — Too Bright 已在已知名称清单，不能重复算新验收曲目；Nervbloc — Slapback
虽标 `instrumental=no`，本次元数据未明确命名 vocal 原轨，暂不作确认的人声正例。

三个有效 catalogue 下载链接如下：

```text
http://www.multitracks.cambridge-mt.com/CarolDant_DoNotStand.zip
http://www.multitracks.cambridge-mt.com/Fin_Echoes.zip
http://www.multitracks.cambridge-mt.com/KarlHungus_GotYourLove.zip
```

本机只读 HEAD 检查均 301 跳到同路径 HTTPS，正常证书校验下 HTTPS 客户端出现
`SSLV3_ALERT_HANDSHAKE_FAILURE`；PowerShell 的正常重定向路径也遇到握手失败。
**不是已下载的数据损坏或训练失败，具体网络/服务原因尚未确定。**
没有关闭 TLS 检查、修改 TLS/驱动/注册表、绕过限制或下载非授权镜像。
另外探测的两个非 catalogue MTK 路径返回 404，不把它们当有效来源或原链接故障证据。

资料筛查依据（本轮仅读取元数据，没有音乐下载）：

- 原 catalogue 快照 `data/datasets/CambridgeMTK-new-candidates-20261001/catalogue_snapshot.html`，
  SHA256 `9071bda2ee169f3430052914f6bc4b5e6b57352f4c1106a2cedcab50d26328db`。
- [Carol Dant 元数据](https://github.com/SiddGururani/mixing_secrets/blob/0781bc32bcd246898b5fa1646e8b1828ae8be631/Medley_Format/Audio/CarolDant_DoNotStand/CarolDant_DoNotStand_METADATA.yaml)、
  [FIN 元数据](https://github.com/SiddGururani/mixing_secrets/blob/0781bc32bcd246898b5fa1646e8b1828ae8be631/Medley_Format/Audio/Fin_Echoes/Fin_Echoes_METADATA.yaml)、
  [Karl Hungus 元数据](https://github.com/SiddGururani/mixing_secrets/blob/0781bc32bcd246898b5fa1646e8b1828ae8be631/Medley_Format/Audio/KarlHungus_GotYourLove/KarlHungus_GotYourLove_METADATA.yaml)。
  这是作者整理的提供方资料，不冒充本次音频检查。

替代来源已重新核对：[MoisesDB 官方仓库](https://github.com/moises-ai/moises-db)
确实提供音乐真分轨及元数据接口，指向[官方研究页面](https://music.ai/research/)。
官方页面介绍 240 首、47 位艺人，并限定非商业研究用途；**不保证其中有满足本项目
全部条件的 10 首独立电子人声歌曲**。本轮没有取得下载包、提交个人资料/申请或
将用户的私人 MP3 许可扩展成 MoisesDB 许可。

## 测试、保留项与下一步

新增适配器 **14 项**测试通过，包括真实 script119 前向与评分、不创建 optimizer/
查询 CUDA、两臂完整提交/失败回滚、RNG/模式/权重恢复、未配对分数与输入变更
拒绝、CPU 检查点身份和不可覆盖旧结果。
原模型选择 13、配对机制 15、蒸馏数据 16、Cambridge 来源 11 项回归通过，合计
本轮 **69 项独立测试**。语法、文档空白及冻结/原协议/原数据锁绑定检查通过。

下一阶段应先取得并隔离新增真实分轨来源，**下载与角色封存发生在新曲评分前**；
复用现有源轨 QC 流程，对实际文件显式映射，检查共同起点、人声采样/SFX、串音和
源轨用途资格。当前新验收数据未完整，不能用教师标签或改名单/标准来凑数。
用户的当前 24 首听感及私人 MP3 训练许可声明继续沿用，不要求重复试听旧包。

正式启动仍需独立批准导入/角色门槛、HTDemucs 未知历史退出码的完成证据选择、
正式训练授权以及 CUDA 训练/恢复实测。原报告 25 中“script119 尚未接入”的状态
已由本报告的**真实 CPU 评估适配器**取代，不等于其余门槛也已通过。
冻结学生、原协议/数据锁/标签、原 6 首分配、旧 CPU 检查点、板端镜像及
`vitis_journal.py` 均保持不变。没有 SD 写入、音频上传、Git 提交/推送、新聊天、
子代理、定时任务或重复分轨。

```powershell
$taskPython = 'C:\Users\30519\.workbuddy\binaries\python\envs\stem-npu\Scripts\python.exe'
& $taskPython scripts/_test_paired_development.py
& $taskPython scripts/146_evaluate_paired_development.py --verify
# 已保存评估，不再次运行默认生成入口；--formal-train 始终拒绝。
```
