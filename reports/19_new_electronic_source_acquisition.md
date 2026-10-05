# 新电子人声资料：下载、去重与隔离审查

日期：2026-10-01。**完成 6 首新歌曲候选的下载和技术检查，未开始正式训练。**
这些是短节选且听查未完成，不是新增 6 首完整训练歌，更不是 6 首已验收盲测。
冻结板端权重、既有数据锁、训练协议 JSON 和 SD 均未修改；没有提交或推送。

## 本轮实际取得什么

公开分轨 ZIP 合计 **406,339,063 B（约 406 MB）**，解压 **708,576,153 B**；
保留 ZIP 和原始文件合计约 1.115 GB，不含很小的下载/QC 回执。
包含 **111 个 24-bit/44.1 kHz WAV**，单声道和立体声混合；6 首节选的最长原轨
时长合计 **209.981 秒（约 3.50 分钟）**。没有仍在运行的下载。

| 曲目 | 下载前预分配用途 | WAV / 人声轨 | 节选最长轨时长 |
|---|---|---:|---:|
| Triviul — Alright? | 训练候选 | 9 / 2 | 32.783 s |
| Triviul — Better? | 训练候选 | 21 / 12 | 28.326 s |
| Triviul — Gimme | 训练候选 | 11 / 8 | 36.324 s |
| Triviul — To Sam Rawfers | 训练候选 | 15 / 11 | 50.526 s |
| Giselle — You | 开发候选 | 40 / 2 | 32.677 s |
| Speak Softly — North Is False | 最终验收候选 | 15 / 6 | 29.345 s |

角色在下载音频、模型评分和训练前写入带内容哈希的 `assignment.json`。
同一批新数据里的艺人不跨角色；**3 位艺人均在旧 MUSDB 出现过**，不能声称新歌手
隔离。歌曲新颖性依据现有 MUSDB 全部 144 个文件、个人曲库 329 个文件名、原数据锁、
回归和本地教师 source ID 清单，以及手工核对的真实标题别名。
导入 WAV 的原文件哈希也未与原数据锁重合；这不是完整声学指纹审计，也不保证
未保存的历史外部资料从未包含这些歌。因此始终叫“新歌曲候选”。

Triviul 为目录中的 Leftfield Pop/Electronica 专辑选曲，Giselle 为
Electronica/Classical Crossover，Speak Softly 为 Atmospheric Electronic Pop。
**目录分类不证明每首节选都是典型 House/DNB**；本批风格分布仍偏跨界电子流行。

## 去重比增加下载量更重要

本次排除 12 个已在 MUSDB 训练/测试或 Cambridge 旧回归中出现的曲目。特别是：

- `AMContra_HeartPeripheral` 已是 MUSDB test，不能补为盲测。
- `MR1011_GirlsUnderGlass` 实际是 Girls Under Glass — We Feel Alright，已是 MUSDB test。
- `MR0907_Punkdisco` 实际是 Punkdisco — Oral Hygiene，已是 MUSDB test。
- `MR0905_ChrisDurban` 实际是 Chris Durban — Celebrate，已是 MUSDB train。

作者的自动元数据对 `MR...` 的 artist/title 拆分不准确；仅比较压缩包名就会漏检。
同曲换成完整包、重混或更多裁剪都不增加独立歌曲数。

此外，Skelpolu 的 Anomalous Weeping、Cold Strive、Entwine、Long Road 和 Galias 的
Horizon 均未找到明确命名的人声原轨，不按电子人声正例下载。尤其是 Galias 的
`instrumental=no` 字段与所列原轨不能共同证明其有人声；这修订报告 11 的候选优先级，
不改写旧下载/实验记录。[研究作者的原始元数据仓库](https://github.com/SiddGururani/mixing_secrets)

## 下载来源与许可边界

Cambridge 主目录本次仍返回 403，但作者公开存档页面中明确列出的
`www.mtkdata.cambridgemusictechnology.co.uk` ZIP 直链返回 200。没有登录、修改权限、
关闭 TLS 检查或绕过主站访问限制。保存的是作者仓库固定提交
`0781bc32bcd246898b5fa1646e8b1828ae8be631` 的页面快照及哈希，而不是猜测文件地址。
确切 URL、大小、角色、排除项见[获取计划](../docs/cambridge_acquisition_plan_20261001.json)。
[公开作者页面存档](https://github.com/SiddGururani/mixing_secrets/blob/0781bc32bcd246898b5fa1646e8b1828ae8be631/page_source.txt)

6 包均保留随包 Readme：教育用途限制及未经版权方明确许可不得商用的边界。这里只作
本地教育/研究资料隔离，**不宣称获得商业训练、量产或音频再分发授权**。
公开仓库只保留获取/映射代码与摘要；音频、完整随包条款、详细 QC 和权重仍在忽略目录。

服务器只提供可用 HTTP，记录的 SHA-256 是本地完整性证据，**不是版权方发布的认证
校验和**，也不能提供 HTTPS 的来源认证保障。完整下载、ZIP CRC、路径/符号链接/
Windows 特殊文件名、文件数量和解压体积检查通过；解压文件逐一哈希，重跑不会覆盖。

MoisesDB 官方入口仍涉及申请流程，没有代用户提交身份或条款；Cadenza validation
整包超过此前约 5 GB 上限，且仅有 4 首 Electronic，不为了凑数下载全部。
MUSDB HQ 仍是原歌曲，不解决新盲测独立性。
[MoisesDB 官方说明](https://github.com/moises-ai/moises-db)、
[Cadenza 官方数据说明](https://cadenzachallenge.org/docs/icassp_2024/data/data_overview)

## 技术检查结果与尚未证明的部分

按**实际节选文件名**逐条映射 vocal，而不是复制完整版元数据。例如 Giselle — You
完整版元数据列出 54–60 号人声/SFXVox，节选只有 `39_LeadVox.wav` 和
`40_BackingVox.wav`；旧编号不能直接用于本包。
LeadVox、BackingVox 和 double 纳入人声，其余原轨各求和一次；不混用 grouped stem
和 raw，不重复加作者元数据中重复出现的文件。
[实际文件映射](../docs/cambridge_stem_map_20261001.json)

全部 111 WAV 可读、数值有限，采样率与通道符合条件；6 首各有非静音的人声及器乐组。
技术参考在共同 sample zero 求和，短尾补零，以混音峰值不超过 0.95 的同一个增益缩放
混音/人声。不能把每条 stem 分别归一化，也不能裁到最短文件导致有效尾部丢失。
这是**原始分轨合成混音**，没有用教师生成伪真值；不是官方母带还原。

本次发现需留意：

- Alright?、Better?、You 原轨长度不等；已按最长轨零补尾，而非沿用共同最短时长。
- Better? 的 DrumLoop 有接近满幅值的样本，需听查；这一标志本身不能证明削波。
- You 的合成混音 RMS 为 -18.22 dBFS，人声为 -42.85 dBFS，约差 24.63 dB。
  它是值得关注的弱人声开发候选；**没有据此调参、改角色或运行模型**。
- Loop/SFX/SynthsAndSFX 中可能有人声采样；名称不能证明标签纯净。
  单声道简单复制不恢复原制作声像；共同起点也仍需核对，而非由脚本自动证明。

人声轨器乐泄漏、伴奏轨残留人声、效果尾部和对齐需要听查。本轮没有声称听过或
完成这类审查。每首 `training_eligible=false`、`acceptance_ready=false`，新歌验收
`blind_status=incomplete`，不会仅因技术检查成功就开训练或降低验收门槛。

## 实现和验证

- `128_acquire_cambridge_candidates.py`：先封存候选分配，再下载、CRC/安全解压和回执；
  `verify` 只核验，不会偷偷下载缺失文件。
- `129_audit_cambridge_candidates.py`：显式 vocal 映射、声道/数值/原文件哈希检查，
  共增益合成与质量旗标；不加载模型、不输出模型分数、不认证泛化。
- `_test_cambridge_candidates.py`：11 项测试通过，覆盖跨角色/同曲泄漏、假盲测状态、
  路径逃逸、Windows 特殊名/大小写碰撞、符号链接、解压预算、条款缺失、重复映射、
  人声效果误入伴奏、共增益/补零和不可静默提升候选。

压缩包和原文件独立重哈希、QC/配方/角色绑定验证通过。Windows 测试夹具最初会由
ZipInfo 自动规范化反斜杠，未真实测试恶意 ZIP；已改为保留原始头，修正后通过。
原数据锁重新核验及既有训练测试结果以本轮最终检查回执为准。

最终复核：原 198 条/1,025 原音频的数据锁验证通过；既有训练机制 10 项和新导入
11 项测试合计 **21 项通过**；语法、JSON 和 `git diff --check` 通过。
冻结权重、协议 JSON、原数据锁 SHA-256 均与报告 18 一致；Git 忽略规则确认
原数据与 QC 回执不入库，历史硬件/软件改动与 `vitis_journal.py` 保留。

本地证据：

```text
data/datasets/CambridgeMTK-new-candidates-20261001/assignment.json
data/datasets/CambridgeMTK-new-candidates-20261001/*_receipt.json
results/cambridge_new_sources_20261001/source_qc.json
```

复现只需 `requests`、NumPy、soundfile 和已存在的原数据锁/曲目清单：

```powershell
$py = 'C:\Users\30519\.workbuddy\binaries\python\envs\stem-npu\Scripts\python.exe'
& $py scripts/_test_cambridge_candidates.py
# 首次执行 prepare 使用新目录；已有分配绝不覆盖。
& $py scripts/128_acquire_cambridge_candidates.py prepare
& $py scripts/128_acquire_cambridge_candidates.py download
& $py scripts/128_acquire_cambridge_candidates.py verify
& $py scripts/129_audit_cambridge_candidates.py
& $py scripts/129_audit_cambridge_candidates.py --verify
```

## 下一步

先完成源轨听查并补入更长、更多艺人的电子人声真分轨，版本化审核后才进入训练清单。
当前只有 1 首验收候选，离预声明的至少 10 首目标还差 9 首；已有 4 首训练候选也只有
同一艺人的约 148 秒节选，不能解决风格覆盖。最终数量以审查后的独立歌曲为准。
原 198 条数据锁不修改，原 145 段仍只作回归；正式 10,000 步配对训练暂未启动。
