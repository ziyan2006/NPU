# Mel源系数辅助强度500步对照：实际启动

2026-10-03 12:22（Asia/Shanghai）。NONRELEASE；启动不等于完成，也不是新音质成绩。

## 封存与验证

报告46预选的.02/.2强度对照工具165–168及新测试已齐备。启动前发现历史源状态 `stopped_at=2000`，按报告47明确新阶段固定共同500步停止规则，保留历史终止证据、原LR配置和候选best/stale/patience_anchor，不修改任何旧实验。该规则同时适用于两臂，唯一差别仍为辅助强度。

24项新单元测试通过，覆盖.02梯度与历史辅助完全一致、.2显式等权求和、来源候选而非历史控制、模式/Adam/游标迁移、磁盘恢复无别名、事务回滚、旧早停不能默默引入、两臂非零辅助及跨设备输入身份。82项相关回归通过（8+13+16+16+16+13），共106项；167 PowerShell语法检查及git diff --check通过。合成单元不替代真实机制。

审批：`results/mel_source_strength_import_20261003/approval.json`，SHA256 `cbb5460d86d746efc98c20721e41ea0e0e7cbdc10404a6afc30303151383d794`。新协议、165–168、新测试、报告46/47及真实源/诊断绑定已封存，之后不得改写。

真实CPU和CUDA各最多3个唯一输入步2001..2003，另含两步磁盘重放和第二臂故障注入；均退出0。完整model/Adam/模式/暴露、选定候选的日程历史、CPU/CUDA RNG、输入游标迁移通过；重放位一致、全事务含CUDA RNG回滚、保存无别名通过。两端输入PCM哈希及元数据完全一致，但不声称CPU/GPU数值相等。两个强度分支实际各有6个有效辅助槽；CUDA三步累计贡献分别约.00497843和.04975561。机制不计入正式500共同更新、不算音质证据。

CPU mechanism SHA256：`20899f4aba04519a78fdc6eff6de6239a3d8b3e9c0955b4d66008bed18a299dc`。
CUDA mechanism SHA256：`737429d2f381dd799f61bb0d8750a4cb4313ce0f7d56168b1da565fd2f3622ea`。
独立verify_proof及跨设备输入检查退出0，日志在新import目录。该日志字段 `source_checkpoint_sha256` 误命名为新binding对象；真实源摘要以封存protocol的 `origin_checkpoint_sha256=9bb5bfd7436da1579803b4e0430042688c4cb377207f2f2ef2778959f2965c00` 为准，旧源文件与SHA未改。

## 实际运行

Windows PowerShell5.1/WMI独立启动探针通过：helper父进程WmiPrvSE、无job object、ffmpeg/ffprobe可见。真实启动前无其他教师/学生训练worker，GPU空闲5580MiB、D盘空闲28888444928B，均满足门槛；未结束用户GPU应用。

只实际启动一次：`results/mel_source_strength_launch_20261003/detached_launch_20261003_042125_5173887.json`，receipt SHA256 `aedd105da0946d4cb1a689c01b59394457b1a6f36f3f25fb24079bad4de6f26d`。helper60052、venv shim60792、实际worker57996，是父子链而非双训练；worker创建于12:21:29。这些PID仅启动关联，后续以状态动态PID/实际创建时间和命令行为准。

运行目录 `results/mel_source_strength_20261003`。预算2001..2500共500共同步，每250评分。兼容键 `htdemucs_waveform_control=source_aux002_control(λ=.02)`、`kim_melband_waveform_candidate=source_aux020(λ=.2)`；两组都是Mel标签/真值，不是HT教师比较。起点为完整source_aux002候选2000，两臂全六槽相同目标/等权/同输入增益顺序，3真值3伪标签，原图/LF44/24首/FP32/裁剪与辅助活动门槛不变。伪标签及纯器乐不启用辅助。

12:22首次复查：status=running，phase=development_baseline，step2000，additional_step0，error=null；起点全状态已保存SHA256 `c5b662ea7fc23370b4442769c9716d005531f527256dca58ec6a43e63516fec0`。实际worker CPU时间由19.03125增加至102.375秒，stdout核验消息，stderr空；未发现重复任务。当前仍是CPU起点评分，不声称GPU500步已完成，基线未提交评分前保持pending。后续不在CPU开发阶段导出音频。

进度/复核/通知写独立 `results/mel_source_strength_monitor_20261003`。2000仅基线，2250/2500才是两个新增评分；独立168核对31首177视图、模型Adam/哈希、逐曲聚合和原资格政策。保留历史早停事件，不因新阶段独立预算而视为符合资格。比较原冻结、2000起点与并行控制的残留、重建、伴奏和纯器乐；不以loss排名，不把旧开发当新盲测。结束须相符detached_exit真实0、完整终态、无worker后166 verify/168复核。只导出原预选三源、同输入同音量，每阶段一次。

巡检保持ACTIVE每20分钟，并已按实际目录与回执更新；正常安静，不自行停用。冻结/历史脚本/协议/缓存/标签/板端及vitis_journal.py不改，无SD/部署/Git提交推送。全程NONRELEASE，独立真实验收与人工本轮听审不自动满足。

12:26启动后追加核验：worker进入training、状态step2010/additional10、CPU788.9375秒、error=null。新168 summary2000退出0；31首177视图、两臂model/Adam/模式均重现所选源候选，评分相同，旧终止证据/新固定预算身份核验通过。基线复核SHA256 `39c1d669a0ac01e263b84db18b7fe010809231db05981765ad26d37107a0dc47`。2000已分析，只是起点而非新优化成绩；两臂资格仍false（弱域残留仍高于冻结），不能视为新训练退步或改进。已存在summary2000不要再生成；下一新增阶段2250。
