# 54 — Mel 纯器乐保护对照已启动

2026-10-03，NONRELEASE。依据用户持续优化授权和报告53，独立启动一次新500共同更新对照；未修改旧2500上限、旧审批或板端镜像。

## 配方与状态迁移

两臂从强度轮完整 `.2` 候选2500的同一model/Adam/参数顺序/全部模式/更新暴露/CPU-CUDA RNG/游标/日程分叉。控制为 `instrumental_weight1_control`，候选为 `instrumental_weight4`；内部兼容键不代表HT教师。

唯一变量是纯器乐TRAIN槽2的基础wave+complex L1权重1→4。其余五槽基础权重1，两臂辅助λ均为.2，活动RMS/Gram门槛不变；器乐和伪标签没有辅助项。分母固定6：`sum(w_i*base_i+.2*aux_i)/6`。控制保持旧166的FP32操作顺序。图、LF44、24首Mel标签、3真值/3伪标签、PCM/目标/增益/顺序/窗口/裁剪/LR配置均不变。

源实际stop=2500、legacy=[2250,2500]和旧best/stale/patience_anchor作为证据保留。新固定500共同阶段活跃stop起点清None；旧日程历史不重置，后续旧政策触发记legacy_stop_events，3000硬终止。新更新2501..3000，每250评分；2500只是基线，2750/3000才是新增训练阶段。

## 已完成启动门槛

新实现与损失核40项单元、相关62项回归共102项通过。固定真实TRAIN counter2500六槽检查模型更新0：控制数值/梯度与旧λ=.2实现位一致，候选梯度匹配控制加3倍器乐基础梯度，最大元素误差2.98e-8。这是单批机制检查，不是音质或全库因果证明。

封存后CPU/CUDA各3个唯一输入2501..2503验证通过：源model/Adam/模式/RNG/游标/日程迁移、严格源CUDA运行环境、磁盘位一致重放、保存无别名、第二臂异常含CUDA RNG全事务回滚、实际非零器乐加权及两臂有效.2辅助。独立verify_proof及跨设备输入PCM/元数据比较退出0。CPU是显式机制迁移，不声称跨设备数值resume；机制重放不算本轮500步或质量成绩。

| 证据 | SHA256 |
| --- | --- |
| 新approval.json | `b5dbbd1f50c3e1a7f9bcc1b08e5eba4de4d037fa2750ff8cc22da1d09aa1d810` |
| TRAIN audit plan | `5de722b90bd9c6d34f5ef5a1b4bf5cde921c011b270835bc9935732e47f98093` |
| TRAIN audit result | `507bf201849136fd501f128ecbfa0626e6d133e83a1280fd4ae13edc5fd2551b` |
| CPU mechanism | `7368691a2adea59442b4278863a7fd924d0ec2e39168c43395a84adce748b3f1` |
| CUDA mechanism | `be0414322bb9aba20b01af9803d834efe6f5f5077f15e301019007b3ad12fe82` |
| 独立proof_verify.log | `d440e06707ffdc5152dea82c062518ba6fdba1589ba1ad631c75c542deb08382` |

新171/172/173/174、170/159、测试、协议、报告52/53及绑定现已封存，不修改。详细准备证据位于 `results/mel_instrumental_protection_monitor_20261003/preparation_review.json`。

## 实际启动与巡检

启动前磁盘空余28,417,814,528 bytes，GPU空余5724MiB，无重复活跃训练，满足12GiB/2300MiB门槛。WindowsPowerShell5.1/WMI独立启动命令退出0；真实回执为 `results/mel_instrumental_protection_launch_20261003/detached_launch_20261003_070144_6350378.json`，SHA256 `7c0f0ad39bd93da0d29fa5277a7d8a505bcd5dfd8d5b66b2442c2599a05f9ac4`。

回执UTC为2026-10-03T07:01:47Z，WMI parent=WmiPrvSE.exe、in_job_object=false。初始helper60432、venv shim59340、实际worker57764，创建时间和命令行均匹配172 train新目录；这是单worker父子链，不是重复训练。后续以run_status动态PID为准，不固定依赖这些初始PID。

首次状态running/development_baseline/step2500/additional0/error=null，起点完整状态已保存。worker CPU从20.5625秒增加到315.546875秒，frozen_scores/selection_suite有新写入，stderr空，初始化/CPU开发评估正常。启动不是完成，当前未有本轮新音质成绩；不能将原2500基线当新优化结果。

15:05后续检查：实际worker57764已进入training/step2501/additional1/error=null，CPU增加到635.21875秒，stderr仍空。CPU开发基线结束后，174 summary2500独立复核退出0，原31首177视图/哈希/模型Adam/原政策/源候选2500与停止证据检查通过，review.json SHA256 `30774f1013e29102fe4411d3e14738da67f18a94fb7f0d0c96566f41f8804e3f`。两臂基线分数相同、eligible=false，与源候选一致；没有新优化质量结论，不重复导出基线试听。独立monitor已标analyzed_steps=[2500]、pending=[]；下一新增评分2750。

实际运行 `results/mel_instrumental_protection_20261003`，独立monitor保存进度、分析步数、历史最好和通知。现有同聊天巡检已按官方OpenAI文档对应的更新流程切到新目录/回执/工具/预算，保持ACTIVE、20分钟；正常无新可行动结果时安静。174摘要/试听只在完整阶段生成一次，不与worker CPU开发评估并行。结束必须相符detached_exit真实exit_code=0、完整终态且无worker，再172 verify和174独立复核。

目标是验证能否恢复器乐保护而不丢失人声削弱。若两者仍权衡，不机械增加器乐权重或λ；按证据检查辅助内部目标或LF44表示/重建。所有成果NONRELEASE、release_selection=NONE，不自动满足人工听审/独立真实验收。未上传音乐、写SD、部署、Git提交推送，冻结学生和历史绑定保持不变；巡检不会自行停用。
