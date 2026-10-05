# 68 — 完整CPU诊断的独立复核恢复协议

2026-10-03 21:31（Asia/Shanghai）。182的新CPU零更新诊断已经完成36模型批216槽，前台session14585真实exit0，21:20:47终态complete/error=null，worker61448和shim67380已退出。不是新的训练500步，也不是音质改进。182首次独立verify真实exit1，错误Changed committed row；原脚本、测试、plan、36行、inputs、diagnostic、状态及失败verify日志全部封存不修改、不重跑模型诊断。

只读原因核对：128.seal在写row前向该同一字典添加content_sha256，随后rows.append的嵌入行也含此字段。182.verify仅对文件行调用plain去掉封存字段，却与含封存字段的嵌入行相比，字段集合不一致而拒绝。36/36文件实际SHA均与diagnostic清单相符，PowerShell读取后完整文件行与嵌入行的JSON都相同，两边都含content_sha256；这只是定位证据，还不替代独立完整封存复核。没有文件被改、模型失败或GPU问题证据。

保留[运行完成但首次核验未通过的证据](D:/WORKBUDDY/STEM/npu-stem-repo/results/train_context_reconstruction_recovery_monitor_20261003/run_completion_unverified.json)，真实run exit0和首次verify exit1分别记录，不能合称182 verify通过或未验证结果已经可发布。

持续优化授权内新建183_verify_train_context_reconstruction.py和_test_train_context_independent_verifier.py，仅prepare/verify入口，不训练/模型前向/反向/优化器/CUDA/音频导出。使用results/train_context_reconstruction_review_20261003新目录，不修改182/181及历史绑定。独立预算只是核对已有12批72输入、36行216槽一次，无模型更新、源游标/Adam/模式/权重/grad/RNG不动。

唯一校验修复为对称比较文件行与嵌入行的完整封存文档：先按原128.content_digest分别核对两者封存哈希，再要求完整canonical JSON（类型敏感）相同，且文件SHA与原清单相符。不删任一边字段、不忽略封存摘要或掩码/波形指标，不放宽误差、角色、预算或模型身份。新单元覆盖128真实seal变异的阳性、首次旧非对称比较必失败的回归、缺摘要/坏摘要/换摘要/正文篡改/重封后变更/整数bool差异/清单SHA不符/非有限/少行等阴性。

新plan绑定183/测试/本报告、原182封存plan与全部绑定、所有36行/inputs/diagnostic/status、真实run完成与首次verify失败证据和日志。准备前确认没有活跃训练诊断，磁盘>=12GiB；工具与测试齐备才prepare。verify独立执行一次：原plan、源model/Adam/参数模式/角色/停止证据/数据锁/哈希仍全核对；输入与真实177等价审计、行顺序/计数、同区间/同谱/CPU严格FP32/runtime/未改模型状态/伪标签及零参考边界、原summary重新聚合由不变182/181验证函数检查。原182.verify不再调用，旧全库DEV/试听/真实诊断不重算。新结果review只写新目录，完成需实际exec退出0/无活worker，失败保留证据只读诊断，不盲重启。

成功独立复核后才分析上下文/LF44/重建结果；保持共同比较130*256..length-2*256，native98仅描述、小样本不推广全库/板端/理论上限、LF能量比例不是波形残留；不按训练loss或伪标签当最终真值选模型。不因此直接修改LF44/生产图/硬件，后续单变量有界方案另写协议，改图/板端状态先审Z7020/NPU周期/算子/BRAM/DSP，不放宽目标。

全程NONRELEASE、release_selection=NONE，人工3500听审及独立真实验收PENDING。当前无GPU训练，不新增试听或重新生成历史结果。巡检保持每20分钟ACTIVE；健康无新可行动结果安静，真实失败/新结论才通知。不提交推送、不上传音乐、不写SD、不部署、不新聊天或子代理、不改驱动注册表推理精度，保留所有历史证据及其他未提交修改。
