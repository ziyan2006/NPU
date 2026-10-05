# 新202规范Adam存量诊断接口恢复计划（NONRELEASE）

仅恢复报告99明确的201准备接口错误：使用原197实际DEFAULT_OUT读取其封存三inputs，不改197或201，不补OUT别名或猴子补丁。新工具`scripts/202_diagnose_train_adam_memory_canonical_api_recovery.py`、新测试/协议、输出`results/train_adam_memory_canonical_api_recovery_20261004`及独立monitor。先核查已有文件、输出与任务，不覆盖；新完整单位和草稿审查通过后才prepare，前台run一次，真实exit0/无worker/完整预算后新0前向verify一次。

完整继承报告98的未完成目标，不扩大样本、训练或方法：共同LF32 4000 ARMS0与同4500 PT control ARMS0/half ARMS1；原counters4500..4502、197固定输入哈希逐槽一致，3唯一批18槽；9模型批54主前向+每模型首批6独立原194参考，共18重复参考/72槽前向。MUSDB[+6,0,-12]/MIR[+6,-12,-12]/器乐3/pseudo9，不补缺失档，pseudo非最终真值，不拼接任何旧计算。

CPU2、FP32/complex64、detach几何FP64；原6microbatch/352/warmup96/native25088..89344/kill32/辅助.2/器乐4/分母6/活动与增益顺序均不变。autograd.grad不写parameter.grad，0 Adam构造、step、CUDA、更新、PT、音频、DEV或部署。完整源model/modes/已有grad/Adam逐参数映射与全部flags/游标日程/RNG/输入只读不变。

唯一权威为直接原194完整组合标量梯度按原六槽FP32顺序累计；规范base+原完整aux仍原rtol2e-4/atol2e-7硬门槛而非位一致。完整aux同图完整余切VJP逐位一致，余切分项和及合并VJP保留原strict门槛；独立分项参数和的strict失败保留测量，不放宽、不补误差到组。每模型首批原194参考的PCM/wave/mask/标量和逐槽、总权威梯度逐位一致。所有父文档、文件和嵌入行、totals各自验seal后完整类型敏感对称比较，防止199单边seal回归。

新增单位必须覆盖原197实际DEFAULT_OUT/三封存inputs路径（0前向），同时保留201的所有数值/只读/计数/拒绝覆盖门槛。新plan绑定报告99/100和201失败全部文件、报告98/97/95/96/94/93/92/91/90、新200实际证据、原379绑定及LR状态/审批/launch-exit/verify/summary/stream/矩阵/缓存依赖；历史绑定不动。

磁盘>=12GiB、无重复仓库任务才执行；失败保留真实started/completed暴露与全部证据，不盲重启。u与d=-实际groupLR*u是存量moments方向代理，g·d负值仅一阶局部指示，不是实际delta、下一步Adam、应用收益、音质或因果证明，不据此清空Adam或改LR/beta/LF/损失。无新GPU训练。人听与独立真实验收PENDING，NONRELEASE；巡检持续ACTIVE、20分钟、只重要结果通知。
