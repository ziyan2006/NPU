# 108：EMA真实输入组件进展与解码器链边界

2026-10-04；NONRELEASE，release_selection=NONE。

## 结论

新208真实音频输入组件的35项新CPU检查已经实际通过；前台session64482，start chunk9aad01，completion chunk7b471f，真实native exit0，完整消费`NATIVE_EXIT_CODE=0`，单位耗时44.997秒。输出证据文件在08:05:20Z写出，退出输出在本轮收集；不能把后续heartbeat时间当作单位执行耗时或退出时刻。

实际读取一个固定TRAIN counter4500的六槽输入，并在恢复相同语义游标后进行一次同批重放。输入PCM、Mel目标PCM及完整类型敏感元数据与封存197 inputs[0]一致；两次真实张量逐位相同且无别名。只有1个唯一批/6个唯一槽，另6槽是重复重放，不是独立样本、训练4501或Adam更新。

归档复核同时发现208的解码器绑定范围缺口：PATH选择的是`ffmpeg.cmd`/`ffprobe.cmd`，208绑定了它们的字节及报告版本，但没有绑定实际转发目标EXE字节。因此本结论是“真实输入组件单位通过”，不是“完整真实音频后端所有依赖认证通过”。完整后端gate仍PENDING；保留208原实现与成功证据，在新隔离集成中补足，不回写或重跑本35套。

## 实际路线与不变项

208沿用原143锁定真值TRAIN加载器、149批准的24首单Mel目标加载器、136局部counter recipe，通过封存206新的4500..5000有界单目标路由。没有构造旧193/149双臂训练stream，没有抬旧4500守卫；旧审批只作为输入来源身份，不借用为新训练授权。

真值池顺序与数量MUSDB73/MIR81/器乐11；每批原三真值后原三pseudo。固定counter4500实际真值是Patrick Talbot - A Reason To Leave.stem、bug_5、Guete_des_Geschicks_v1，vocal_db=[6,6,0]；pseudo是song_0058、song_0219、song_0139。三器乐/弱角色不改、器乐目标精确零，pseudo不是最终真值。所有原裁剪、共同音量、原源完整PCM身份、352帧/warmup96/native25088..89344和字段保留。

构造前验证完整类型敏感原4500 sampler（seed20261002/control原LR审批/真值锁/kim_melband）；构造与两次draw不消耗全局CPU/Python/NumPy RNG。原真值LRU最多3首、pseudo完整源LRU最多1首；原音频与标签只读。真值池/配置、pseudo rows/by_id/标签映射/签名、实现函数、解码缓存字节及绑定文件签名变异拒绝。失败draw会poison这个新后端；单独小合成路由复现BaseException poison，不额外调用真实音频。

外国解码LRU不是完整raw/Adam/EMA/CUDA事务。没有因206语义游标回滚就认定外国后端cache已经回滚；不能静默重试失败后端。

## 真实执行范围

两次成功真实输入draw（1唯一+1重复）。音频源字节哈希和原数据解码确实发生，不能说真实音频输入0。只读检查24份源MP3及48份Mel标签的SHA/header，不是再次275首扫描或教师推理。

原学生训练PT反序列化0、学生Module构造0、forward0、autograd engine0、Adam构造/step0、学生训练更新0、DEV0、输出音频0、教师推理0、CUDA初始化false。禁止API在构造/音频draw前安装，单位对禁止API的主动探测均在执行前拦截，不能把这些被拦截尝试算实际执行。外层CPU/Python/NumPy RNG恢复。

只读使用197 inputs.json数据及其原seal，未执行197/202工具或旧数值verify。20346/20470/20534/20628/20737及历史训练诊断单位不重跑。非WMI，无detached_exit，不伪造。

## 封存及独立归档复核

新文件SHA256：

- scripts/208_ema_authenticated_audio_input.py：8d208e34ed1ce6b18997c64d75ad1aa03d67401b7752fa8d995da23d64982825
- scripts/_test_ema_authenticated_audio_input.py：43f3fe3fdf6c389472d0f652be28a4ddbd262fc4e40cd927f533ef9a85051c4c
- docs/ema_authenticated_audio_unit_scope_20261004.json：df91a1517a124a606255848d257c1b6e1ac7a4dd25788a68e62ab88fcfc3d573
- results/ema208_unit_tests_attempt01.log：51777a1731a86d431db20398aa9a9c8477aaba9d1b4cf1910a0e61c09e2ac81c
- actual_input_replay_evidence.json：d198b7f32b0b208645496de0005a929af11174d757e89ee5f366af4b01387a3a；自身typed205 seal=3d034bb4ef2b07c4f2d58d9a2fd0d51457e8c79b53f0479bbbedb483c306df75；嵌套contract自身typed205 seal=5f95a3f434fd5669ce5547fca51c0e43b62261750558c85992eb8d8d86f75718。

monitor=`results/mel_ema_audio_backend_monitor_20261004`。

独立新只读reader `_review_ema_audio_input_archive_20261004.py` SHA45c103296fdf65ebfcf63d5f590bd030f2bf1081c31db6c852296c56ebe915d9，session60218/start72a50c/completion01c25d实际exit0，完整消费NATIVE_EXIT_CODE=0。202文件绑定通过，包括173个输入contract绑定及19个实际选中真值原始文件；自身及嵌套seal各自检查后与原输入完整对称类型敏感比较。禁止真实后端构造、音频解码/读取、PT反序列化、Module/Adam/autograd/CUDA执行；新增draw/模型计算/更新均0，RNG不变。

独立archive文件SHAd1fe276c759dfbbcc4b1c0425426f50c20d8d0151cb974a5541ee9a0c3a73bc5，content seal=ae02f47d30e4ce83a468ff0a4af279831ff090f5d6a551031baed48c7fde9f94；decoder_chain_scope_review文件SHA9408cfad301350237863cbe5a58e74ae45b8d156e40349cc5ba9a0456bea226d，content seal=4cb58b0e72664da56bebe696e8fb73798eec9a91444aaa3d95e7c33334740c9e。

新unit_gate文件SHA3fb6fd6610f8290a9ed5479d2ce1177e9445798902b4e8256309b20b908c0fc1，content seal=392ecd38faf6b29a1204eaadf270594f5789b7a00b7054fe9c1d928019f5b076。gate包含204绑定；23c346只读检查全部204 SHA与三个新monitor seal真实exit0；不是单位/音频重放或旧verify。

以上工具、测试、scope、成功log、原input evidence、review及gate封存；发现缺陷只能新隔离恢复，不修改208或用新代码冒充原35套成功。

## 解码器链边界的真实证据

414a99只读查看两份PATH wrapper：分别转发`C:\ffmpeg\bin\ffmpeg.exe`及`C:\ffmpeg\bin\ffprobe.exe`，参数原样`%*`。25ffc4在2026-10-04T11:36:34.5085389Z只读哈希：目标ffmpeg.exe SHA72a489eccd008c2ec2c0a5856c5c75bc3d8bbfa90166c4566865c246445e6aa3；ffprobe.exe SHA19202b23c0043f15ad1b7bce2344f406fd52bd6efd8f995ce02e7392a1cec52f。

这是单位后的目标字节快照，不能倒推单位前/执行中的目标字节认证。35项实际成功、固定输入身份测量与完整后端依赖门槛是分开事实；不是音频/训练运行失败。后续新隔离组件须在新真实输入前核定目标链及适用的传递运行时依赖，并在draw间验证；不能仅绑定转发脚本或靠相同版本字符串通过。

## 仍待完成与下一步

208没有接入207全状态容器（207仍绑定合成后端、actual_audio_backend_verified=false），不是正式trainer/CLI或CUDA恢复容器。原学生训练权重仍4500；报告102单raw/单原Adam、EMA唯一评估变量、500共同更新4501..5000、新4750/5000及硬5000、原LR1/kill32/完整aux.2/器乐4/分母6/FP32图、累计best-stale-anchor/旧stop/全部legacy不变。

先核对实际新产物，补新隔离解码器目标守卫，再接单raw/单原Adam与完整CPU-CUDA活容器。继续完整阅读训练-only imported modules、完整锁/cache清单与其余实际依赖；此前截断大cache/锁输出不算已完整读完。真实零更新模型音频审计、最多3唯一4501..4503 CPU-CUDA实际更新机制、严格原CUDA FP32runtime/原CUDA RNG、原194首个raw实际Adam更新逐位一致、真实非零EMA/noalias、完整磁盘轨迹重放/含shadow计数与CPU-CUDA RNG整组回滚、跨设备PCMmetadata全部PENDING。

正式trainer、协议、审批、封存trainingplan、worker尚无，不提前封存缺工具计划或启动GPU。全部真实gate及disk>=12GiB/freeVRAM>=2300MiB/no duplicate通过后才WindowsPS5.1/WMI一次启动并保留真实receipt。

2026-10-04T11:38:27.3232264Z动态CIM无仓库Python任务，另核对相对208单位/reader命令；D盘free24.56106948852539GiB，无209工具或新EMA训练目录。a0b12b的20项原tracked dirty diff保持。没有新音质、容量、平台期、Adam/LR-beta根因或发布结论；人听及独立真实验收PENDING。20分钟同聊巡检ACTIVE不暂停/停用/删除，健康无新可行动结果安静。本组件结果通知一次，后续不重复正常完成播报。
