# 111 — 实际加载文件路径已观测；训练仍为4500（NONRELEASE）

归档日期：2026-10-05（Asia/Shanghai）。这是解码器基础设施进展，新增学生训练更新0，不能算4501或音质改善。当前目标仍为报告102的单raw/单原Adam、独立EMA固定500更新至5000。

## 实际结果

新211最终24项纯/mock单位通过，前台exec edef25真实exit0，完整消费NATIVE_EXIT_CODE=0，单位0.027秒。随后仅运行两个显式绝对EXE的-version诊断；exec32efd8真实exit0，完整消费native0，实际等待3.6783481秒。ffmpeg PID87964与ffprobe PID97548，各44个调试事件、36个实际文件事件（1EXE+35DLL），实际调试EXIT与Popen wait均0。共72文件事件、37唯一物理文件路径。

实际区间分别为2026-10-04T13:17:24.541955+00:00..13:17:25.085049+00:00及13:17:25.086675+00:00..13:17:25.590443+00:00；不把后续心跳或归档时间算执行耗时。实际进程早已退出，没有WMI、持续session或detached_exit。

从Windows CREATE_PROCESS/LOAD_DLL提供的实际文件句柄读取最终路径、文件ID、大小、最后写入时间与SHA；每个事件句柄保留至实际EXIT，退出后同句柄完整类型/字节复测相等。两EXE分别保持封存209目标SHA72a489eccd008c2ec2c0a5856c5c75bc3d8bbfa90166c4566865c246445e6aa3及19202b23c0043f15ad1b7bce2344f406fd52bd6efd8f995ce02e7392a1cec52f。

其中comctl32.dll实际来自Windows WinSxS目录，不能把DLL basename机械映射到System32。这是实际物理路径观测，不是SxS/API-set/forwarder因果解析。[Microsoft调试事件](https://learn.microsoft.com/en-us/windows/win32/debug/debugging-events)、[LOAD_DLL事件结构](https://learn.microsoft.com/en-us/windows/win32/api/minwinbase/ns-minwinbase-load_dll_debug_info)。

## 严格范围

仅创建自己的两个新版本诊断进程，shell=False、显式executable/argv0、close_fds=True、CREATE_NO_WINDOW及DEBUG_ONLY_THIS_PROCESS、隐藏startupinfo、env/cwd继承。没有附加现有进程、写内存/寄存器、注入或终止用户程序。只吞初始first-chance loader断点；其他异常不吞。失败清理仅允许detach自己新建的PID，不是任意用户进程处理。[Microsoft DebugSetProcessKillOnExit](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-debugsetprocesskillonexit)。

实际child/decoder版本执行为2，不能说原生Windows调用或decoder0。音频源/真实draw、学生PT反序列化、Module/forward/autograd/Adam构造step/学生更新/DEV/音频输出/教师推理均0，torch/numpy/scipy/soundfile未导入，CUDA未初始化。父Python RNG/环境不变，不称完整训练RNG事务。

加载通知发生在文件已映射之后，文件SHA不是加载内存/重定位字节认证。句柄保留不是完整预执行运行时清单、恶意OS/目录防护或主EXE持续只读锁保证。版本诊断不证明条件音频解码路径、非调试启动等价、PCM身份、API-set host/forwarder/SxS/KnownDLL因果、Python soundfile/NumPy/SciPy原生链；不能倒推208历史child路径。full_audio_backend_gate保持PENDING。207合成后端标志及208/209/210全部原证据保持，不自动回写。

## 失败、归档与封存

首18单位752af1 exit1因新ABI期望80错误，实际x64 CREATE_PROCESS_DEBUG_INFO为72；在未封存新211修正期望，旧文件不改。23单位89c641成功草稿保留。第一次新入口7084f5 exit1因PS7 UTF8日志被误按UTF16解码，发生在Native构造/输出目录/Popen之前，实际child0；严格按BOM解码并补第24项单位后才执行成功两诊断。全部草稿和失败日志原样保留，不伪称失败尝试成功。

新独立只读reader不导入211或调用Popen/音频/模型。exec206b45真实native0，完整消费退出，70文件绑定；原始proof和各row分别验typed209 own seal，完整顺序事件journal/保存嵌入/全review对称typed比较一致。当前37文件后置重哈希相符，仍不是倒推预执行认证。新增诊断child、draw及模型计算均0。

关键文件SHA：

- 211源：4d700a762fa998450571eb993e0702c430dfed4a15d2cd5e22df200597513eeb
- 新测试：3c037442ebc7020081a18b53ba867f31c01a06a19b68e8eeede406b76d33397b
- 范围：c7a5c38d523455bbac4c6334d81718fdcdc0d63dc292b1e80ff7f75be8dd4a5a
- 最终单位日志：4bc2c9d41655c337b2a9a5811ea4ec63d0a4749f42ae77cc82860338efc44cbc
- 实际诊断日志：0d4d8a6c29945a24b0cb9057ba3e0f76d001e11a81dec39df003b140ea2c4a61
- loaded_image_evidence：5dd237dfeea7861bdddc519a6979997ebe5b460a8528d2ab1b605d9c818dbb75；typed209 seal399af12e23506dfb929efeb54c37647ede7fcd0fdccc294d79421ccb6b8f67f8
- 独立reader：04f72afb9f0e572527388f46120131b861329144e0943b4bd39a23ad5aefa644
- 独立review：b38f13db8144475d10f648497aaaa21b632f695e1ed1f60bae37c80bb58e8a33；typed209 seal97ecc838f0d901c9feda7276e2ea105318f58824f2922f0b1180227c51d3f397

全部证据及新gate/progress在results/mel_ema_loaded_image_observation_20261004。现在封存211源/测试/scope/日志/原始证据/reader/review/receipts/gate，不重复24单位或两个version诊断。封存缺陷只在新隔离恢复。

## 当前训练状态与后续

2026-10-04T16:51:12.9738983Z动态CIM无仓库Python，D空余24.559917449951172GiB；20项原tracked dirty仍469insertions/44deletions，不改。最新正式训练完整终点仍LR4500，旧匹配启动/退出已结束，不重启原硬4500工具。

用户在2026-10-05明确要求继续训练。接续范围仍报告102：4501..5000、阶段4750/5000，唯一EMA评估变量，原LR1/kill32/完整aux.2/器乐4/分母6/FP32、原累计best-stale-anchor及旧stop/legacy不变。真实音频全运行时、训练器/CLI/完整CPU-CUDA状态、零更新音频审计、最多3唯一4501..4503真实机制（含原194首raw位一致、EMA非零/noalias、磁盘轨迹与shadowcount/CPU-CUDARNG整组回滚、跨设备PCM）仍PENDING。正式工具/协议/审批/完整封存trainingplan/worker尚无；不先封存缺工具计划、不启动GPU。

接下来完成真实后端与全部训练依赖阅读/认证，再完成单raw/单原Adam训练器与上述机制；全门槛及disk>=12GiB/freeVRAM>=2300MiB/no duplicate通过后才PS5.1/WMI一次启动。历史203..210/202及训练诊断不重跑，197failed/199verify1/201prepare1永留。没有新音质、容量或平台期结论，NONRELEASE、release NONE、人听及独立真实验收PENDING。

定时配置本次实读为PAUSED，未擅自恢复；已向用户询问是否同时恢复并精简提示词。以后应只保存固定规则和最新状态路径，不追加历史阶段全文。继续当前工程不代表正式训练已启动。
