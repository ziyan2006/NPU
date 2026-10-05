# 109：EMA解码器直接目标守卫进展，完整运行时仍待认证

2026-10-04；NONRELEASE，release_selection=NONE。

## 结论与实际执行

新隔离209直接目标文件守卫已经实现，35项**新**元数据/PE/文件锁单位前台exec chunk `be6ebb` 实际exit0，完整消费 `NATIVE_EXIT_CODE=0`；单位报告耗时1.832秒，exec等待2.423秒。没有持续session、WMI启动或detached_exit，不伪造退出文件。此次没有执行203..208或任何旧单位/固定输入重放。

只构造1次实际只读目标守卫，并执行1次内部为空的真实文件锁定区间：构造检查五文件一次，区间前检查一次、保持原句柄时重新打开读取作区间后检查一次。两份wrapper、两份实际EXE及当前System32 cmd.exe各读取3次；不是三次解码，更不是真实音频draw前后认证。真实锁定区间开始12:08:22.044983Z、完成12:08:23.119974Z。完整证据保留每个实际PE导入名称和符号。

209封存为**直接目标组件**，完整真实音频后端gate仍PENDING。不能倒推208执行时的EXE认证，不能把209空区间当作208重放、模型零更新音频审计或4501更新机制。208成功输入身份与208目标链缺口均保持原事实，不回写208；207仍使用合成后端、actual_audio_backend_verified=false。

## 新守卫范围

固定识别原两行ASCII wrapper的绝对引用目标及原样 `%*`，拒绝额外命令、变量扩展/其他shell形式。匹配实际PATH wrapper选择、完整PATH/PATHEXT/COMSPEC/SYSTEMROOT/cwd/Python/Windows版本身份。绑定真实EXE字节，不依赖版本字符串。当前COMSPEC必须是实际System32 cmd.exe；其字节在本轮首次只读测量后绑定，不宣称独立供应链信任。

Windows只读句柄只允许READ共享，在持有期间拒绝WRITE/DELETE共享；原句柄仍在时再次读取并核对五文件SHA。小临时合成文件检查了拒写/拒删除及部分打开失败清理，不对真实目标尝试写入/删除。环境、选择目标、预先或随后字节、内部身份变异、BaseException和嵌套区间均拒绝/poison；poison后禁止继续。导出contract为无别名深复制。此锁定不保证对抗恶意OS/目录变更，不锁未枚举传递依赖，不是raw/Adam/EMA训练事务或回滚。

真实直接文件SHA256：

- ffmpeg.cmd：632b62597b7f0a982601dcb0bafe59029ba9dbc21837abe0f4235aaef01db982
- C:\ffmpeg\bin\ffmpeg.exe：72a489eccd008c2ec2c0a5856c5c75bc3d8bbfa90166c4566865c246445e6aa3
- ffprobe.cmd：2b2905bfb7bdfcb7e437e04341871f8e40a7d9851d4f1c358dc8434509df1b5b
- C:\ffmpeg\bin\ffprobe.exe：19202b23c0043f15ad1b7bce2344f406fd52bd6efd8f995ce02e7392a1cec52f
- C:\Windows\System32\cmd.exe：97ac98b1a92c286054cce55239cfccdfc23a5517bd07fe693072c9ca96c7dabb

## 静态依赖盘点与未完成边界

新解析器按实际raw section映射RVA，识别PE32/PE32+名称及ordinal、普通/延迟导入；检查文件/section/directory/name/thunk边界、终止符与重叠，拒绝未支持的VA格式延迟表、IAT-only绑定格式。它不加载PE，不是完整PE安全审计、ABI认证或Windows loader解析器。实现依据[Microsoft PE格式](https://learn.microsoft.com/en-us/windows/win32/debug/pe-format)。

| 实际x64文件 | 普通DLL/符号 | 延迟DLL/符号 | 动态加载符号 |
| --- | --- | --- | --- |
| ffmpeg.exe | 31 / 725 | 0 / 0 | GetProcAddress、LoadLibraryA/W、LoadLibraryExA/W |
| ffprobe.exe | 30 / 721 | 0 / 0 | GetProcAddress、LoadLibraryA/W、LoadLibraryExA/W |
| cmd.exe | 43 / 287 | 5 / 19 | GetProcAddress、LoadLibraryExW |

两份EXE均含CRT API-set导入和动态加载函数；cmd还有API-set及延迟导入。静态符号存在不证明某次输入实际加载了什么，也不能假定EXE目录没有DLL就证明静态自包含。

Windows DLL解析还受API-set、SxS/重定向、已加载模块/KnownDLL与搜索规则影响；API-set名称是逻辑契约，不一定对应同名物理DLL。因此不能把导入basename机械映射到System32就称完成链认证。[Microsoft DLL搜索规则](https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order)，[Microsoft API-set说明](https://learn.microsoft.com/en-us/windows/win32/apiindex/windows-apisets)。

后续仍须在**新隔离实现**中完成适用API-set host/forwarder/SxS/实际loader和动态LoadLibrary/GetProcAddress路径、实际shell启动策略/参数与相关配置、Python soundfile/scipy/numpy原生运行依赖认证。当前cmd字节及环境绑定不是AutoRun/真实shell调用认证。209对“完整运行时授权”请求始终拒绝；没有静默给full gate放行。新真实输入必须满足全部依赖前置及执行区间验证，不能仅凭本次空锁定区间启动音频或GPU，不重新解码208固定counter4500参考批。

## 实际零计算边界与测试

真实音频draw0、解码器/任何child process调用0、原学生训练PT反序列化0、Module构造0、forward0、autograd engine0、Adam构造/step0、学生更新0、DEV0、音频输出0、教师推理0、CUDA初始化false。此次读取的是程序文件字节及既有JSON/日志，不是音频源或模型。工具和新套件都未导入torch/numpy/scipy/soundfile；执行前禁止subprocess/os.system，且没有被拦截尝试。没有修改PATH、注册表、驱动、系统安装或用户程序。

35新单位覆盖真实五文件目标检查/空持有、合成PE32/64、名称/ordinal/延迟RVA、动态符号盘点、缺失/截断/重叠/异常格式拒绝、严格wrapper、小合成磁盘metadata拒覆盖往返、bool/int/list/tuple与字典顺序类型敏感seal、预后字节/环境/目标变异、BaseException/nested/poison、导出无别名、Windows临时文件共享门槛及失败清理。Python RNG前后不变；CPU/NumPy/CUDA运行库根本未加载，不声称验证它们的训练事务。

## 封存和独立只读归档

- scripts/209_ema_decoder_target_guard.py：6b904f9f1b8497d16f3ecca85d9122b000604f4e357456eef0a0e9ddf28fe7d2
- scripts/_test_ema_decoder_target_guard.py：6ffadcb649ed1d594f821b083cded6d9cf7193bf01856f74eae51dd57a9c02d2
- docs/ema_decoder_target_unit_scope_20261004.json：767c96fff50cf8f6bea58f754a65ed6d7edc6e7e69a4a5096953d011d85f6699
- results/ema209_target_unit_tests_attempt01.log：5d290a90e59f3dbb9453848960a65830525de19279345d7ab50e960ee57481a7
- results/ema209_actual_target_hold_attempt01.json：f8c0b1c199e4f77fed650a833bf1281d3e4b795dd693cfd9148a5505592c6454；自身typed209 seal=019a7aba7805a2ae3003f4f0d7b9bd8e2edef736b3445eaa20f4d488e0512752；嵌套contract own typed209 seal=8b6a53c851e0a62eb248c5b1637f085ffb9c0557ded893146a34087d354a1db0。

monitor=`results/mel_ema_decoder_target_monitor_20261004`。

独立新只读reader `_review_ema_decoder_target_archive_20261004.py` SHA d78dc295d0b8a829d5b7b7a9d798e15ccd6ea4781cb9f0ebe91186616eaae15f，前台chunk `07d6dc` 实际exit0，完整消费NATIVE_EXIT_CODE=0。其21绑定（含实际退出receipt、新209原始证据和封存203..208核心/208证据）前后相符；新证据及嵌套contract各自用独立typed实现检查own seal，保留完整静态inventory，文件/保存嵌入副本与整个review完整对称typed比较。新增guard构造0/decoder音频PT模型AdamCUDA0；不是旧verify或重复35套。

independent_archive_review SHA cf5f44579268ddf1c29961bb1ba8263efee7e60f94254b420a5789901dcc215d；自身typed209 seal=164e553fa717fe7c16e14c83f5b682f292c4e8fd71f63500e6661faacb5788b3。review日志SHA a47dea4e0d80d84bfe09fdb05c7d5efd893954958e576ce40f9cafe0e1eb747d。

新unit_gate包含25文件绑定，file SHA56f6564c1246281d014e455829185c957e145342af9349807022324bff0c6776，plain monitor content seal=1a400c42c46074e9b2b69b104ba4cb6d22896dd3bcd906a0fea7ba2f8cd58c47。它只认直接目标组件单位通过，full_audio_backend_gate=PENDING、formal_training_authorized=false。所有209工具/测试/scope/log/evidence/reader/review/receipt/gate在此封存，不修改或重复单位/实际空锁定区间；发现缺陷只能新隔离恢复。

## 接续状态

2026-10-04T12:13:07.7289103Z动态CIM无仓库Python，另核对相对203..209命令；D盘free24.560630798339844GiB，210工具及新EMA正式训练目录未见。20项原tracked dirty仍469 insertions/44 deletions，不修改。没有新音质、容量、平台期或发布结论。

下一轮先完整读109/108/107/106/105/104/103/102/101/100/99/36及全部原必读报告、封存209及208..203实际文件，再核对动态任务和现有产物。首先接续传递运行时/真实启动路径依赖的认证，不重跑209/208单位或空锁定/固定参考输入。完整训练-only imported modules、全部损失/frontend/matrix、完整输入锁/cache清单仍须读完；此前截断不算完整阅读，209也未认证这些依赖。

报告102固定单raw/单原Adam、EMA唯一评估变量、500共同更新4501..5000、新4750/5000/硬5000、LR1/kill32/完整aux.2/器乐4/分母6/FP32图、原累计best-stale-anchor及全部旧stop/legacy不变。真实全模型零更新音频审计、最多3唯一4501..4503 CPU-CUDA实际更新/严格源CUDA FP32runtime和原CUDA RNG/原194首个raw实际Adam位一致/真实非零EMA-noalias/完整轨迹磁盘重放/含shadowcount与CPU-CUDA RNG整组回滚/跨设备PCMmetadata全部PENDING。

正式trainer/CLI、协议、审批、封存trainingplan、worker尚无；不提前封存缺工具计划、不启动GPU。全实际gate及disk>=12GiB/freeVRAM>=2300MiB/no duplicate后才WindowsPS5.1/WMI一次启动并保留真实receipt。20346/20470/20534/20628/20737/20835/20935/202及历史诊断训练关闭不重跑，197failed/199verify1/201prepare1永久留证。

人听及独立真实验收PENDING，NONRELEASE。20分钟同聊巡检保持ACTIVE，不暂停/停用/删除；健康无新可行动结果安静，只通知重要新结果、失败或人工处理。按照OpenAI Docs技能核对并仅更新现有巡检的接续事实，保存原周期/状态/目的和完整历史，不创建重复任务；[官方同聊定时任务说明](https://learn.chatgpt.com/docs/automations?surface=app)。本209新结果及部分认证边界通知一次，后续不重复正常完成播报。
