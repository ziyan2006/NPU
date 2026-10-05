# EMA 完整 CPU 活状态容器集成进展（NONRELEASE）

2026-10-04T07:35:23.3182715Z。新207的37项 CPU 集成单位真实通过；前台 session52308，启动 chunk6c9edc、完成 chunk93151b，native exit0，完整消费 NATIVE_EXIT_CODE=0。单位72.579秒。非WMI，没有 detached_exit，不制作该回执。

本次完成的是 CPU 存储与事务集成，不是新训练。原始学习权重仍在4500；真实音频、前向、autograd engine、Adam.step、学生训练更新、CUDA初始化均为0。新的500个合成状态曝光是手动修改小夹具的权重/moments/计数并递推影子，绝不是500次真实训练。CPU 显式迁移不等于源 CUDA 数值 resume。

## 已验证的集成范围

新脚本 `scripts/207_ema_live_cpu_state.py` 统一封装单 raw model、单原 Adam、独立203 EMA、206输入语义游标及最后metadata、源日程和完整原始评分journal，以及CPU/Python/NumPy RNG。复用205的位/类型存储和raw/Adam/shadow检查，但不调用205不兼容的上下文验证器、构造器或旧容器schema。旧203/204/205/206及历史门槛保持原样，没有重跑任何封存单位。

37项单位覆盖完整参数名字顺序、trainability、持久及非持久buffer、全部mode、已有grad/None、完整Adam ID/m-v-step/group options/defaults、EMA独立存储和计数、输入/日程/Adam/影子同一曝光、RNG、完整嵌套own seal及父树类型敏感比较和noalias；磁盘拒覆盖、SHA先于反序列化、完整磁盘位恢复；联合事务中的输入失败、半提交、合法评分后异常、权重/moments/grad/modes/buffer/defaults/RNG故障注入，部分load中途失败，以及回滚失败poison禁止继续。

新日程在同一个raw/Adam/EMA完整包内正确保存4750合法eligible改善后的stale0、best字典 `{step:4750,rank_gain_db:0.1}` 和anchor0.1；合成5000的0.11可更新best但不足原0.02耐心阈值，因此stale1、anchor仍0.1。未活动历史臂保留stale18。4750尚未评分时拒继续；5000硬预算与评分门槛保持，旧stopped_at4500及全部legacy不修改。这些评分是合成元数据，不是真实政策评分或新资格。

## 实际源状态与永久失败证据

最终套件仅读取封存205的零更新CPU存储一次，不重新读取原训练PT；构造一个实际22参数/27mode raw模型和一个匹配CPU Adam，用于存量加载而非step。E0、完整原Adam、原父4500/stale18/bestnull/anchornull/旧stop及源CUDA RNG字节全部保留。新产物 `results/mel_ema_live_cpu_state_monitor_20261004/source_live_cpu_container_4500.pt` SHA `fc34269e12ef5c54c170e3495d9d055c67e702ea3cf9c9a2d1d668ccab8ac5d1`，是零更新CPU存储，不是新训练checkpoint。后端由合成fixture供给，actual_audio_backend_verified=false；语义cursor/metadata被恢复，外国backend cache等副作用没有纳入事务。

首轮 chunk15d611 真实exit1，Ran0 tests：新未封存入口把原204 model=dict与205 tensors=OrderedDict直接做全树相等而拒绝，发生在实际Module/Adam构造前。失败日志attempt01保留。只读定位chunkcd1dfa exit0确认Adam/modes相同、22名字顺序相同、每张量逐位相同，差异仅映射接口容器类型。仅在尚未封存的新207中明确要求这两种实际类型、同名顺序及逐值位/类型比较；完整父和完整存储包仍分别保留自己的类型，不改旧文档、不抹seal、不放宽数值。

本轮封存205源存储读取总计3：首轮失败入口1、独立只读定位1、最终套件1；原训练PT反序列化0。实际raw模型和匹配实际CPU Adam构造各1，另有新小型合成CPU Adam夹具，不能说Adam构造0。最终套件外层CPU/Python/NumPy RNG恢复；所有源接口文件及旧43绑定已只读确认不变。禁止API检查在执行前拦截，不计为计算。

## 仍未完成

真实音频backend认证和其全部frontend/损失/矩阵/输入锁/cache依赖完整阅读、真实零更新音频审计、最多3唯一4501..4503 CPU-CUDA实际机制、严格源CUDA FP32 runtime及源CUDA RNG、原194首个raw真实Adam更新逐位一致、真实非零EMA、完整CPU-CUDA轨迹磁盘重放及包含影子计数的全事务回滚均PENDING。207只有CPU完整状态存储能力，不是完整正式CUDA训练容器、trainer、训练CLI或训练审批。

下一步接真实后端与单raw/单Adam训练器，不套194双真实训练臂，EMA不反馈raw/Adam。完整实际gate及资源/no duplicate全通过后才新协议审批、实现完整的封存training plan和一次WindowsPS5.1/WMI正式启动；目前无正式trainer/协议/审批/plan/worker，不启动GPU。报告102固定4501..5000/4750及5000/500共同更新/硬5000，原图FP32、LR1、kill32、完整aux.2、器乐4、分母6、原累计资格和所有旧stop不变。

动态CIM在07:35:23.3182715Z无仓库Python（也检查了相对路径207单位命令）；D盘余24.561473846435547GiB。只有工程进展，没有新音质、容量或平台期结论。旧197failed/199verify1/201prepare1永远留证，202保持关闭。人听和独立真实验收PENDING，NONRELEASE/release NONE。20分钟同聊巡检保持ACTIVE，仅重要新结果/失败/人工处理通知。
