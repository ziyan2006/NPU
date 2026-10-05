# 状态快照与哈希优化实测（2026-10-05）

本轮新增状态复制/哈希优化，并完成 CPU/CUDA 合成维护路径和完整 CPU 合成引擎对照。**完整 CPU 合成步骤耗时减少 31.9%；不是实际音频学生训练的整步加速结论。** 正式 4501..5000 已完成，没有重跑或自动替换封存的 221 worker。

## 实测结果

每项三轮，交替执行先后顺序，表内为中位数。比例为原版/候选版，减少量为 1−候选/原版。

| 路径 | 原版 | 候选版 | 比例 / 耗时减少 |
|---|---:|---:|---:|
| 完整 CPU 合成引擎一步，含事务前后完整验证 | 1.093412 s | 0.744653 s | 1.4684× / 31.90% |
| CPU 两次完整状态快照与 seal | 0.480796 s | 0.235437 s | 2.0421× / 51.03% |
| CUDA 两次完整状态快照与 seal | 0.475394 s | 0.207252 s | 2.2938× / 56.40% |
| CPU 合成维护路径：状态+缓存+日志 | 2.062572 s | 1.074123 s | 1.9202× / 47.92% |
| CUDA 合成维护路径：状态+缓存+日志 | 2.132166 s | 0.872972 s | 2.4424× / 59.06% |

最后两行包含此前 228 的 PCM 哈希优化，不能全归因于本次新增 232，也不能把各倍数相乘。CUDA 是实际设备上的合成张量复制/检查测试，不是学生前后向。GPU 与其他工作共享，样本有明显波动，三轮结果只支持本次局部基准，不保证未来固定倍数。

完整 CPU 引擎的三次原版为 `[1.2434701,1.0142120,1.0934123]` 秒，候选版为 `[0.7333017,0.7446528,0.7676350]` 秒。全部样本与分段时间保存在 234/236 benchmark.json。

## 采用与未采用的改动

232 将 `detach().cpu().clone()` 换为阻塞的、拥有独立存储的单次 CPU 复制，省掉 CUDA→CPU 后的额外 CPU clone。CPU→CPU 仍复制并拒绝 alias。状态哈希直接读取 contiguous byte memoryview，有限值按 262144 元素分块检查；bfloat16 等不能转 NumPy 的类型仍使用 Torch 检查。

每次仍检查每个元素、哈希每个字节。完整 raw、Adam moments/step/defaults/已有 grad、EMA/count、buffer/mode、游标/日程和 RNG 全部保留；事务前后两次完整快照和所有守卫没有删除。非张量元数据仍使用原 205 逐类型、逐顺序算法；有符号零、浮点类型、list/tuple、dict/OrderedDict 不混同。无 tensor-version/mtime/签名代替内容认证，无精度、原 Adam、EMA 数学或损失顺序修改。

235 为新私有命名空间重新编译原 205/207/212/219 定义，函数和类 AST 不改，仅替换状态 helper 绑定。原模块 globals/类对象未修改。新集成测试独立在相同 definition-only 编译环境核对原函数/访问器代码与所有 guard/transaction/update 定义。

**233 日志标量批量回传不采用。** 独立 54 标量计时：CPU 0.07326→0.29308 ms，CUDA 3.75774→3.83666 ms；本配置没有收益。其数学/顺序等价检查虽通过，但性能不通过，保留负结果，不接入 235。

235 目前只是 CPU 基准集成，不是新的正式训练 CLI/worker，也不是新的完整 activation 绑定。不得向其提供或借用旧正式 activation。未来实际 worker 必须完整绑定这些新 helper，并通过适用的真实输入、CUDA 数学/事务/磁盘机制及新有界审批；本次不制造这项授权。

## 一致性、回滚与真实暴露

234 用 22 片、824900 个合成参数元素，加 raw/m/v/grad/shadow、父状态和 RNG、224 MiB 的 7 片缓存 PCM。每样本两次完整快照、四次全缓存检查、六组日志。每设备的全部原版/候选 before/after typed seal、缓存身份和标量结果一致，无 alias，恢复通过。12 个手动合成状态变换不是 Adam 更新。该基准确实初始化 CUDA，不能声称本轮所有 CUDA 操作为零。

236 使用**新合成模型和 toy loss，不是音频学生的网络或音乐输入**，但执行原 219 完整事务、六次顺序反传、实际 CPU Adam 原函数、EMA、计数/游标和全组回滚。三轮 A/B 全部 raw/Adam/grad/EMA/上下文/CPU-Python-NumPy RNG 位一致，两个完整 commitment 分别为：

- 初态 `7d1f1a6e13fd506e488835e6fcea4c3c8340b539d76e859f9a391fee38ba404a`。
- 更新后 `ab59a1a1dcd7aee1b7c011ac97764836dcc88b703d0811c7b21189bd898cbead`。

额外一个新合成 Adam/EMA 更新后的 KeyboardInterrupt，消费三类 RNG 后，完整 raw/Adam/EMA/count/游标/RNG 回滚逐位恢复。成功基准共 42 个实际合成前后向、7 次实际合成 CPU Adam step（6 提交、1 回滚），无 CUDA 初始化。另失败初始化构造 1 个合成模型/CPU Adam，无前向或 step。**真实学生源 PT 反序列化、真实音乐/decoder、新学生更新/DEV 均 0。** 未新增完整真实学生磁盘重放或 CUDA 事务机制。

完整 CPU 一步用外部墙钟包围 `update_next`，包括原内部 seconds 字段没有覆盖的事务尾部验证。这个结果比纯 helper 测量更接近引擎维护成本，但仍未覆盖真实网络/损失、native decoder/source lease、缓存守卫、DEV 或 checkpoint I/O。不能用它预测已完成 500 步的节省时间。

## 新执行及失败保留

| 执行 | 结果 | 实际 exec chunk |
|---|---|---|
| 232 新 CPU 单位 | 17 项，native0 | e37427 |
| 233 新 CPU stub 单位 | 6 项，native0 | 123d48 |
| 234 合成 CPU/CUDA 维护基准 | 三轮/设备，native0 | start516a51/session7192，completion3747c2 |
| 236 attempt01 | 初始化漏 `synthetic_fixture:true`，native1，无前向/更新 | 251d1c |
| 236 attempt02 | 完整 CPU 三轮与实际合成故障回滚，native0 | starteaec55/session94561，completionc31079 |
| 235 static attempt01 | 原模块/definition-only 编译提示不同，4 通过/1 失败，native1 | 013b70 |
| 235 static attempt02 | 新测试把 property 当 function，4 通过/1 error，native1 | 841ec3 |
| 235 static attempt03 | 5 项全部通过，native0 | 24f072 |

最终 28 项新测试通过，不是重复旧封存单位。两次静态测试修复仅作用于当时未封存的新测试：同编译环境比较、property 使用 fget；没有跳过方法或降低数值阈值。236 只修当时未封存的新夹具声明，旧引擎守卫未改。全部失败日志、原因、原失败工具 SHA 保留。所有退出来自前台 exec/session 完整消费 native exit；非 WMI，没有 detached_exit。

## 当前状态

只读正式 completion 仍为 5000/500 追加更新，最终 PT SHA `d2be133da5bdd778fb615ef28167bf263e130a819596c6630e8c3a8726c1ab91`。本轮没有切换正式 worker、借旧审批、重放已完成预算或产生新音质结论。原 20 tracked dirty、469 插入/44 删除保持。12:55:39 UTC 动态检查无仓库 Python worker，D 约 24.13 GiB，GPU 空余 2584 MiB/利用率 93%（共享任务，不视为故障）。未修改巡检启停状态。

NONRELEASE、release NONE、人听与独立真实验收仍 PENDING。本轮详细只读归档：`results/ema237_state_speed_archive_20261005/review.json`；该归档只验证文件、各自 seal、样本统计和范围，不再次计算模型或重跑基准。
