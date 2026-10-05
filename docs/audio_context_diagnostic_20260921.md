# FullStem 无欠载但仍有断续感：上下文诊断

## 范围与结论

用户 2026-09-21 17:26 日志：`stem=1`，`uf/of/dec_err/npu_err/codec_err/deadline_miss=0`，
FIFO 最低 12288，块最大耗时 61294 us，小于 4096/44100 = 92.880 ms。
这段观测没有记录到供数不足，不能据此声称听感验收通过。

已在同一首 SD 歌曲上复现：**16 帧任务每块清空网络时间上下文，改变了掩码并引入周期调制。**
这是一项已证实的算法实现缺陷，符合断续感的可能成因；尚未采集板上模拟输出，
不能把所有电流声、残余人声或断续感都归因于它。

## 复现

```powershell
& 'C:\Users\30519\.workbuddy\binaries\python\envs\stem-npu\Scripts\python.exe' scripts/105_diagnose_audio_context.py --seconds 30 --integer-blocks 8
python scripts/_test_audio_player.py
```

输出为忽略目录 `hardware/build/audio_context_diagnostic/` 中的 report.json 和 A/B WAV。
这些是电脑端计算结果，不是板上录音；不提交歌曲或音频到 Git。

- 歌曲 SHA256：`f9efc0d14e0c4239ae2bf28169b10e3d2b52a157033cd94eef3bb7669421d676`
- checkpoint SHA256：`cb16d333c372d9558ef563b9b4baa9b9f7f46f0ec72d5b71f9e26557a1abfb55`
- 固定相同 STFT（1024/256，初始零历史）、分析/合成滤波器、仿射掩码、LF44 保护，
  仅改变模型时间上下文；CUDA TF32 关闭，避免不同卷积形状的低精度误差污染比较。
- 模型为 bott2，训练 crop=256 帧；板端每次仅输入 16 帧，约 92.88 ms。

## 30 秒控制变量实验

掩码误差使用范围 [0,1]，统计声道 0/1、band 44..127，排除前 128 帧启动区。
边界跳变是相邻帧掩码绝对差均值，**不是波形断点、掉样率或分离质量分数**。

| 推理方式 | 相对连续推理的掩码 MAE | 16 帧边界跳变 |
| --- | ---: | ---: |
| 连续浮点 | 0 | 0.040323 |
| 每 16 帧重新开始 | 0.143829 | 0.146121 |
| 额外保留 16 帧输入历史 | 0.080091 | 0.089979 |
| 额外保留 32 帧输入历史 | 0.044239 | 0.068408 |
| 额外保留 64 帧输入历史 | 0.004519 | 0.043208 |
| 额外保留 96 帧输入历史 | 0.000000126 | 0.040323 |
| 逐层缓存、每次只计算新 16 帧 | 0.000000119 | 0.040323 |

重新开始的边界跳变约为连续推理的 **3.62 倍**。相同频谱重复输入时，
连续模型稳定后完全不变，而 16 帧重启仍产生边界跳变约 0.10235，说明不能仅用
音乐本身的节拍解释这项变化。重启周期约 10.77 Hz。

逐层缓存浮点原型最大掩码误差为 `2.861e-6`，已设置 `1e-4` 失败阈值。
它验证了保存历史的数学方案，**没有实现 NPU 的状态保存与读回**。

固定点任务解释器还检查了第 8..15 块真实输入：相对同样量化输入的浮点小窗模型，
全 128 band 掩码 MAE 为 0.0344..0.0447；这包含权重/内部激活量化等差异，
不是硬件错误的证明，也不能与上表不同 band 范围直接作比例比较。
脚本另对恒定频谱运行真实编译任务，报告重复块首尾跳变。
其 INT12 首尾跳变为 0.096404，块内相邻帧跳变为 0.021331；因此周期变化也存在于
当前编译任务的整数参考中，并非只在浮点模型中出现。
本段浮点重建中 vocal/residual 超出 [-1,1] 的比例均为零；不能推广到所有歌曲。
没有这首歌的真实分轨，不能宣称人声指标提升多少 dB。

## 下一步：状态化任务，不重复计算大窗口

不直接改成 112 帧输入后保留最后 16 帧：这相当于约 7 倍窗口计算量，
现有单块 NPU 33.7 ms，无足够实时时间预算。也不以提高减法增益替代上下文修复。

1. 编译/参考模型添加逐层 causal halo 状态：10 个时间卷积各保存输入末尾 2 帧。
   16 帧提交保持所有 stride-2 相位对齐；新文件/复位清零，普通块不能清零。
2. 历史按 INT16 存储，未按 lane 补齐为 189440 bytes；NHWC8 后约 **192512 bytes
   （188 KiB）**。双缓冲约 376 KiB，优先考虑 DDR，不声称这些空间能全部塞入剩余 BRAM。
   以上不包含描述符、对齐额外开销与中间激活。
3. 为历史设计独立生命周期，防止 activation arena 重用覆盖；量化尺度必须与对应
   卷积输入一致，尤其 concat、残差和多分支；状态 DMA 后才能声明块完成。
4. 先验证多块 INT12 参考与连续量化参考逐元素一致，再做 RTL/DDR/缓存一致性验证；
   故障后不能把未完整更新的状态作为下个有效块使用。
5. 最后构建候选镜像：先 60 秒/长曲循环 `uf=of=deadline_miss=0`，再做同曲听感
   A/B 和有真实分轨的离线评价。额外 DMA 的时间开销尚未测量，不能沿用旧的 33.7 ms 签核。

## 同轮发现的独立旁路问题

仿射掩码修复后，`q=0` 意味着 mask=0.5，不再是静音。旧故障路径将输出清零，
且只在首次 NPU_ERROR 清空 vocal，后续锁定旁路可能继续送出 vocal 或 OLA 残留。

已改为故障输出填 `-2047`（零掩码），播放器在**每个**锁定旁路块强制 vocal=0。
回归测试令后端在旁路时故意返回非零残留，旧代码断言失败、修复后通过。
当前用户日志 `npu_err=0`，因此此修复**不能解释或解决当前正常路径的听感问题**。
本轮未部署新的 BOOT.BIN，现有板端仍是先前 affine-mask 版本。

本轮检查通过：完整 `_test_audio_player.py`、`41_check_preboard.py`、
`96_check_navigator_audio_soc.py` 和 `git diff --check`。未重建应用/镜像，
因此这些结果不代表新 C 代码已通过 Vitis 编译或上板验收。

## 后续推进：整数状态化与可编译候选任务

新增 `npu_streaming_reference.py` 与 `_test_npu_streaming.py`，两套不同执行方式比较：
一套按整层计算完整序列，一套沿用 tile 算术并注入历史。零输入、随机输入、恒定输入、
边界脉冲/饱和值和同曲前 256 帧（约 1.486 秒）均逐元素一致；覆盖四个输出通道。
复位、故障后拒绝续跑、历史提交原子性通过。故意不保留历史能触发差异，
故意重叠 activation 分配会被拒绝。此验证只覆盖有限测试序列，不是听感或实时时序验收。

随后实现 `106_compile_streaming_task.py`，不修改已签核的 generated 包，
只在 `hardware/build/streaming_task_candidate/` 生成实验产物：

- 为需要历史的物理 tensor 每行增加两列前缀，同一 tensor 的多个消费者共享历史。
- 写回/上采样使用原时间长度的 core view；卷积读取包含历史的 expanded view，
  `pad_left=0`。残差另用 `pad_left=2` 的向量描述符，以免把历史当成当前 skip。
- 每块成功结束后，ARM 把各 core view 的末两列复制到前缀；必须先更新前缀，再写入
  下一块 input。出错不继续，换文件/重新初始化清零整个状态。板端驱动尚未接入。
- 使用已有 ISA v1.0 指令，无新增 opcode。命令仍为 1869 条；计算输出 tile 数不增加。
- 新 task 2001856 bytes，activation 1097728 bytes；CPU 每块复制 147456 bytes（144 KiB）。
  此数小于逐层独立缓存的 188 KiB，因为共享 encoder/skip 源历史。
- NPU DMA payload 3216304 bytes（旧版 3020480）；容量、覆盖和调度冲突检查通过，
  但吞吐/ARM 拷贝/cache 维护成本尚未实测。

`_test_npu_streaming_task.py` 使用**未修改的 TaskReference**执行实际新任务，
不进行任何额外历史注入：零输入 48 帧、随机输入 128 帧、真实歌曲 256 帧均与
连续整数网络逐元素一致；不执行 ARM 前缀复制时，反例能检测到输出差异。

现有 UPSAMPLE2X RTL 拒绝带行间隔的 tensor。已把行跨度从“必须等于逻辑行长”改为
“不得小于逻辑行长”，仍保留对齐和逐行 allocation 越界检查。旧/新三条实际上采样指令
均通过 AXI RTL 仿真；过短行跨度和 allocation 均被拒绝。修改前新布局测试确实失败。

复现命令：

```powershell
python scripts/_test_npu_streaming.py --input-npy hardware/build/audio_context_diagnostic/input_int12.npy
python scripts/106_compile_streaming_task.py
python scripts/_test_npu_streaming_task.py --input-npy hardware/build/audio_context_diagnostic/input_int12.npy
python scripts/_test_npu_upsample.py --program hardware/build/streaming_task_candidate
python scripts/_test_npu_upsample.py
python scripts/_test_npu_top_task.py --program hardware/build/streaming_task_candidate --warmup-blocks 1
```

最后一项用参考计算的前一块生成非零历史，再将完整下一块交给 XSim。它验证的是
**一块带历史的完整 RTL 执行**，不是 ARM 驱动或实板的连续多块验证。
该项已通过：1869 条命令退休，32768 byte 输出逐字节一致；仿真总周期 3310030，
CSR 核心周期 3309989。此周期数不能代替实板 DDR、ARM、音频联合负载下的耗时。
候选 task SHA256：`a73e5471f0e5e5fc5fd442d7e384a13d7fbcdfabbc8dcbee0d74c38b2134bbee`。
新 RTL 还需重新综合/布局布线，旧 bitstream 的时序结果不能替代新版本签核。
当前 SD 镜像未替换，播放器仍未接入前缀复制、带 stride 的 input 写入及相关 cache 同步。

下一轮执行顺序：为生成的 streaming metadata/host copy 表增加 C 端绑定和主机测试；
让 resident session 按行写入输入、成功后 invalidate 历史源并复制前缀、clean 下次读取区，
失败锁定旁路；之后重新跑完整 RTL/软件门槛、生成新硬件和四模式镜像，再安排上板 A/B。
旧编译/调度/DMA/ISA header/task image 回归均通过，原有 generated 包未修改。

## 板端驱动接入（后续本轮）

已实现并通过主机 C 测试：

- metadata 区分紧凑输入字节数、实际跨度及行 stride；128 行分别复制，避免把下一行
  数据覆盖进历史列。原来紧凑输入/输出接口大小仍为各 32768 bytes。
- 首版 streaming 模式提交前 clean 全部 activation arena（1097728 bytes），完成后
  invalidate 全部 arena，再把当前尾部复制到历史前缀。下一次提交才 clean 新前缀。
  这是保守缓存方案；其额外 CPU 耗时需要上板观测，不能宣称没有开销。
- 普通旧任务仍只 clean 输入、invalidate 输出；init 从原始 payload 重新初始化。
- 错误码、完成 tag 不符、退休指令数不符、超时都锁定 session；失败不复制前缀，
  后续提交拒绝，必须重新初始化（真实硬件故障还需先恢复 NPU 状态）。
- 测试的 fake DDR 数据只在 invalidate 回调时变成 CPU 可见，用于检查历史不能提前
  复制；下一次 clean 检查历史内容。四类故障、状态清零和旧模式均通过。

生成器验证 copy 表与 core/history view 的 JSON、二进制 tensor descriptor 以及 task
镜像一致；漏项、重复项、改坏的源偏移及另一份 descriptor 均被拒绝。

Vitis 通过显式 `-TaskProgram` 选择任务，生成隔离的 payload，不覆盖原始 generated
模型包。构建回执记录实际导入源文件、ELF、task 和 XSA 的 SHA；打包/验收时要求
XSA 内 bitstream 与导出 bitstream 相同，ELF 内的上下文和任务 SHA 标记也必须匹配。
这些绑定新增了反例测试，避免软件/硬件/模型的旧新文件混用。

```powershell
python scripts/_test_streaming_payload.py
python scripts/_test_audio_player.py --case npu_session --task-program hardware/build/streaming_task_candidate --task-payload hardware/build/streaming_payload
python scripts/_test_audio_provenance.py
powershell -ExecutionPolicy Bypass -File scripts/102_build_navigator_audio_boot.ps1 -Clean -TaskProgram hardware/build/streaming_task_candidate
powershell -ExecutionPolicy Bypass -File scripts/103_test_navigator_audio_boot.ps1
```

新硬件构建开始前，原 BOOT.BIN、bitstream、XSA 已复制到
`hardware/build/sd_backups/before_streaming_20260921/`，未写入 SD 卡。
