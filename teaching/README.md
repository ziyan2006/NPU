# NPU 时钟实验室

基于本仓库 RTL 的教学网页：选择指令，逐时钟查看硬件模块、数据流、握手波形和计算结果。当前提供 **8 条已实现指令、14 个指令实例、1,596 个真实时钟周期**。

![教学页面预览](preview.png)

## 打开页面

**最方便的方式：下载 [npu-lab.html](https://raw.githubusercontent.com/ziyan2006/NPU/work/teaching/npu-lab.html)，保存为 `.html` 文件，然后用浏览器打开。** 如果链接显示源码，可以右键链接选择“另存为”。这个单文件包含全部样式、脚本与轨迹，演示本身不需要联网。

已经下载仓库时，也可以打开本目录的 `index.html`。需要通过本地 HTTP 访问时，在仓库根目录执行：

```bash
python -m http.server 8000
```

然后访问 <http://localhost:8000/teaching/>。GitHub 的文件预览只显示源码，不会运行网页。

## 建议的学习顺序

1. 默认选择 `CONV2D`，使用“下一关键沿”观察描述符读取、参数预取、A/W 读响应、MAC、后处理、FIFO 与 O0 写回。
2. 用“下一时钟”检查连续的 MAC 流水级，以及后处理按通道执行的多个状态。
3. 切换 `DMA_LOAD` 的“输入 → A0”，查看地址生成、填充清零和 DDR 搬运；再选“权重 → W0”，观察 CP 的派发等待。
4. 查看 `WAIT`，理解异步指令退休与完成事件的区别。
5. 查看 `VEC_ADD`、`DMA_STORE` 和 `UPSAMPLE2X`，把整个小任务串起来。

点击模块可查看用途、接口宽度和 RTL 源码。框图支持缩放；波形支持 16 / 32 / 64 周期窗口，点击波形即可跳到对应周期。可导出当前框图 SVG 和所选实例的原始 CSV 轨迹。

键盘：左右箭头单步，空格播放 / 暂停，`N` 跳到下一关键沿。播放始终逐周期推进；跳转按钮明确跳过中间周期，时间轴保留真实周期编号。

## 演示任务与支持指令

张量采用 NHWC8，每个像素 8 通道；激活以 INT16 容器保存 INT12。输入通道 0 为 3，其余为 0。1×1 卷积仅 `weight[0,0]=2`，其余权重和偏置为 0，量化比例为 1，无激活。卷积结果通道 0 为 6；加上输入残差后为 9；写回 DDR 并上采样得到 2×2 的四个像素，每个都是 `[9,0,0,0,0,0,0,0]`。

| 指令 | opcode | 行为 |
|---|---|---|
| NOP | `0x00` | 直接退休 |
| WAIT | `0x01` | 等待全部指定事件，不清除事件 |
| END | `0x03` | 等所有单元空闲，结束任务 |
| DMA_LOAD | `0x10` | 异步 DDR → A / W |
| DMA_STORE | `0x11` | 异步 O → DDR，等待 AXI B 响应 |
| CONV2D | `0x20` | 异步卷积与融合后处理 → O |
| VEC_ADD | `0x30` | 同步残差重定量、饱和相加 |
| UPSAMPLE2X | `0x42` | 同步 DDR → 行缓冲 → DDR |

`ACT` 和 `REQUANT` 虽然能通过 CP 的 opcode 解码，当前执行前端没有独立实现，不能作为支持指令展示。激活与重定量通过卷积的融合后处理实现。枚举中的其他预留指令也未加入选择器。

完整任务：`NOP → 5×DMA_LOAD → WAIT(A0_READY|W0_READY) → CONV2D → WAIT(C0_DONE) → VEC_ADD → DMA_STORE → WAIT(S0_DONE) → UPSAMPLE2X → END`。

## 周期与高亮的含义

- 所有信号在**上升沿前**采样。`VALID=1 && READY=1` 表示数据在当前上升沿被接收，青色路径只标记这个沿实际发生的传输。
- 橙色模块表示仍在处理；粉色虚线表示 `VALID=1 && READY=0`。灰色有效数据不代表本沿发生传输。
- `K+0` 是所选指令被 CP 接收的沿。绝对周期从复位释放后的第一个上升沿开始计数，包含前面的 CSR 配置和任务装载。
- 每个实例保留完整小任务的上下文。异步指令派发后，取指、下一条 DMA 或 WAIT 可以同时出现。没有重排、补造或删除等待周期。
- A0 / W0 / O0 数值显示截至当前沿已观察到的通道 0 数据；DDR 栏显示已经握手发送的通道 0。其余 7 通道在本例均为 0。
- MAC valid 显示进入当前沿前的寄存器状态。无停顿时，输入在 K 沿接收，K+4 沿产生结果，因此上升沿前采样在 K+5 才看到 result_valid。后处理、写回与事件到达还有后续周期。

轨迹来自 `npu_top` 与仓库已有的 `tb_npu_top_task.sv` AXI 内存模型。**这里的延迟是这个固定仿真任务和内存模型的结果，不能直接当作实板 DDR 延迟或整模型性能。** 页面是记录回放，不支持在线改变张量尺寸或量化参数后重新仿真。

教学框图保留了 [NPU 原始内部框图](../框图/03_npu_internal_io.md) 的模块与接口关系，并展开了部分内部寄存缓冲。事件寄存器属于 CP，结果缓冲属于 CONV pipeline，写回 FIFO 属于 DMA subsystem；这些在图中单列是为了教学。AXI 固定 ID / PROT 等旁带字段未逐一展开，完整 59 个顶层端口见原框图的 I/O 表。

## 重新生成与验证

运行页面不需要安装工具。重新生成轨迹需要 Python 3 和 Icarus Verilog 13；浏览器验证需要 Python Playwright 和 Chromium。

```bash
python teaching/generate_trace.py
python teaching/build_offline.py
python teaching/check_page.py
```

`generate_trace.py` 创建临时小任务，使用已有 RTL 文件清单与完整任务 testbench，验证 14 条指令退休、正常完成与最终 DDR 输出后才写入 `trace.js`。Icarus 无法直接处理源码中的 7 处 enum 三目赋值，脚本只在临时副本中改写为等价 `if/else`，不修改仓库 RTL。源文件 SHA-256、信号层级路径、状态枚举和采样定义均保存在轨迹文件中。

若使用解包的 Icarus，可传入 `--iverilog`、`--vvp` 与 `--ivl-base`；`--keep` 可保留仿真文件供检查。修改 `index.html`、`style.css`、`app.js` 或 `trace.js` 后，应重新运行 `build_offline.py` 更新单文件版本。
