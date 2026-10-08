"""Draw the implemented NPU internals and document every top-level port."""
from pathlib import Path
import re

from render_hardware_diagrams import Diagram, BLUE, RED, GRAY

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def internal():
    d = Diagram("03_npu_internal_io", 2600, 2160)
    d.text(1300, 48, "NPU 内部硬件框图与顶层 I/O", 36)
    d.text(1300, 103, "对应当前 npu_top / npu_core RTL · AXI 通道方向均以 NPU 为参照", 22, GRAY)
    d.box(390, 175, 1760, 1795, "npu_top：NPU IP 边界", group=True, size=24)
    d.box(420, 465, 1690, 1450, "npu_core：任务执行核心", fill="#fcfdff", dashed=True, group=True, size=21)

    d.box(30, 225, 330, 480,
          "AXI4-Lite 从接口\ns_axi_ctrl_*\n\n输入 NPU：\nAWADDR[11:0], AWPROT[2:0], AWVALID\nWDATA[31:0], WSTRB[3:0], WVALID\nBREADY\nARADDR[11:0], ARPROT[2:0], ARVALID\nRREADY\n\n输出 NPU：\nAWREADY, WREADY, BRESP[1:0], BVALID\nARREADY, RDATA[31:0], RRESP[1:0], RVALID",
          fill="#faeaea", size=16)
    d.box(430, 240, 450, 170, "控制 / 状态寄存器 · npu_csr\n任务地址、长度、tag、watchdog\ndoorbell / IRQ / 错误 / 性能计数器\nCSR 窗口：4 KiB", fill="#faeaea", size=20)
    d.line([(360, 325), (430, 325)], RED, double=True)
    d.box(30, 765, 330, 80, "输出：irq_o（1 bit）\n完成 / 错误 / watchdog 中断", fill="#faeaea", size=19)
    d.line([(430, 380), (390, 380), (390, 805), (360, 805)], RED)
    d.box(30, 940, 330, 115, "输入：aclk（100 MHz）\n输入：aresetn（低有效复位）\nCSR 与 core 共用", fill="#eef0f3", size=19)
    d.line([(360, 997), (390, 997)], GRAY, dashed=True)
    d.text(194, 1130, "IN / OUT 表示信号方向\n双向连线合并请求与反馈\n握手：valid && ready\n在时钟上升沿完成传输", 19, GRAY)

    d.box(1040, 240, 1010, 150,
          "状态 / 错误 / 性能统计汇总（npu_core 内逻辑）\nbusy / done / error；错误 code[15:0]、PC[31:0]、tag[15:0]\n完成 tag[31:0]、retired[31:0]；周期 / 字节计数器各 64 bit",
          fill="#f9eeee", size=20)
    d.line([(1040, 315), (880, 315)], RED)
    d.text(960, 290, "状态返回", 17, RED)
    d.text(1545, 420, "执行单元的 busy / done / error / event 汇入控制逻辑；部分反馈连线省略", 18, RED)

    d.box(450, 540, 260, 135, "任务生命周期控制\nIDLE / LOAD / RUN\nPREPARE / RECOVER", fill="#faeaea", size=19)
    d.line([(550, 410), (550, 540)], RED)
    d.text(675, 445, "task_base[63:0]；bytes/tag/watchdog[31:0]；start/reset", 15, RED)
    d.box(770, 540, 260, 135, "任务加载器\nnpu_task_loader\nheader 校验 / section 基址\nLUT 预载 [L]", fill="#edf4fa", size=18)
    d.box(1080, 540, 270, 135, "指令读取与寄存级\nnpu_command_fetch\n128-bit 指令 + 32-bit PC", fill="#edf4fa", size=18)
    d.box(1400, 540, 290, 135, "命令处理器\nnpu_command_processor\n译码 / 分派 / WAIT / END\nevent / watchdog", fill="#faeaea", size=18)
    d.box(1760, 540, 315, 135, "执行前端 + 描述符缓存\nnpu_execution_frontend\nCONV / VEC / UPSAMPLE\n4-line descriptor cache", fill="#faeaea", size=17)
    d.line([(710, 606), (770, 606)], RED)
    d.line([(1030, 595), (1080, 595)], RED)
    d.line([(1350, 595), (1400, 595)], BLUE)
    d.line([(1690, 595), (1760, 595)], RED)
    d.text(1090, 513, "loader_done 启动取指与命令处理", 16, RED)
    d.text(1570, 690, "compute / vector：128-bit 指令", 15, RED)
    d.line([(1500, 675), (1500, 707), (430, 707), (430, 790), (480, 790)], RED)
    d.text(657, 689, "DMA：128-bit 指令 + 32-bit PC", 16, RED)

    d.box(480, 740, 470, 100, "DMA 前端 + 描述符缓存 + AGU\nnpu_dma_frontend（4-line cache）\n地址 / stride / 容量检查", fill="#edf4fa", size=19)
    d.box(1060, 740, 480, 100, "四客户端共享块请求仲裁\nnpu_memory_arbiter4\n地址 64 bit / 长度 7 bit / 返回 512 bit", fill="#edf4fa", size=19)
    d.box(1650, 740, 380, 100, "AXI 块读取器\nnpu_axi_block_reader\nAXI 64-bit → block 512-bit", fill="#edf4fa", size=18)
    d.line([(900, 675), (900, 723), (1140, 723), (1140, 740)], BLUE, double=True)
    d.line([(1215, 675), (1215, 740)], BLUE, double=True)
    d.line([(1870, 675), (1870, 718), (1480, 718), (1480, 740)], BLUE, double=True)
    d.line([(950, 790), (1060, 790)], BLUE, double=True)
    d.line([(1540, 790), (1650, 790)], BLUE, double=True)
    d.text(1290, 889, "四个客户端：Loader / Fetch / DMA descriptor / Execution descriptor", 17, BLUE)

    d.box(480, 1010, 470, 120, "DMA 引擎 · npu_dma_engine\nDMA_LOAD / DMA_STORE / ZERO\nAXI burst / 4 KiB 拆分 / 错误检查", fill="#e5f2fa", size=20)
    d.line([(715, 840), (715, 1010)], RED)
    d.text(735, 922, "DMA request / ready", 17, RED, "left")
    d.box(1650, 1010, 380, 120, "三客户端 AXI 读仲裁\nnpu_axi_read_arbiter3\nBlock reader / DMA / Upsample", fill="#e5f2fa", size=19)
    d.line([(1840, 840), (1840, 1010)], BLUE, double=True)
    d.line([(950, 1055), (1650, 1055)], BLUE, double=True)
    d.text(1230, 1034, "DMA：AR / R，64-bit 数据", 17, BLUE)
    d.line([(920, 1010), (920, 958), (1595, 958), (1595, 1645), (1650, 1645)], BLUE, double=True)
    d.text(1230, 939, "DMA：AW / W / B", 17, BLUE)

    d.box(480, 1310, 470, 405, "Scratchpad · npu_scratchpad\n6 bank / 224 KiB / 56 BRAM36", fill="#fffaf0", group=True, size=20)
    d.box(510, 1390, 410, 85, "A0 / A1：各 64 KiB\n输入激活；CONV 读口 128 bit", fill="#fff1dc", size=19)
    d.box(510, 1495, 410, 85, "W0 / W1：各 32 KiB\n权重 / bias / quant；读口 512 bit", fill="#fff1dc", size=18)
    d.box(510, 1600, 410, 85, "O0 / O1：各 16 KiB\n输出激活；CONV 写口 128 bit", fill="#fff1dc", size=19)
    d.line([(715, 1130), (715, 1310)], BLUE, double=True)
    d.text(735, 1205, "DMA 读 / 写口：64 bit\n地址 32 bit / strobe 8 bit", 17, BLUE, "left")

    d.box(1040, 1190, 510, 515, "CONV2D 计算通路", fill="#f7fcf8", dashed=True, group=True, size=20)
    d.box(1080, 1250, 430, 95, "卷积控制器 + 8×8 MAC\n地址循环 / mask / first / last\nINT16 × INT8 → 8 × INT32", fill="#e6f3ea", size=19)
    d.box(1080, 1440, 430, 105, "Requant / Post · npu_requant_post\nbias + Q31 / RNE + clamp / activation\ntanh LUT 预载：[L] data 16 / addr 12 bit", fill="#e6f3ea", size=17)
    d.box(1080, 1610, 430, 65, "2-entry × 128-bit 输出写回 FIFO", fill="#edf4fa", size=19)
    d.line([(1295, 1345), (1295, 1440)], BLUE)
    d.text(1315, 1390, "256 bit（8 × INT32）", 17, BLUE, "left")
    d.line([(1295, 1545), (1295, 1610)], BLUE)
    d.text(1320, 1580, "128 bit（8 × INT16）", 16, BLUE, "left")
    d.line([(920, 1423), (1000, 1423), (1000, 1275), (1080, 1275)], BLUE)
    d.text(993, 1240, "A：128 bit", 17, BLUE)
    d.line([(920, 1530), (1020, 1530), (1020, 1320), (1080, 1320)], BLUE)
    d.text(980, 1475, "W：512 bit", 17, BLUE)
    d.line([(1020, 1530), (1020, 1490), (1080, 1490)], BLUE)
    d.line([(1080, 1642), (920, 1642)], BLUE)
    d.text(994, 1620, "O：128 bit", 17, BLUE)
    d.box(1060, 1780, 490, 100, "VEC_ADD · npu_vec_add\nResidual requant / 饱和相加 / 原位写 O\nScratchpad vector 读 / 写端口各 512 bit", fill="#e6f3ea", size=18)
    d.line([(950, 1675), (985, 1675), (985, 1830), (1060, 1830)], BLUE, double=True)
    d.text(715, 1800, "bank / kind / address / byte strobe\nCONV：A/W 独立读取，O 写回\nVEC：共享 vector 端口", 18, BLUE)

    d.box(1650, 1310, 380, 140, "UPSAMPLE2X\nnpu_upsample2x\n最近邻 2× / 1 KiB 行缓冲\n直接在 DDR 张量之间读写", fill="#e6f3ea", size=19)
    d.line([(1840, 1310), (1840, 1130)], BLUE, double=True)
    d.text(1860, 1215, "AR / R", 17, BLUE, "left")
    d.box(1650, 1600, 380, 105, "两客户端 AXI 写仲裁\nnpu_axi_write_arbiter2\nDMA / Upsample", fill="#e5f2fa", size=19)
    d.line([(1840, 1450), (1840, 1600)], BLUE, double=True)
    d.text(1860, 1525, "AW / W / B", 17, BLUE, "left")
    d.line([(1960, 675), (1960, 698), (2090, 698), (2090, 1830), (1550, 1830)], RED)
    d.line([(2090, 1280), (1510, 1280)], RED)
    d.line([(2090, 1370), (2030, 1370)], RED)
    d.text(2070, 935, "执行启动\n与描述符", 16, RED)
    d.text(1790, 1780, "指令 128 / descriptor 512 bit\nCONV bank 选择 / event[7:0]\nUPSAMPLE：src/dst 各 512 bit", 17, RED)

    d.box(2200, 225, 365, 340, "外部连接说明\n\n控制口 → PS GP0\n内存口 → 协议适配 → PS HP0\nIRQ → PS IRQ_F2P\n\n模型输入 / 输出均存放在 DDR\n通过 AXI 读入、计算、写回\n当前无独立 PCM / AXIS 音频口", fill="#f8fafc", size=19)
    d.box(2200, 685, 365, 560,
          "AXI4 主接口：读通道\nm_axi_mem_*\n\nAR 输出：\nARID[3:0], ARADDR[63:0]\nARLEN[7:0], ARSIZE[2:0]\nARBURST[1:0], ARLOCK\nARCACHE[3:0], ARPROT[2:0]\nARQOS[3:0], ARVALID\nAR 输入：ARREADY\n\nR 输入：\nRID[3:0], RDATA[63:0]\nRRESP[1:0], RLAST, RVALID\nR 输出：RREADY", fill="#e5f2fa", size=18)
    d.line([(2030, 1040), (2200, 1040)], BLUE)
    d.text(2115, 1014, "AR 地址 →", 17, BLUE)
    d.line([(2200, 1100), (2030, 1100)], BLUE)
    d.text(2115, 1140, "← R 数据", 17, BLUE)
    d.box(2200, 1390, 365, 555,
          "AXI4 主接口：写通道\nm_axi_mem_*\n\nAW 输出：\nAWID[3:0], AWADDR[63:0]\nAWLEN[7:0], AWSIZE[2:0]\nAWBURST[1:0], AWLOCK\nAWCACHE[3:0], AWPROT[2:0]\nAWQOS[3:0], AWVALID\nAW 输入：AWREADY\nW 输出：WDATA[63:0], WSTRB[7:0]\nWLAST, WVALID；输入 WREADY\nB 输入：BID[3:0], BRESP[1:0], BVALID\nB 输出：BREADY", fill="#e5f2fa", size=17)
    d.line([(2030, 1630), (2200, 1630)], BLUE)
    d.text(2115, 1604, "AW / W →", 17, BLUE)
    d.line([(2200, 1675), (2030, 1675)], BLUE)
    d.text(2115, 1720, "← B 响应", 18, BLUE)
    d.text(1300, 2020, "蓝色：数据 / 存储请求与响应    红色：控制 / 地址 / 状态    灰色虚线：时钟与复位", 21, GRAY)
    d.text(1300, 2070, "AXI 箭头按有效载荷方向标示，READY 反向。INT12 激活按 INT16 存储。[L] 为同名 LUT 预载连接。", 19, GRAY)
    d.text(1300, 2110, "功能展开图：DMA 前端、引擎、Scratchpad、CONV 与 FIFO 同属 npu_dma_subsystem；MAC 实例位于卷积控制器内。", 18, GRAY)
    d.save()


def io_document():
    text = (REPO / "hardware/rtl/npu_top.sv").read_text()
    header = text.split("module npu_top (", 1)[1].split(");", 1)[0]
    ports = re.findall(r"\b(input|output)\s+logic\s*(?:\[(\d+):(\d+)\]\s*)?(\w+)", header)
    groups = [("时钟、复位、中断", lambda n: not n.startswith(("s_axi_", "m_axi_"))),
              ("AXI4-Lite 控制口", lambda n: n.startswith("s_axi_ctrl_")),
              ("AXI4 内存读口 AR / R", lambda n: n.startswith(("m_axi_mem_ar", "m_axi_mem_r"))),
              ("AXI4 内存写口 AW / W / B", lambda n: n.startswith(("m_axi_mem_aw", "m_axi_mem_w", "m_axi_mem_b")))]
    out = ["# NPU 内部硬件框图与 I/O", "", "本页描述当前 `hardware/rtl/npu_top.sv` 和 `npu_core.sv` 的实现。", "",
           "![NPU 内部硬件框图与 I/O](03_npu_internal_io.png)", "",
           "[SVG](03_npu_internal_io.svg) · [PDF](03_npu_internal_io.pdf) · [可编辑 drawio](03_npu_internal_io.drawio)", "",
           "## 顶层接口", "", "输入 / 输出方向均以 NPU 为参照。以下表格由顶层 RTL 端口声明直接生成。", "",
           "模型输入、权重和输出均位于 DDR；NPU 经 AXI4 主接口读写，不暴露独立的 PCM 或 AXI-Stream 音频接口。", "",
           "AXI4-Lite 是从接口，接收 PS 的控制请求；AXI4 是主接口，主动发起 DDR 访问。", "",
           "所有 AXI 通道都通过 VALID/READY 握手；READY 的方向与该通道有效负载和 VALID 相反。", ""]
    for title, predicate in groups:
        out += [f"## {title}", "", "| 信号 | 方向 | 位宽 |", "|---|---|---:|"]
        for direction, msb, lsb, name in ports:
            if predicate(name):
                width = int(msb) - int(lsb) + 1 if msb else 1
                out.append(f"| `{name}` | {'输入' if direction == 'input' else '输出'} | {width} |")
        out.append("")
    out += ["## 内部连接与位宽", "", "| 连接 | 载荷 / 行宽 | 说明 |", "|---|---|---|",
            "| CSR → Core | task base 64；bytes/tag/watchdog 各 32 bit | start / soft reset 为 1-bit 脉冲 |",
            "| Fetch → Command Processor | command 128；PC 32 bit | 经 core 内单级指令寄存 |",
            "| Command Processor → DMA / Execution Frontend | command 128 bit | DMA 与 compute/vector 为独立分派接口 |",
            "| Loader / Fetch / DMA descriptor / Execution descriptor → Memory Arbiter | address 64；bytes 7；response 512 bit | 四个客户端共享 Block Reader |",
            "| Execution Frontend → CONV / VEC | operator descriptor 512 bit | CONV 还含 A/W/O bank 选择和 event 8 bit |",
            "| Execution Frontend → UPSAMPLE | source / destination descriptor 各 512 bit | 上采样直接访问 DDR |",
            "| DMA Engine ↔ Scratchpad | data 64；address 32；write strobe 8 bit | A/W/O bank 读写 |",
            "| Scratchpad → CONV | A 128；W 512 bit | 独立 A/W 端口；W bank 也提供 bias/quant |",
            "| MAC → Requant/Post | 256 bit = 8 × INT32 | 64 个 INT16×INT8 乘法 lane |",
            "| Post → FIFO → O bank | 128 bit = 8 × INT16 | 2-entry 输出 FIFO，写回排空后产生完成事件 |",
            "| Scratchpad ↔ VEC_ADD | read/write data 各 512 bit；write strobe 64 bit | 有效向量可能只使用部分 byte lane |",
            "| Loader → CONV tanh LUT [L] | data 16；address 12 bit | 4096 项 LUT 预载；图中同名连接省略长连线 |",
            "", "## 读写仲裁", "", "- 读仲裁器有三个客户端：Block Reader、DMA、UPSAMPLE。",
            "- 写仲裁器有两个客户端：DMA、UPSAMPLE。",
            "- descriptor cache 分别位于 DMA Frontend 和 Execution Frontend，不是只有一个全局 cache。",
            "- busy/done/error/event 返回 core 控制逻辑；图中为避免遮挡省略部分反馈和 section 基址分发连线。",
            "", "## 数值与外部适配", "", "- INT12 是激活有效精度；实际 bank 存储和 MAC 输入使用 16 bit。",
            "- 数据端口地址是 64 bit；接 Zynq-7000 HP0 时经 AXI4→AXI3 适配。",
            "- ARID/AWID 固定为 0；RID/BID 被接受但不参与 core 响应匹配。",
            "- 控制口 AWPROT/ARPROT 被接受但不参与 CSR 权限译码。",
            "- aclk 当前为 100 MHz；aresetn 低有效，所有 NPU 功能逻辑同一时钟域。",
            "", "## 重新生成", "", "```sh", "python 框图/render_npu_internal.py", "```", ""]
    (ROOT / "03_npu_internal_io.md").write_text("\n".join(out), encoding="utf-8")
    print(f"Documented {len(ports)} RTL top-level ports")


if __name__ == "__main__":
    internal()
    io_document()
    print("NPU internal I/O figure: PNG / SVG / PDF / drawio")
