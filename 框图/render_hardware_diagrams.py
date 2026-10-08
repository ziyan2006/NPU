"""Render project hardware figures and editable diagrams.net sources.

Coordinates are shared by the PNG/SVG/PDF and drawio exports. This is a
functional architecture view, not a replacement for the RTL netlist.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

from fontTools.ttLib import TTCollection

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.path import Path as MPath
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parent
# Matplotlib opens face 0 of a TTC by default: this collection starts with JP.
# Extract the SC face by family name so all renderers use simplified Chinese.
_FONT_DIRECTORY = TemporaryDirectory(prefix="npu-diagram-font-")
_SC_FONT_PATH = Path(_FONT_DIRECTORY.name) / "NotoSansCJKsc-Regular.otf"
_collection = TTCollection("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", lazy=True)
try:
    _sc_font = next(font for font in _collection.fonts
                    if font["name"].getDebugName(1) == "Noto Sans CJK SC")
    _sc_font.save(_SC_FONT_PATH)
finally:
    _collection.close()
FONT = FontProperties(fname=str(_SC_FONT_PATH))
if FONT.get_name() != "Noto Sans CJK SC":
    raise RuntimeError("The diagram font must be Noto Sans CJK SC")
# Outline SVG text to preserve the selected glyphs on computers without Noto.
# Text remains editable in the diagrams.net exports; PDF embeds the font.
matplotlib.rcParams.update({"svg.fonttype": "path", "pdf.fonttype": 42})
BLUE, RED, GRAY = "#147db3", "#c94d4d", "#77818b"


class Diagram:
    def __init__(self, name, width, height):
        self.name, self.width, self.height = name, width, height
        self.fig, self.ax = plt.subplots(figsize=(width / 100, height / 100))
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.fig.patch.set_facecolor("white")
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.axis("off")
        self.xml = ET.Element("mxfile", host="app.diagrams.net")
        diagram = ET.SubElement(self.xml, "diagram", name=name)
        model = ET.SubElement(diagram, "mxGraphModel", page="1", pageWidth=str(width), pageHeight=str(height))
        self.root = ET.SubElement(model, "root")
        ET.SubElement(self.root, "mxCell", id="0")
        ET.SubElement(self.root, "mxCell", id="1", parent="0")
        self.counter = 1

    def cell(self, **kwargs):
        self.counter += 1
        return ET.SubElement(self.root, "mxCell", id=str(self.counter), parent="1", **kwargs)

    def text(self, x, y, label, size=19, color="#253343", align="center", weight="normal"):
        self.ax.text(x, y, label, fontproperties=FONT, fontsize=size * .72,
                     ha=align, va="center", color=color, weight=weight, zorder=5,
                     linespacing=1.65)
        lines = label.count("\n") + 1
        w = max(160, max(map(len, label.splitlines())) * size)
        c = self.cell(value=label.replace("\n", "<br>"), vertex="1",
                      style=f"text;html=1;align={align};verticalAlign=middle;fontFamily=Noto Sans CJK SC;fontSize={size};fontColor={color};")
        ET.SubElement(c, "mxGeometry", x=str(x - w / 2 if align == "center" else x),
                      y=str(y - lines * size / 2), width=str(w), height=str(lines * size), **{"as": "geometry"})

    def box(self, x, y, w, h, label="", fill="#ffffff", stroke="#526475", dashed=False, group=False, size=20):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=stroke,
                                   linewidth=1.6, linestyle=(0, (5, 4)) if dashed else "-", zorder=0 if group else 2))
        style = f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;fontFamily=Noto Sans CJK SC;fontSize={size};fontColor=#253343;"
        if dashed:
            style += "dashed=1;"
        if group:
            style += "verticalAlign=top;spacingTop=12;align=left;spacingLeft=15;"
        c = self.cell(value=label.replace("\n", "<br>"), vertex="1", style=style)
        ET.SubElement(c, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
        if label:
            self.ax.text(x + 15 if group else x + w/2, y + 25 if group else y + h/2,
                         label, fontproperties=FONT, fontsize=size * .72,
                         ha="left" if group else "center", va="center", color="#253343",
                         linespacing=1.6, zorder=5)

    def line(self, points, color=BLUE, double=False, dashed=False, arrow=True, width=2.4):
        path = MPath(points, [MPath.MOVETO] + [MPath.LINETO] * (len(points)-1))
        self.ax.add_patch(FancyArrowPatch(path=path, arrowstyle="<->" if double else "->" if arrow else "-",
                                         mutation_scale=17, color=color, linewidth=width,
                                         linestyle=(0, (5, 4)) if dashed else "-", zorder=3))
        style = f"html=1;rounded=0;strokeColor={color};strokeWidth={width};endArrow={'classic' if arrow else 'none'};"
        if double:
            style += "startArrow=classic;"
        if dashed:
            style += "dashed=1;"
        c = self.cell(edge="1", style=style)
        geo = ET.SubElement(c, "mxGeometry", relative="1", **{"as": "geometry"})
        ET.SubElement(geo, "mxPoint", x=str(points[0][0]), y=str(points[0][1]), **{"as": "sourcePoint"})
        ET.SubElement(geo, "mxPoint", x=str(points[-1][0]), y=str(points[-1][1]), **{"as": "targetPoint"})
        if len(points) > 2:
            arr = ET.SubElement(geo, "Array", **{"as": "points"})
            for x, y in points[1:-1]:
                ET.SubElement(arr, "mxPoint", x=str(x), y=str(y))

    def save(self):
        for suffix in ("png", "svg", "pdf"):
            self.fig.savefig(ROOT / f"{self.name}.{suffix}", dpi=180, facecolor="white")
        svg = ROOT / f"{self.name}.svg"
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
        ET.indent(self.xml)
        ET.ElementTree(self.xml).write(ROOT / f"{self.name}.drawio", encoding="utf-8", xml_declaration=True)
        plt.close(self.fig)


def system():
    d = Diagram("01_system_hardware", 1920, 1390)
    d.text(960, 48, "Zynq-7020 人声分离 NPU · 系统硬件框图", 34)
    d.text(960, 96, "当前实现：PS 控制 + DDR 存储 + PL 可编程 CNN 加速器", 20, GRAY)
    d.box(330, 170, 1540, 1080, "Zynq XC7Z020", fill="#ffffff", stroke="#35485b", group=True, size=23)
    d.box(355, 225, 300, 980, "PS · 处理系统", fill="#f8fafc", dashed=True, group=True)
    d.box(680, 225, 1150, 980, "PL · 可编程逻辑", fill="#fcfdff", dashed=True, group=True)
    d.box(875, 285, 925, 890, "NPU IP：npu_top / npu_core", stroke="#4d8eaf", dashed=True, group=True, size=18)

    d.box(35, 570, 255, 180, "板载 DDR3L\n1 GiB / 32-bit\n任务 · 权重 · 输入 / 输出", fill="#fff1dc", size=20)
    d.box(390, 325, 230, 125, "ARM Cortex-A9\n任务提交 / 结果检查\n裸机 NPU 驱动", fill="#e8eef8")
    d.box(390, 580, 230, 150, "PS 内部互连\n与 DDR 控制器", fill="#e8eef8")
    d.box(390, 940, 230, 115, "FCLK0：100 MHz\nPS 复位输出", fill="#eef0f3")
    d.line([(505, 450), (505, 580)], double=True)
    d.text(525, 515, "存储器访问", 17, BLUE, "left")
    d.line([(290, 655), (390, 655)], double=True)
    d.text(340, 625, "DDR", 17, BLUE)

    d.box(950, 330, 240, 115, "CSR 控制 / 状态\nnpu_csr\n基址 0x43C00000", fill="#faeaea", size=19)
    d.box(1270, 330, 460, 115, "命令处理器 / 任务生命周期\n译码 · 分派 · event · watchdog", fill="#faeaea", size=20)
    d.line([(620, 375), (950, 375)], RED, double=True)
    d.text(780, 350, "GP0 → AXI4-Lite（32-bit）", 16, RED)
    d.line([(1190, 388), (1270, 388)], RED, double=True)
    d.line([(1030, 330), (1030, 270), (570, 270), (570, 325)], RED)
    d.text(750, 294, "IRQ_F2P：完成 / 错误中断", 15, RED)

    d.box(700, 570, 150, 190, "AXI 互连\n与协议适配\nAXI4 ↔ AXI3", fill="#edf4fa", size=17)
    d.box(915, 570, 220, 190, "共享 AXI 访问\n读 / 写仲裁器\n64-bit 数据接口", fill="#e5f2fa", size=19)
    d.line([(620, 670), (700, 670)], double=True)
    d.text(660, 642, "HP0", 16, BLUE)
    d.line([(850, 670), (915, 670)], double=True)
    d.text(882, 642, "AXI4", 14, BLUE)
    d.box(1190, 485, 540, 95, "任务 / 指令 / 描述符前端\nLoader · Fetch · Cache · Execution Frontend", fill="#edf4fa", size=19)
    d.line([(1500, 445), (1500, 485)], RED, double=True)
    d.line([(1025, 570), (1025, 533), (1190, 533)], double=True)
    d.text(1105, 512, "共享块读取", 16, BLUE)

    d.box(1190, 650, 245, 100, "DMA 前端 / 引擎\n描述符解析 · 地址生成\nDDR ↔ BRAM 搬运", fill="#e5f2fa", size=18)
    d.box(1480, 650, 250, 100, "UPSAMPLE2X\n最近邻 2× 上采样\n1 KiB 行缓冲", fill="#e6f3ea", size=18)
    d.line([(1305, 580), (1305, 650)], RED)
    d.line([(1605, 580), (1605, 650)], RED)
    d.line([(1135, 705), (1190, 705)], double=True)
    d.line([(1080, 760), (1080, 790), (1605, 790), (1605, 750)], double=True)
    d.text(1550, 776, "直接访问 DDR", 16, BLUE)

    d.box(1190, 845, 540, 105, "片上 Scratchpad · BRAM（224 KiB）\nA0/A1：各 64 KiB    W0/W1：各 32 KiB    O0/O1：各 16 KiB", fill="#fff1dc", size=18)
    d.line([(1305, 750), (1305, 845)], double=True)
    d.text(1325, 810, "64-bit DMA 端口", 16, BLUE, "left")
    d.box(1190, 1030, 300, 105, "CONV2D 计算通路\n控制器 + 8×8 MAC\n量化 / 激活 + 写回 FIFO", fill="#e6f3ea", size=18)
    d.box(1530, 1030, 200, 105, "VEC_ADD\n残差重定标\n饱和相加", fill="#e6f3ea", size=18)
    d.line([(1340, 950), (1340, 1030)], double=True)
    d.text(1210, 988, "A/W 读取 · O 写回", 15, BLUE)
    d.line([(1630, 950), (1630, 1030)], double=True)
    d.line([(1730, 545), (1770, 545), (1770, 990), (1455, 990), (1455, 1030)], RED)
    d.line([(1770, 990), (1695, 990), (1695, 1030)], RED)
    d.text(1760, 885, "执\n行\n控\n制", 16, RED)

    d.line([(620, 995), (740, 995), (740, 1155), (875, 1155)], GRAY, dashed=True)
    d.text(765, 1135, "时钟 / 复位", 16, GRAY)
    d.text(1065, 1156, "NPU 内部同一时钟域", 15, GRAY)
    d.line([(430, 1300), (500, 1300)], BLUE, double=True)
    d.text(515, 1300, "数据 / 存储访问", 18, BLUE, "left")
    d.line([(820, 1300), (890, 1300)], RED)
    d.text(905, 1300, "控制 / 状态 / 中断", 18, RED, "left")
    d.line([(1270, 1300), (1340, 1300)], GRAY, dashed=True)
    d.text(1355, 1300, "时钟 / 复位", 18, GRAY, "left")
    d.text(960, 1350, "功能框图：合并部分共享访问与握手信号。WM8960、I2S、STFT/iSTFT 与按键音频通路尚待系统集成。", 17, GRAY)
    d.save()


def convolution():
    d = Diagram("02_conv_hardware", 1740, 1260)
    d.text(870, 48, "NPU 卷积计算子系统 · 硬件展开框图", 32)
    d.text(870, 96, "对应 npu_dma_subsystem：BRAM 供数 → 8×8 MAC → 后处理 → 输出写回", 19, GRAY)
    d.box(40, 180, 1660, 980, "DMA / Scratchpad / CONV2D 子系统", dashed=True, group=True, size=22)
    d.box(420, 265, 930, 700, "卷积流水线：npu_conv2d_pipeline", fill="#fbfdff", dashed=True, group=True, size=19)
    d.box(80, 420, 250, 160, "输入特征存储\nA0 / A1\n各 64 KiB BRAM", fill="#fff1dc")
    d.box(80, 710, 250, 160, "权重 / 参数存储\nW0 / W1\n各 32 KiB BRAM", fill="#fff1dc")
    d.box(470, 320, 830, 95, "卷积控制器 · npu_conv2d_controller\n输出坐标 / 通道 / 卷积核循环 · A/W 地址 · first/last · lane mask", fill="#faeaea", size=19)
    d.box(470, 510, 370, 170, "8×8 Tensor MAC\nnpu_tensor_mac_8x8\n64 路 INT16 × INT8 乘法\n加法树 + INT32 累加", fill="#e6f3ea", size=20)
    d.box(950, 510, 350, 170, "重定标 / 激活\nnpu_requant_post\nbias + Q31 / RNE + clamp\nReLU / LeakyReLU / tanh LUT", fill="#e6f3ea", size=18)
    d.box(950, 765, 350, 115, "偏置 / 量化参数预取与寄存\n由 W bank 读取", fill="#edf4fa", size=19)
    d.box(1410, 540, 250, 110, "输出写回 FIFO\n2-entry × 128-bit", fill="#edf4fa", size=20)
    d.box(1410, 785, 250, 120, "输出存储\nO0 / O1\n各 16 KiB BRAM", fill="#fff1dc", size=20)
    d.box(80, 1030, 250, 85, "DMA 引擎\n连接 DDR / AXI", fill="#e5f2fa", size=19)

    d.line([(330, 500), (470, 550)], BLUE)
    d.text(395, 478, "128-bit A", 17, BLUE)
    d.text(660, 725, "A：8 × INT16 存储值（有效 INT12）\nW：8 × 8 × INT8", 17, BLUE)
    d.line([(330, 770), (380, 770), (380, 635), (470, 635)], BLUE)
    d.text(399, 613, "512-bit W", 17, BLUE)
    d.line([(840, 585), (950, 585)], BLUE)
    d.text(895, 548, "256-bit", 17, BLUE)
    d.text(895, 623, "8 × INT32", 16, BLUE)
    d.line([(1300, 595), (1410, 595)], BLUE)
    d.text(1355, 563, "128-bit", 17, BLUE)
    d.line([(1535, 650), (1535, 785)], BLUE)
    d.text(1555, 720, "8 × INT16", 17, BLUE, "left")
    d.line([(330, 825), (950, 825)], BLUE)
    d.text(635, 800, "bias / multiplier / shift / clamp 参数", 16, BLUE)
    d.line([(1125, 765), (1125, 680)], BLUE)
    d.line([(555, 415), (555, 510)], RED)
    d.text(570, 456, "首尾标记 / 有效控制", 16, RED, "left")
    d.line([(470, 350), (360, 350), (360, 445), (330, 445)], RED)
    d.line([(360, 350), (350, 350), (350, 735), (330, 735)], RED)
    d.text(355, 321, "A/W 读取地址", 16, RED)
    d.line([(1300, 365), (1380, 365), (1380, 560), (1410, 560)], RED)
    d.text(1450, 342, "结果地址 / lane mask", 16, RED)
    d.line([(885, 225), (885, 320)], RED, double=True)
    d.text(1100, 243, "启动 / 忙 / 完成 / 错误", 17, RED)
    d.line([(180, 1030), (180, 870)], BLUE)
    d.line([(180, 1030), (365, 1030), (365, 920), (65, 920), (65, 535), (80, 535)], BLUE)
    d.line([(1535, 905), (1535, 1070), (330, 1070)], BLUE)
    d.text(1000, 1044, "输出经 DMA 写回 DDR（64-bit 搬运端口）", 18, BLUE)
    d.text(870, 1195, "蓝色：数据通路    红色：控制 / 地址通路    所有模块共用时钟与复位（省略）", 18, GRAY)
    d.text(870, 1230, "注：MAC 位于卷积控制器内部；输出 FIFO 位于 DMA 子系统。完成事件需等待写回排空。", 16, GRAY)
    d.save()


if __name__ == "__main__":
    system()
    convolution()
    print("Rendered system and convolution diagrams: PNG, SVG, PDF, drawio")
