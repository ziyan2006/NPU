"use strict";
(() => {
  const T = window.NPU_TRACE;
  const $ = id => document.getElementById(id);
  const ns = "http://www.w3.org/2000/svg";
  const signals = Object.fromEntries(T.signals.map((s, i) => [s, i + 1]));
  const frames = T.rows;
  const raw = (f, k) => f?.[signals[k]] ?? null;
  const num = (f, k) => raw(f, k) === null ? null : parseInt(raw(f, k), 16);
  const one = (f, k) => num(f, k) === 1;
  const hsKey = (prefix, side) => signals[prefix + "_" + side] === undefined ? prefix + side : prefix + "_" + side;
  const fire = (f, prefix) => one(f, hsKey(prefix,"v")) && one(f, hsKey(prefix,"r"));
  const blocked = (f, prefix) => one(f, hsKey(prefix,"v")) && num(f, hsKey(prefix,"r")) === 0;
  const hex = (value, width = 2) => value === null ? "X" : "0x" + value.toString(16).padStart(width, "0");
  const valueHex = (f, k) => raw(f, k) === null ? "X" : "0x" + raw(f, k);
  const state = (f, k, type) => T.enums[type]?.[num(f, k)] ?? "X";
  const signedLow = (f, key, bits) => {
    const s = raw(f, key); if (s === null) return "X";
    const n = Number(BigInt("0x" + s) & ((1n << BigInt(bits)) - 1n));
    return n >= 2 ** (bits - 1) ? n - 2 ** bits : n;
  };
  const element = (tag, attrs = {}, text = "") => {
    const el = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
    if (text !== "") el.textContent = text;
    return el;
  };
  const instructionMeta = {
    NOP: ["空操作", "0x00", "flags = 0", "完成译码后直接退休，不启动运算单元。"],
    WAIT: ["等待事件", "0x01", "flags = 0", "imm[7:0] 是事件掩码；全部指定事件置位才继续。WAIT 不清除事件。"],
    END: ["结束任务", "0x03", "flags = 0 或 IRQ (0x02)", "等待执行单元全部空闲后结束；CSR 根据中断配置产生 irq_o。"],
    DMA_LOAD: ["加载片上数据", "0x10", "flags = ASYNC (0x01)", "DDR → A / W bank。描述符读取、地址生成、必要的填充清零都需要周期；派发时退休，搬运完成后置位事件。"],
    DMA_STORE: ["写回 DDR", "0x11", "flags = ASYNC (0x01)", "O bank → DDR。读片上数据后经过 AXI AW / W / B；完成事件晚于最后一次数据发送。"],
    CONV2D: ["卷积与融合后处理", "0x20", "flags = ASYNC | SATURATE | FUSED_POST (0x0d)", "从 A 读取 128 位激活、从 W 读取 512 位权重，产生 8 路 INT32 结果；偏置、逐通道重定量与可选激活后写回 O。"],
    VEC_ADD: ["残差相加", "0x30", "flags = SATURATE (0x04)", "用独立的后处理单元重定量 A 中的残差，再与 O 相加并饱和到 INT12。同步执行，CP 等待完成。"],
    UPSAMPLE2X: ["二倍最近邻上采样", "0x42", "flags = 0", "直接从 DDR 读取 NHWC8 数据，用行缓冲复制列与行，再写 DDR。经过读写仲裁器，不经过 Scratchpad。同步执行。"],
  };
  const caseLabels = ["空操作", "输入 → A0", "权重 → W0", "偏置 → W0", "卷积量化参数 → W0", "残差量化参数 → W0", "等 A0 / W0 就绪", "1×1 卷积 / 8 通道", "等 C0_DONE", "A0 + O0 → O0", "O0 → DDR", "等 S0_DONE", "1×1 → 2×2", "结束并产生 IRQ"];
  const captureIndices = new Map();
  frames.forEach((f, i) => { if (one(f, "capture")) captureIndices.set(num(f, "fetch_pc") / 16, i); });
  const findAfter = (start, predicate) => {
    for (let i = start; i < frames.length; i++) if (predicate(frames[i], i)) return i;
    throw new Error("Trace does not contain completion");
  };
  const cases = T.commands.map((cmd, index) => {
    const start = captureIndices.get(index);
    let end;
    if (cmd.name.startsWith("DMA_") || cmd.name === "CONV2D") {
      // DMA loads with no event: observe the busy falling edge after dispatch.
      const dispatch = findAfter(start, f => fire(f, cmd.name === "CONV2D" ? "conv" : "dma"));
      if (cmd.imm & 255) end = findAfter(dispatch, f => ((num(f, cmd.name === "CONV2D" ? "conv_event" : "dma_event") || 0) & (cmd.imm & 255)) !== 0) + 1;
      else end = findAfter(dispatch + 1, f => state(f,"de_state","dma_engine_state_e")==="DE_DONE" && num(f,"dma_engine_tag")===index)+1;
    } else if (cmd.name === "VEC_ADD" || cmd.name === "UPSAMPLE2X") {
      end = findAfter(start, f => one(f, cmd.name === "VEC_ADD" ? "vec_done" : "up_done")) + 1;
    } else if (cmd.name === "END") {
      end = findAfter(start, f => one(f, "irq"));
    } else {
      end = findAfter(start + 1, f => num(f, "retired") > index);
    }
    return { ...cmd, index, start, end: Math.min(end, frames.length - 1), label: caseLabels[index] };
  });
  let chosen = cases[7], position = chosen.start, timer = null, selectedModule = "controller";
  const lessons = window.createNPULessons({frames,cases,num,one,fire,state,signedLow});
  const stepLessons = window.createNPUSteps({frames,cases,lessons,num,one,fire});
  const stepMode = () => $("mode").value === "steps";
  const currentStep = () => stepLessons[chosen.index].find(p=>position-chosen.start>=p.start&&position-chosen.start<=p.end);
  const currentStory = () => stepMode()?currentStep().story:lessons[chosen.index].stories[position-chosen.start];
  // The shared block reader also fetches future instructions. Track its real
  // client so a background fetch cannot masquerade as this step's descriptor.
  const blockOwners=[];let blockOwner=null;
  frames.forEach(f=>{for(const client of ["fetch","df","ef"])if(fire(f,client+"_req"))blockOwner=client;blockOwners.push(blockOwner);if(blockOwner&&fire(f,blockOwner+"_res"))blockOwner=null;});
  const cpPlain = {CP_IDLE:"已空闲",CP_RUN:"可以接收下一条命令",CP_DISPATCH:"检查并交接命令",CP_WAIT_EVENT:"等指定工作完成",CP_WAIT_VECTOR:"等同步运算做完",CP_WAIT_END:"等全部单元空闲",CP_ERROR:"遇到执行错误"};

  // These modules and routes preserve the RTL's hierarchy and interface widths.
  // Request and response directions are shown as separate edges.
  const nodes = [
    ["host", 15, 35, 110, 70, "处理器 / PS", "AXI4-Lite master", "外部控制端", "npu_top.sv", "通过 AXI4-Lite 写入任务地址、长度和启动寄存器。声音样本必须先整理成 DDR 张量。", "OUT 地址 12 位 / 数据 32 位", "IN CSR 状态与 irq_o"],
    ["csr", 155, 35, 135, 70, "CSR 寄存器", "npu_csr", "控制 / 状态", "npu_csr.sv", "接收任务配置与启动请求，保存状态、性能计数和中断标志。本例启动与任务头预装载已在轨迹起点之前完成。", "IN AXI4-Lite 五通道", "OUT task_start / 配置 / irq_o"],
    ["loader", 320, 35, 140, 70, "任务装载", "npu_task_loader", "头部与 LUT 已装载", "npu_task_loader.sv", "读取任务头，配置各段基地址，并装载 4096 项 tanh LUT。页面从第一条指令开始展示；启动阶段已执行完毕。", "OUT 64 位读地址 / 7 位长度", "IN 512 位块响应"],
    ["fetch", 490, 35, 150, 70, "取指与寄存级", "npu_command_fetch", "128-bit command", "npu_command_fetch.sv", "从 DDR 取回 16 字节指令，经过 npu_core 中的一级寄存后交给指令处理器。取指可以与异步运算重叠。", "OUT 128 位指令 / 32 位 PC", "IN command_ready"],
    ["cp", 675, 35, 175, 70, "指令处理器", "npu_command_processor", "译码 / 派发 / 等待", "npu_command_processor.sv", "验证 opcode、flags 和字段，派发给 DMA 或执行前端；维护 8 位事件状态。异步命令派发即退休，同步命令等待完成。", "IN 128 位指令 / event_set", "OUT DMA / compute / vector valid"],
    ["events", 880, 35, 175, 70, "事件状态", "inside command_processor", "8-bit event mask", "npu_command_processor.sv", "这是 CP 内部的事件寄存器，并非独立顶层硬件实例。DMA / CONV 完成置位，新的异步派发清除其完成掩码；WAIT 只检测事件。", "IN DMA / CONV event_set[7:0]", "OUT event_state[7:0]"],
    ["irq", 1170, 35, 155, 70, "中断输出", "irq_o", "CSR → PS", "npu_csr.sv", "core_done 进入 CSR，CSR 根据 IRQ 配置更新中断状态并输出 irq_o。它不是 CONV 单元的独立完成中断。", "IN CSR 中断状态 / 使能", "OUT irq_o · 1 位"],
    ["mem", 155, 170, 165, 75, "描述符请求仲裁", "npu_memory_arbiter4", "Loader / Fetch / DMA / EF", "npu_memory_arbiter4.sv", "仲裁任务装载、取指、DMA 描述符和执行描述符四个客户。块读取器接收请求后，返回完整的 512 位响应。", "IN 4 × 地址 64 / 长度 7", "OUT 块响应 512 位"],
    ["df", 350, 170, 195, 75, "DMA 前端", "npu_dma_frontend", "4-line cache + AGU", "npu_dma_frontend.sv", "读取并缓存 tensor / operator / quant 描述符，AGU 使用迭代乘法器计算 DDR 与片上地址、长度和填充范围。", "IN 128 位 DMA 命令", "OUT DMA request / 事件掩码"],
    ["ef", 675, 170, 175, 75, "执行前端", "npu_execution_frontend", "independent 4-line cache", "npu_execution_frontend.sv", "与 DMA 前端拥有各自独立的描述符缓存。给 CONV 传递 512 位 operator 描述符及 bank / event；给 VEC / UPSAMPLE 传递命令和描述符。", "IN compute / vector 命令", "OUT CONV / VEC / UP start"],
    ["block", 155, 305, 165, 75, "AXI 块读取", "npu_axi_block_reader", "64-bit AXI → 512-bit block", "npu_axi_block_reader.sv", "把描述符与指令的块请求转成 AXI 突发读，拼接最多 64 字节的响应。它不负责张量的 DMA 搬运。", "OUT AR / RREADY", "IN RDATA 64 位"],
    ["dma", 350, 310, 195, 80, "DMA 引擎", "npu_dma_engine", "DDR ↔ Scratchpad · 64-bit", "npu_dma_engine.sv", "执行 AGU 生成的请求，包括填充清零、AXI 读写、片上读写与突发边界处理。完成后上报事件；64 位 scratchpad 路径与计算端不同。", "IN DMA request", "OUT 64 位片上数据 / strobe 8"],
    ["ddr", 15, 450, 110, 190, "DDR 模型", "external memory", "64-bit AXI data", "tb/tb_npu_top_task.sv", "仿真中的外部 AXI 内存模型。输入、指令、描述符、常量与最终输出均在这里；延迟取决于该模型，实际板上时序可能不同。", "IN AR / AW / W", "OUT R / B"],
    ["read", 155, 450, 165, 75, "AXI 读仲裁", "npu_axi_read_arbiter3", "Block / DMA / Upsample", "npu_axi_read_arbiter3.sv", "三个读客户共享外部 AXI AR / R 通道；地址与返回数据的方向相反。只有被选中的客户接收自己的 R 通道。", "OUT ARADDR 64 / ARLEN 8", "IN RDATA 64 / RLAST"],
    ["write", 155, 565, 165, 75, "AXI 写仲裁", "npu_axi_write_arbiter2", "DMA / Upsample", "npu_axi_write_arbiter2.sv", "DMA 与上采样共享 AW / W / B 通道。发送最后一个 W beat 后，还需要接收 B 响应，写事务才完成。", "OUT AWADDR 64 / WDATA 64", "IN BRESP 2 / BVALID"],
    ["a", 600, 300, 190, 65, "A0 / A1 激活", "2 × 64 KiB", "CONV read 128-bit", "npu_scratchpad.sv", "两个激活 bank 存放 INT12 数据，使用 16 位容器。CONV 每次读入 8 个通道；VEC 通过 512 位共享端口读取残差。", "IN DMA write 64 位", "OUT CONV 128 / VEC 512 位"],
    ["w", 600, 405, 190, 65, "W0 / W1 权重", "2 × 32 KiB", "weights + bias + quant", "npu_scratchpad.sv", "存放 O8I8 排列的 INT8 权重，以及对齐的偏置、卷积量化参数和残差量化参数。本例地址依次为 0 / 64 / 128 / 256。", "IN DMA write 64 位", "OUT CONV / VEC read 512 位"],
    ["o", 600, 510, 190, 65, "O0 / O1 输出", "2 × 16 KiB", "CONV write 128-bit", "npu_scratchpad.sv", "卷积的 INT12 输出以 INT16 容器写入；VEC 将残差相加的结果写回这里。DMA_STORE 使用 64 位路径读出。", "IN CONV 128 / VEC 512 位", "OUT DMA read 64 位"],
    ["controller", 890, 310, 190, 80, "卷积控制器", "npu_conv2d_controller", "setup + metadata FIFO", "npu_conv2d_controller.sv", "先迭代计算 tile 地址与步长，再请求激活与权重。8 深度 metadata FIFO 对齐 first / last / mask 与同步读响应，然后将 token 送入 MAC。", "OUT A / W 读请求", "IN A 128 / W 512 位响应"],
    ["mac", 890, 425, 190, 70, "8 × 8 MAC", "npu_tensor_mac_8x8", "64 multipliers / 8 outputs", "npu_tensor_mac_8x8.sv", "64 路并行乘法，经两层加法树和 dot 级后累加到 INT32。输出阻塞会冻结流水线。它不是带邻接转发的脉动阵列。", "IN 激活 128 / 权重 512 位", "OUT INT32 × 8 = 256 位"],
    ["buffer", 1110, 425, 225, 70, "结果寄存缓冲", "inside conv2d_pipeline", "non-fall-through buffer", "npu_conv2d_pipeline.sv", "接住控制器返回的 256 位累加结果，切断慢速后处理向 MAC 的组合反压路径。结果接收后，下一周期才可送入后处理。", "IN 256 位 INT32 结果", "OUT 256 位 / lane mask"],
    ["post", 1110, 530, 225, 70, "融合后处理", "npu_requant_post", "bias → requant → activation", "npu_requant_post.sv", "逐通道处理偏置、Q31 乘法、RNE 舍入、clamp 和可选激活。每个通道经历多个状态；本例为单位量化比例、无激活。", "IN INT32 × 8 / quant / bias", "OUT INT16 × 8 = 128 位"],
    ["fifo", 1110, 635, 225, 70, "卷积写回 FIFO", "inside dma_subsystem", "2 entries · data + mask", "npu_dma_subsystem.sv", "两项 FIFO 保存 128 位结果、地址、bank 和 lane mask。CONV 完成事件必须等 FIFO 排空后才上报。", "IN 128 位后处理输出", "OUT 128 位 O 写回 / strobe 16"],
    ["vec", 605, 710, 225, 75, "残差相加", "npu_vec_add", "own requant + INT12 add", "npu_vec_add.sv", "使用独立后处理单元对 A 残差重定量，与 O 输入相加后饱和写 O。端口宽 512 位，但本例只有低 128 位是 8 个通道的数据。", "IN 共享读口 512 位 / W 参数", "OUT 共享写口 512 / strobe 64"],
    ["up", 350, 710, 195, 75, "二倍上采样", "npu_upsample2x", "DDR → row buffer → DDR", "npu_upsample2x.sv", "直接访问 DDR，先读一行，用 1 KiB 有效行缓冲复制像素和行。源码数组的物理深度为 256×64 位，但当前几何上限最多使用 128 beats。", "IN DDR RDATA 64 位", "OUT DDR WDATA 64 位"],
  ].map(([id, x, y, width, height, title, subtitle, status, source, detail, input, output]) => ({id, x, y, width, height, title, subtitle, status, source, detail, input, output}));
  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));
  const edges = [
    ["host_csr", "host", "csr", "M125 70 H155", "AXI-Lite", 127, 60],
    ["loader_fetch", "loader", "fetch", "M460 70 H490", "base", 462, 60],
    ["fetch_cp", "fetch", "cp", "M640 70 H675", "128", 644, 60],
    ["cp_events", "events", "cp", "M880 70 H850", "8", 859, 58],
    ["done_csr", "cp", "csr", "M720 35 V18 H225 V35", "core_done", 500, 15],
    ["csr_irq", "csr", "irq", "M270 35 V8 H1248 V35", "irq_o", 1120, 20],
    ["cp_df", "cp", "df", "M710 105 V140 H445 V170", "DMA command · 128", 480, 135],
    ["cp_ef", "cp", "ef", "M790 105 V170", "128", 795, 151],
    ["fetch_req", "fetch", "mem", "M510 105 V115 H190 V170", "fetch req", 326, 111],
    ["fetch_res", "mem", "fetch", "M165 170 V130 H610 V105", "instruction block", 199, 129],
    ["df_req", "df", "mem", "M350 190 H320", "req", 322, 181],
    ["df_res", "mem", "df", "M320 225 H350", "512", 322, 237],
    ["ef_req", "ef", "mem", "M675 195 H580 V155 H245 V170", "descriptor request", 356, 153],
    ["ef_res", "mem", "ef", "M310 245 V275 H830 V245", "512-bit response", 555, 268],
    ["mem_block", "mem", "block", "M210 245 V305", "addr64 / len7", 213, 285],
    ["block_read", "block", "read", "M195 380 V450", "AR", 200, 420],
    ["read_block", "read", "block", "M270 450 V380", "R64", 274, 420],
    ["df_dma", "df", "dma", "M445 245 V310", "DMA request", 451, 295],
    ["dma_a", "dma", "a", "M545 330 H600", "64", 567, 321],
    ["dma_w", "dma", "w", "M545 350 H565 V435 H600", "64", 575, 429],
    ["dma_o", "o", "dma", "M600 552 H555 V375 H545", "64", 569, 540],
    ["dma_o_req", "dma", "o", "M545 382 H549 V567 H600", "read req", 555, 587],
    ["a_req", "controller", "a", "M890 320 H850 V310 H790", "req", 809, 303],
    ["a_mac", "a", "controller", "M790 345 H840 V360 H890", "128", 810, 359],
    ["w_req", "controller", "w", "M900 390 V400 H810 V425 H790", "req", 815, 417],
    ["w_mac", "w", "controller", "M790 455 H855 V380 H890", "512", 807, 451],
    ["ef_conv", "ef", "controller", "M850 205 H985 V310", "opdesc512 + bank / event", 909, 237],
    ["ctrl_mac", "controller", "mac", "M985 390 V425", "A128 / W512 + token", 991, 413],
    ["mac_buf", "mac", "buffer", "M1080 460 H1110", "256", 1084, 450],
    ["buf_post", "buffer", "post", "M1220 495 V530", "256", 1226, 515],
    ["post_fifo", "post", "fifo", "M1220 600 V635", "128", 1226, 621],
    ["fifo_o", "fifo", "o", "M1110 670 H865 V550 H790", "128 · strobe16", 930, 660],
    ["ef_vec", "ef", "vec", "M820 245 H850 V748 H830", "start", 853, 725],
    ["vec_ef", "vec", "ef", "M830 767 H859 V239 H850", "done", 863, 772],
    ["vec_req", "vec", "a", "M650 710 V655 H585 V365 H640", "read req", 592, 641],
    ["vec_wreq", "vec", "w", "M700 710 V696 H820 V490 H760 V470", "read req", 825, 520],
    ["vec_oreq", "vec", "o", "M690 710 V575", "read req", 642, 601],
    ["a_vec", "a", "vec", "M610 365 V630 H670 V710", "512", 618, 603],
    ["w_vec", "w", "vec", "M710 470 V490 H810 V690 H700 V710", "512 · params", 700, 685],
    ["o_vec", "o", "vec", "M720 575 V710", "512", 727, 612],
    ["vec_o", "vec", "o", "M770 710 V575", "512 / strobe64", 775, 641],
    ["ef_up", "ef", "up", "M690 245 V260 H580 V690 H450 V710", "command + src / dst desc", 360, 680],
    ["up_ef", "up", "ef", "M545 773 H570 V255 H700 V245", "done", 548, 794],
    ["ddr_r", "ddr", "read", "M125 500 H155", "R64", 128, 490],
    ["read_ddr", "read", "ddr", "M155 470 H125", "AR64", 127, 458],
    ["read_dma", "read", "dma", "M320 480 H335 V365 H350", "R64", 335, 438],
    ["dma_read", "dma", "read", "M350 335 H330 V460 H320", "AR", 331, 401],
    ["dma_write", "dma", "write", "M390 390 V603 H320", "AW / W64", 335, 593],
    ["write_dma", "write", "dma", "M320 620 H410 V390", "B", 414, 622],
    ["write_ddr", "write", "ddr", "M155 595 H125", "AW / W", 127, 586],
    ["ddr_b", "ddr", "write", "M125 625 H155", "B", 135, 619],
    ["up_read", "up", "read", "M350 725 H340 V515 H320", "AR", 256, 706],
    ["read_up", "read", "up", "M320 523 H345 V748 H350", "R64", 325, 689],
    ["up_write", "up", "write", "M350 765 H300 V640", "AW / W64", 166, 747],
    ["write_up", "write", "up", "M270 640 V781 H350", "B", 276, 772],
    ["dma_event", "dma", "events", "M545 370 H575 V120 H930 V105", "dma_event_set", 592, 118],
    ["conv_event", "fifo", "events", "M1335 670 H1360 V128 H1005 V105", "conv_event_set (after drain)", 1100, 128],
  ].map(([id, from, to, path, label, x, y]) => ({id, from, to, path, label, x, y}));
  const edgeMap = Object.fromEntries(edges.map(e => [e.id, e]));
  function buildDiagram() {
    const svg = $("diagram");
    const defs = element("defs");
    [["arrow", "#68819d"], ["arrow-active", "#59dddc"], ["arrow-wait", "#fb8da6"]].forEach(([id, color]) => {
      const marker = element("marker", {id, viewBox:"0 0 10 10", refX:9, refY:5, markerWidth:5, markerHeight:5, orient:"auto-start-reverse"});
      marker.append(element("path", {d:"M0 0 L10 5 L0 10 Z", fill:color})); defs.append(marker);
    }); svg.append(defs);
    [["Scratchpad · 6 banks / 224 KiB", 590, 283, 211, 305], ["CONV pipeline + output FIFO", 875, 283, 475, 434]].forEach(([label, x, y, width, height]) => {
      svg.append(element("rect", {x, y, width, height, class:"group-box"}));
      svg.append(element("text", {x:x+8, y:y-9, class:"group-label"}, label));
    });
    edges.forEach(e => {
      const g = element("g", {id:"edge-"+e.id, class:"edge"});
      g.append(element("path", {d:e.path, "marker-end":"url(#arrow)"}));
      g.append(element("text", {x:e.x, y:e.y}, e.label)); svg.append(g);
    });
    nodes.forEach(n => {
      const g = element("g", {id:"node-"+n.id, class:"module", tabindex:0, role:"button", "aria-label":n.title});
      g.append(element("rect", {x:n.x, y:n.y, width:n.width, height:n.height}));
      g.append(element("text", {x:n.x+n.width/2, y:n.y+23, "text-anchor":"middle", class:"node-title"}, n.title));
      g.append(element("text", {x:n.x+n.width/2, y:n.y+40, "text-anchor":"middle", class:"node-subtitle"}, n.subtitle));
      g.append(element("text", {x:n.x+n.width/2, y:n.y+57, "text-anchor":"middle", class:"node-status"}, n.status));
      g.append(element("rect", {x:n.x-4,y:n.y-4,width:n.width+8,height:n.height+8,class:"lesson-focus-ring"}));
      g.addEventListener("click", () => inspect(n.id));
      g.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); inspect(n.id); } });
      svg.append(g);
    });
    svg.append(element("text", {x:15, y:790, class:"svg-note"}, "aclk = 100 MHz · aresetn 低有效 · AXI 地址 / 数据方向与 READY 相反"));
  }
  function inspect(id) {
    selectedModule = id;
    const n = nodeMap[id];
    $("module-title").textContent = n.title;
    $("module-detail").textContent = n.detail;
    $("module-io").replaceChildren(...[n.input, n.output].map(text => {
      const row = document.createElement("div"); row.className = "io-row";
      const b = document.createElement("b"); b.textContent = text.split(" ")[0];
      const span = document.createElement("span"); span.textContent = text.substring(text.indexOf(" ")+1);
      row.append(b, span); return row;
    }));
    $("module-source").href = "https://github.com/ziyan2006/NPU/blob/work/hardware/rtl/" + n.source;
    document.querySelectorAll(".module").forEach(g => g.classList.toggle("selected", g.id === "node-" + id));
  }

  function describe(f) {
    const active = new Set(), wait = new Set(), busy = new Set(), notes = [], handshakes = [];
    const mark = id => { active.add(id); const e = edgeMap[id]; if (e) { active.add(e.from); active.add(e.to); } };
    const transfer = (prefix, routes, note, display = prefix) => {
      if (fire(f, prefix)) { routes.forEach(mark); if (note) notes.push(note); handshakes.push({name:display, v:1, r:1}); }
      else if (blocked(f, prefix)) { routes.forEach(id => wait.add(id)); handshakes.push({name:display, v:1, r:0}); }
    };
    if (one(f, "dma_frontend_busy")) busy.add("df");
    if (one(f, "dma_engine_busy")) busy.add("dma");
    if (one(f, "conv_busy")) busy.add("controller");
    if (num(f, "ef_state")) busy.add("ef");
    if (one(f, "vec_busy")) busy.add("vec");
    if (one(f, "up_busy")) busy.add("up");
    if (num(f, "post_state")) busy.add("post");
    if (["mac_product", "mac_l1", "mac_l2", "mac_dot", "mac_out_v"].some(k => one(f, k))) busy.add("mac");
    if (one(f, "post_v")) busy.add("buffer");
    if (num(f, "fifo_count")) busy.add("fifo");
    if (num(f,"cp_state")>0) busy.add("cp");
    transfer("fetch", ["fetch_cp"], "取指寄存级把 128 位命令交给 CP。", "command_valid / ready");
    transfer("dma", ["cp_df"], "CP 向 DMA 前端派发异步命令；本沿退休，搬运仍继续。", "cp_dma_valid / ready");
    transfer("conv", ["cp_ef"], "CP 向执行前端派发卷积命令；本沿退休，执行前端开始解析描述符。", "cp_compute_valid / ready");
    transfer("vec", ["cp_ef"], "CP 派发同步命令，随后等待执行前端完成。", "cp_vector_valid / ready");
    transfer("fetch_req", ["fetch_req"], "取指单元请求读取下一条 16 字节命令。");
    transfer("fetch_res", ["fetch_res"], "取指单元接收完整的命令块响应。");
    transfer("block_req", ["mem_block"], "仲裁后的块读请求被 AXI 块读取器接收。");
    transfer("df_req", ["df_req"], "DMA 描述符缓存未命中，发出块读请求。");
    transfer("df_res", ["df_res"], "完整的 512 位描述符返回 DMA 前端。");
    transfer("ef_req", ["ef_req"], "执行前端缓存未命中，发出描述符块读请求。");
    transfer("ef_res", ["ef_res"], "完整的 512 位描述符返回执行前端。");
    transfer("conv_start", ["ef_conv"], "512 位 operator 描述符与 bank / event 被卷积流水线接收。");
    transfer("vec_start", ["ef_vec"], "残差相加单元接收命令和描述符。");
    transfer("up_start", ["ef_up"], "上采样单元接收命令与源 / 目标 tensor 描述符。");
    const sp = num(f, "sp_wkind") === 0 ? "dma_a" : "dma_w";
    transfer("sp_w", [sp], `${state(f,"de_state","dma_engine_state_e")==="DE_CLEAR" ? "填充清零写入" : "DMA 数据写入"} ${num(f,"sp_wkind") === 0 ? "A0" : "W0"}，地址 ${hex(num(f,"sp_waddr"),4)}，64 位数据 ${valueHex(f,"sp_wdata")}。`, "scratchpad_write_valid / ready");
    transfer("sp_r", ["dma_o_req"], "DMA 请求从 O0 读取 64 位输出数据。");
    transfer("sp_res", ["dma_o"], "O0 读响应进入 DMA 写通道缓冲。");
    transfer("a_req", ["a_req"], `卷积请求 A0，片上地址 ${hex(num(f,"a_addr"),4)}。`);
    transfer("a_res", ["a_mac"], `A0 返回 128 位激活，通道 0 = ${signedLow(f,"a_data",16)}。`);
    transfer("w_req", ["w_req"], `卷积请求 W0，地址 ${hex(num(f,"weight_addr"),4)}${num(f,"weight_addr")===64 ? "（偏置）" : num(f,"weight_addr")>=128 ? "（量化参数）" : "（权重）"}。`);
    transfer("w_res", ["w_mac"], "W0 返回 512 位权重或预取参数。");
    transfer("mac", ["ctrl_mac"], "A / W 响应与 metadata 对齐，一个输入 token 在本沿进入 MAC。", "mac_input_valid / ready");
    transfer("mac_out", ["mac_buf"], `INT32 累加结果进入缓冲，通道 0 = ${signedLow(f,"mac_data",32)}。`);
    transfer("post", ["buf_post"], "结果缓冲将 8 路 INT32 送入后处理。", "post_input_valid / ready");
    transfer("out", ["post_fifo"], `8 路 INT16 后处理结果入 FIFO，通道 0 = ${signedLow(f,"out_data",16)}。`, "conv_result_valid / ready");
    transfer("o", ["fifo_o"], `FIFO 向 O0 写入 128 位数据，通道 0 = ${signedLow(f,"o_data",16)}。`, "conv_output_write_valid / ready");
    if (fire(f, "v_read")) {
      mark(["vec_req","vec_wreq","vec_oreq"][num(f,"v_kind")]);
      notes.push(`VEC 请求 ${["A0","W0","O0"][num(f,"v_kind")]}，走共享 512 位读接口。`);
    }
    const vResponseKind=state(f,"v_state","vec_state_e")==="VEC_PARAM_RESPONSE"?1:state(f,"v_state","vec_state_e")==="VEC_SRC0_RESPONSE"?2:0;
    transfer("v_res", [["a_vec","w_vec","o_vec"][vResponseKind]], `VEC 接收 ${["A0","W0","O0"][vResponseKind]} 响应，低 16 位 = ${signedLow(f,"v_data",16)}。`);
    transfer("v_write", ["vec_o"], `饱和相加结果写回 O0，通道 0 = ${signedLow(f,"v_write_data",16)}。`);
    if (fire(f, "ar")) {
      mark("read_ddr");
      const route = one(f,"dma_ar") ? "dma_read" : one(f,"up_ar") ? "up_read" : "block_read"; mark(route);
      notes.push(`AXI AR 地址握手：${hex(num(f,"ar_addr"),8)}，${num(f,"ar_len")+1} 个 beat。`);
      handshakes.push({name:"ARVALID / ARREADY",v:1,r:1});
    } else if (blocked(f,"ar")) { wait.add("read_ddr"); handshakes.push({name:"ARVALID / ARREADY",v:1,r:0}); }
    if (fire(f,"r")) {
      mark("ddr_r");
      mark(fire(f,"dma_rdata") ? "read_dma" : fire(f,"up_rdata") ? "read_up" : "read_block");
      notes.push(`AXI R 数据握手：${valueHex(f,"r_data")}${one(f,"r_last") ? "，最后一拍 RLAST=1" : ""}。`);
      handshakes.push({name:"RVALID / RREADY",v:1,r:1});
    } else if (blocked(f,"r")) { wait.add("ddr_r"); handshakes.push({name:"RVALID / RREADY",v:1,r:0}); }
    ["aw","w","b"].forEach(channel => {
      if (fire(f,channel)) {
        mark(channel === "b" ? "ddr_b" : "write_ddr");
        mark(channel==="b" ? (one(f,"up_busy") ? "write_up" : "write_dma") : (one(f,"up_busy") ? "up_write" : "dma_write"));
        notes.push(channel === "aw" ? `AXI AW 地址握手：${hex(num(f,"aw_addr"),8)}，${num(f,"aw_len")+1} 个 beat。` : channel === "w" ? `AXI W 数据握手：${valueHex(f,"w_data")}${one(f,"w_last") ? "，WLAST=1；仍需 B 响应" : ""}。` : "AXI B 写响应被接收，本次写事务完成。");
        handshakes.push({name:channel.toUpperCase()+"VALID / READY",v:1,r:1});
      } else if (blocked(f,channel)) { wait.add(channel === "b" ? "ddr_b" : "write_ddr"); handshakes.push({name:channel.toUpperCase()+"VALID / READY",v:1,r:0}); }
    });
    if (num(f,"dma_event")) { mark("dma_event"); mark("cp_events"); notes.push(`DMA 完成事件 ${hex(num(f,"dma_event"))} 在本沿置位。`); }
    if (num(f,"conv_event")) { mark("conv_event"); mark("cp_events"); notes.push("卷积流水线与写回 FIFO 均已完成，本沿设置 C0_DONE。"); }
    if (one(f,"vec_done")) { mark("vec_ef"); notes.push("VEC 发出 done，执行前端随后解除 CP 的同步等待。"); }
    if (one(f,"up_done")) { mark("up_ef"); notes.push("上采样完成 2×2 输出，发出 done。"); }
    if (one(f,"core_done")) { mark("done_csr"); notes.push("core_done 进入 CSR，设置任务完成状态与中断标志。"); }
    if (one(f,"irq")) { mark("csr_irq"); notes.push("irq_o=1，处理器可读取完成状态。"); }
    return {active, wait, busy, notes, handshakes};
  }

  function phaseText(f) {
    if (one(f,"capture")) return ["接收并锁存指令", "CP 本沿接收命令，下一周期进入 CP_DISPATCH。"];
    if (num(f,"cp_state") === 2 && !one(f,"dma_v") && !one(f,"conv_v") && !one(f,"vec_v")) {
      if (chosen.name === "NOP") return ["空操作退休", "NOP 完成验证后退休；不产生张量数据传输。"];
      if (chosen.name === "WAIT") return ["检测事件掩码", `WAIT 检测掩码 ${hex(chosen.imm&255)}；已置位或本沿到达的事件都参与判断。`];
      if (chosen.name === "END") return ["确认执行单元空闲", "END 在所有执行单元空闲后结束任务。"];
    }
    if (num(f,"post_state") > 0 && num(f,"post_state") < 9) return ["逐通道融合后处理", `通道 ${num(f,"post_lane")}：${state(f,"post_state","post_state_e")}。本沿推进一个后处理状态。`];
    if (["mac_product","mac_l1","mac_l2","mac_dot"].some(k=>one(f,k))) return ["MAC 流水级推进", "有效 token 正沿乘法与加法树流水线推进；下方显示进入本沿前各级 valid。"];
    if (one(f,"vec_busy")) return ["残差相加执行中", `${state(f,"v_state","vec_state_e")}；独立后处理状态 ${state(f,"v_post","post_state_e")}。`];
    if (one(f,"up_busy")) return ["DDR 上采样执行中", `${state(f,"u_state","upsample_state_e")}；读一行后，分别复制列与行。`];
    if (one(f,"dma_busy")) return ["DMA 正在解析或搬运", `前端 ${state(f,"df_state","dma_frontend_state_e")}；AGU ${state(f,"agu_state","agu_state_e")}；引擎 ${state(f,"de_state","dma_engine_state_e")}。`];
    if (num(f,"cp_state") === 3) return ["等待事件", `CP 处于 CP_WAIT_EVENT；事件状态 ${hex(num(f,"events"))}。没有满足条件时，WAIT 保持阻塞。`];
    if (one(f,"conv_busy")) return ["卷积设置与参数预取", `控制器 ${state(f,"ctrl_state","setup_state_e")}；参数预取 ${state(f,"param_state","param_state_e")}。迭代地址计算与同步存储响应需要周期。`];
    if (one(f,"irq")) return ["任务完成，中断已置位", "14 条指令全部退休，DDR 中 4 个输出像素的通道 0 均为 9。"];
    return ["本沿没有数据传输", "控制状态仍可能推进。下一关键沿可跳过当前没有传输的区间；单步保留每个实际周期。"];
  }

  const observations = [];
  let memory = {a:null, w:null, o:null, ddr:null};
  frames.forEach(f => {
    if (fire(f,"sp_w") && num(f,"sp_waddr")===0) {
      if (num(f,"sp_wkind")===0) memory.a = signedLow(f,"sp_wdata",16);
      if (num(f,"sp_wkind")===1) memory.w = signedLow(f,"sp_wdata",8);
    }
    if (fire(f,"o")) memory.o = signedLow(f,"o_data",16);
    if (fire(f,"v_write")) memory.o = signedLow(f,"v_write_data",16);
    if (fire(f,"w") && signedLow(f,"w_data",16) === 9) memory.ddr=9;
    observations.push({...memory});
  });

  function textBox(tag,text,className="") {const el=document.createElement(tag);el.textContent=text;if(className)el.className=className;return el;}
  function renderLessonVisual(s,f) {
    const visual=$("lesson-visual");visual.replaceChildren();let note="";
    if(s.multiplication){
      visual.append(textBox("strong",s.calculation,"lesson-calculation"));
      const rounds=document.createElement("div");rounds.className="calculation-rounds";
      for(let j=1;j<=16;j++){const dot=textBox("span",String(j),j<=s.mulRound?"done":"");rounds.append(dot);}
      visual.append(rounds);
      const done=one(f,s.mulPrefix+"_mul_done");
      note=done?"已读到乘法器结果："+num(f,s.mulPrefix+"_mul_result"):s.mulRound?"正在逐轮算地址，尚未交接乘法结果。":"准备启动这项地址乘法。";
    }else if(chosen.name==="UPSAMPLE2X"){
      const grid=document.createElement("div");grid.className="upsample-visual";
      grid.append(textBox("div","原像素\n9","source-pixel"),textBox("span","→","visual-operator"));
      const output=document.createElement("div");output.className="pixel-grid";
      for(let j=0;j<4;j++){const copied=j<(s.copied||0),cell=textBox("div",copied?"9":"待写",copied?"copied":"");if(j===s.activePixel&&!copied)cell.classList.add("being-written");output.append(cell);}
      grid.append(output);visual.append(grid);note=`已发送完整像素：${s.copied||0} / 4。每个像素有 8 个通道，需要两个 8 字节数据块。`;
    }else if(chosen.name==="WAIT"){
      const bits=(num(f,"events")||0)|(num(f,"event_set")||0);
      const lamps=s.lamps||[[1,"输入 A0"],[4,"参数 W0"],[16,"卷积 O0"],[64,"写回 O0"]].filter(([bit])=>chosen.imm&bit).map(([bit,label])=>({label,ready:!!(bits&bit)}));
      lamps.forEach(l=>visual.append(textBox("div",(l.ready?"● 已完成 · ":"○ 未完成 · ")+l.label,"event-lamp"+(l.ready?" lit":""))));
      note="检查全部指定的灯，包含本沿到达的完成消息；通过后不会关灯。";
    }else if(chosen.name==="CONV2D"||chosen.name==="VEC_ADD"){
      const conv=chosen.name==="CONV2D",observed=conv?num(f,"mac_data")===6||observations[position].o!==null:observations[position].o===9;
      const eq=document.createElement("div");eq.className="lesson-equation";
      [[conv?"输入":"卷积结果",conv?3:6],[conv?"权重":"保留残差",conv?2:3],["目标结果",conv?6:9]].forEach(([label,value],j)=>{
        if(j)eq.append(textBox("span",j===1?(conv?"×":"+"):"=","visual-operator"));
        const box=document.createElement("div");box.className="equation-value"+(j===2&&!observed?" future":"");box.append(textBox("span",label),textBox("b",String(value)));eq.append(box);
      });visual.append(eq);note=observed?"本例的通道 0 已产生这个结果，其余 7 个通道均为 0。":"这是本例的计算目标，经过准备、读取和计算后才产生结果。";
    }else if(chosen.name.startsWith("DMA_")){
      const load=chosen.name==="DMA_LOAD",label=chosen.index===1?"输入 3":chosen.index===2?"权重 2":chosen.index===3?"偏置 0":chosen.index===4||chosen.index===5?"缩放配方 1":"结果 9";
      const route=document.createElement("div");route.className="lesson-route";
      route.append(textBox("div",load?"DDR\n外部仓库":"O0\n计算结果区"),textBox("span","→","visual-operator"),textBox("div",load?(chosen.index===1?"A0\n输入工作台":"W0\n参数配方区"):"DDR\n外部仓库"));visual.append(route,textBox("strong",label,"route-payload"));
      note=s.chapter==="address"||s.chapter==="descriptors"||s.chapter==="accept"?"正在准备搬运，输入或参数本身还没开始交接。":s.chapter==="done"?"这条指令对应的数据已经搬完。":"当前正在执行清零、读写或内存确认。";
    }else{
      visual.append(textBox("strong",chosen.name==="NOP"?"接收 → 检查 → 继续":"全部空闲 → 完成标志 → 通知处理器","control-visual"));
      note=chosen.name==="NOP"?"“处理完一条指令”不代表一定做了计算。":"完成消息经状态寄存器后，产生中断通知。";
    }
    $("lesson-example-note").textContent=note;
  }
  function renderLesson(f){
    const lesson=lessons[chosen.index],relative=position-chosen.start,s=currentStory(),p=currentStep();
    $("lesson-goal").textContent=chosen.name+" · 目标："+lesson.goal;
    $("lesson-title").textContent=s.title;$("lesson-action").textContent=s.action;
    $("lesson-why").textContent=s.why;$("lesson-result").textContent=s.result;
    const current=stepMode()?p.index+1:relative-s.run.start+1,total=stepMode()?stepLessons[chosen.index].length:s.run.end-s.run.start+1;
    $("lesson-progress-label").textContent=stepMode()?`第 ${current} / ${total} 步`:`第 ${s.chapterIndex+1} / ${lesson.chapters.length} 阶段 · ${s.chapterTitle}`;
    $("lesson-progress-detail").textContent=stepMode()?`覆盖 K+${p.start}…K+${p.end} · ${p.end-p.start+1} 个真实时钟 · 当前框图 K+${relative}`:`当前这项工作：第 ${current} / ${total} 个时钟（本次录制）${s.multiplication?" · 乘法本身需 16 轮，另有启动与结果交接":""}`;
    $("lesson-progress").max=total;$("lesson-progress").value=current;
    $("lesson-focus").replaceChildren(...s.focus.filter(id=>nodeMap[id]).map(id=>{const b=textBox("button",nodeMap[id].title);b.addEventListener("click",()=>inspect(id));return b;}));
    const navigation=stepMode()?stepLessons[chosen.index]:lesson.chapters;
    $("lesson-chapters").replaceChildren(...navigation.map((n,j)=>{const active=stepMode()?n===p:n.id===s.chapter;const b=textBox("button",stepMode()?`${j+1}. ${n.story.title}`:`${j+1}. ${n.title} · ${n.cycles} 拍`,active?"current":"");b.setAttribute("aria-current",active?"step":"false");b.addEventListener("click",()=>{pause();position=chosen.start+(stepMode()?n.representative:n.start);render();});return b;}));
    $("step-evidence").hidden=!stepMode();
    const activeEdges=edges.filter(e=>stepActivities(describe(f)).active.has(e.id));
    $("step-flow").textContent=activeEdges.length?"此刻真实交接："+activeEdges.map(e=>nodeMap[e.from].title+" → "+nodeMap[e.to].title).join("；"):"此刻没有数据交接；关注模块内部的准备、计算或等待。";
    $("step-moments").replaceChildren(...p.moments.map(j=>{const b=textBox("button",`K+${j} · ${lessons[chosen.index].stories[j].title}`,j===relative?"current":"");b.addEventListener("click",()=>{pause();position=chosen.start+j;render();});return b;}));
    $("lesson-context").textContent=stepMode()?"框图展示上方标明的一个真实时刻，只高亮所选指令在本步骤的活动。讲解概括整步；点击关键时刻可对照具体交接。":"框图保留全部真实活动，取指或其他异步指令可能同时工作。";
    nodes.forEach(n=>$("node-"+n.id).classList.toggle("lesson-focus",s.focus.includes(n.id)));
    renderLessonVisual(s,f);
  }

  function stepActivities(d){
      const p=currentStep(),allowed=new Set(p.allowed);
      if(p.id==="descriptors"){
        const client=chosen.name.startsWith("DMA_")?"df":"ef";
        if(blockOwners[position]!==client)["mem_block","block_read","read_ddr","ddr_r","read_block"].forEach(id=>allowed.delete(id));
        if(!d.active.has("block_read"))allowed.delete("read_ddr");
        if(!d.active.has("read_block"))allowed.delete("ddr_r");
      }
      if(p.id==="request"&&chosen.name==="DMA_LOAD"&&!d.active.has("dma_read"))allowed.delete("read_ddr");
      if(p.id==="move"&&chosen.name==="DMA_LOAD"&&!d.active.has("read_dma"))allowed.delete("ddr_r");
      if(p.id==="read"&&chosen.name==="UPSAMPLE2X"){
        if(!d.active.has("up_read"))allowed.delete("read_ddr");
        if(!d.active.has("read_up"))allowed.delete("ddr_r");
      }
      const active=new Set([...d.active].filter(id=>allowed.has(id))),wait=new Set([...d.wait].filter(id=>allowed.has(id)));
      [...active].forEach(id=>{if(edgeMap[id]){active.add(edgeMap[id].from);active.add(edgeMap[id].to);}});
      return {...d,active,wait,busy:new Set([...d.busy].filter(id=>p.story.focus.includes(id)))};
  }
  function updateHardware(f, d) {
    if(stepMode())d=stepActivities(d);
    edges.forEach(e => {
      const g = $("edge-"+e.id), active = d.active.has(e.id), wait = d.wait.has(e.id);
      g.classList.toggle("active", active); g.classList.toggle("wait", wait);
      g.firstChild.setAttribute("marker-end", `url(#${active ? "arrow-active" : wait ? "arrow-wait" : "arrow"})`);
    });
    nodes.forEach(n => {
      const g = $("node-"+n.id);
      ["active","busy","wait"].forEach(kind => g.classList.toggle(kind, d[kind].has(n.id)));
      const status = g.querySelector(".node-status");
      const states = {cp:state(f,"cp_state","cp_state_e"), df:state(f,"df_state","dma_frontend_state_e"), dma:state(f,"de_state","dma_engine_state_e"), ef:state(f,"ef_state","execution_frontend_state_e"), post:state(f,"post_state","post_state_e")+" · lane "+num(f,"post_lane"), vec:state(f,"v_state","vec_state_e"), up:state(f,"u_state","upsample_state_e"), events:"event_state = "+hex(num(f,"events")), fifo:"count = "+num(f,"fifo_count")+" / 2", irq:"irq_o = "+num(f,"irq")};
      const lesson=currentStory();
      const defaults={host:"发送任务与启动命令",csr:"控制任务并汇报状态",loader:"启动阶段已完成",fetch:"读取下一条命令",mem:"安排谁先读说明",df:"准备搬运说明与地址",ef:"准备运算说明",block:"从外部内存读取说明",dma:"执行实际数据搬运",ddr:"存放输入、参数与输出",read:"安排谁先读取 DDR",write:"安排谁先写入 DDR",a:"输入工作台 · 两组",w:"权重与参数配方 · 两组",o:"计算结果区 · 两组",controller:"安排卷积读取与计算",mac:"对应相乘，再相加",buffer:"暂存算好的结果",post:"逐通道调整数值",fifo:"计算结果排队写回",vec:"把两份特征相加",up:"复制像素和行",events:"记录工作是否完成",irq:one(f,"irq")?"已通知处理器完成":"等待任务完成"};
      status.textContent = $("show-tech").checked ? (states[n.id] || (d.active.has(n.id)?"本上升沿完成传输":d.busy.has(n.id)?"正在处理":n.status)) : n.id==="cp"?cpPlain[state(f,"cp_state","cp_state_e")]:lesson.focus.includes(n.id)?lesson.chapterTitle+" · 当前关注":d.active.has(n.id)?"本沿交接数据":d.busy.has(n.id)?"仍在处理工作":defaults[n.id];
    });
  }
  const pipelineStages = [["乘法", "mac_product", "24-bit"], ["加法 L1", "mac_l1", "25-bit"], ["加法 L2", "mac_l2", "26-bit"], ["Dot", "mac_dot", "27-bit"], ["累加 / 结果", "mac_out_v", "32-bit"]];
  function updatePipeline(f) {
    $("mac-pipeline").replaceChildren(...pipelineStages.map(([label, key, bits]) => {
      const div = document.createElement("div"); div.className = "mac-stage"+(one(f,key) ? " active" : "");
      const strong = document.createElement("strong"); strong.textContent=label;
      const span = document.createElement("span"); span.textContent=bits+" · valid="+num(f,key); div.append(strong,span); return div;
    }));
    $("memory-values").replaceChildren(...[["A0", "a", "输入 ch0"],["W0", "w", "权重 [0,0]"],["O0", "o", "输出 ch0"],["DDR", "ddr", "已发送 ch0"]].map(([label,key,note])=>{
      const div=document.createElement("div");div.className="memory-value";
      const name=document.createElement("span");name.textContent=label;
      const b=document.createElement("b");b.textContent=observations[position][key] ?? "—";
      const span=document.createElement("span");span.textContent=note;div.append(name,b,span);return div;
    }));
  }

  function waveRows() {
    if(stepMode()){
      const p=currentStep(),chapter=p.story.chapter;
      const base=[["时钟 · 10 ns","clock"]];
      if(p.id==="capture")return [...base,["命令有效","fetch_v"],["控制器可接收","fetch_r"]];
      if(["wait-dispatch","dispatch"].includes(p.id)){const prefix=chosen.name.startsWith("DMA_")?"dma":chosen.name==="CONV2D"?"conv":"vec";return [...base,["派工单有效",prefix+"_v"],["前端可接单",prefix+"_r"]];}
      if(chapter==="descriptors"){const prefix=chosen.name.startsWith("DMA_")?"df":"ef";return [...base,["申请读取说明",prefix+"_req_v"],["说明返回有效",prefix+"_res_v"],["前端接收说明",prefix+"_res_r"]];}
      if(chapter==="address"){const prefix=chosen.name.startsWith("DMA_")?"agu":chosen.name==="CONV2D"?"ctrl":chosen.name==="VEC_ADD"?"v":"u";return [...base,["地址乘法进行中",prefix+"_mul_busy"],["乘法结果可用",prefix+"_mul_done"]];}
      if(p.id==="calculate")return [...base,["乘法级有效","mac_product"],["累加级有效","mac_dot"],["结果有效","mac_out_v"],["接收结果","mac_out_r"]];
      if(p.id==="post"||p.id==="scale")return [...base,["调整状态",p.id==="post"?"post_state":"v_post","post_state_e"],["通道编号",p.id==="post"?"post_lane":"v_post_lane","hex"]];
      if(p.id==="writeback")return [...base,["写回队列项数","fifo_count","hex"],["O0 写入有效","o_v"],["O0 可接收","o_r"]];
      if(p.id==="source0"||p.id==="source1"||p.id==="add")return [...base,["读响应有效","v_res_v"],["写入结果有效","v_write_v"],["写入可接收","v_write_r"]];
      if(p.id==="done"||chapter==="waiting")return [...base,["完成事件","event_set","hex"],["已完成标记","events","hex"],["残差完成","vec_done"],["上采样完成","up_done"],["中断","irq"]];
      if(chosen.name==="DMA_LOAD")return [...base,["DMA 读取申请","dma_ar"],["DMA 返回数据有效","dma_rdata_v"],["片上写入有效","sp_wv"],["片上可接收","sp_wr"]];
      if(chosen.name==="DMA_STORE"||chosen.name==="UPSAMPLE2X")return [...base,["读数据有效","r_v"],["写数据有效","w_v"],["内存可接收","w_r"],["写入确认有效","b_v"]];
    }
    const base = [["clk · 100 MHz","clock"], ["CP state","cp_state","cp_state_e"], ["event_state","events","hex"]];
    if(chosen.name.startsWith("DMA_")) return [...base,["DMA cmd VALID","dma_v"],["DMA cmd READY","dma_r"],["ARVALID","ar_v"],["ARREADY","ar_r"],["RVALID","r_v"],["RREADY","r_r"],["sp_write VALID","sp_wv"],["sp_write READY","sp_wr"],["WVALID","w_v"],["WREADY","w_r"],["BVALID","b_v"],["dma_busy","dma_busy"]];
    if(chosen.name==="CONV2D") return [...base,["A response VALID","a_res_v"],["W response VALID","w_res_v"],["MAC input VALID","mac_v"],["MAC input READY","mac_r"],["product valid","mac_product"],["dot valid","mac_dot"],["MAC result VALID","mac_out_v"],["POST state","post_state","post_state_e"],["O write VALID","o_v"],["C0 event_set","conv_event","hex"]];
    if(chosen.name==="VEC_ADD") return [...base,["VEC state","v_state","vec_state_e"],["read VALID","v_read_v"],["read READY","v_read_r"],["response VALID","v_res_v"],["POST state","v_post","post_state_e"],["write VALID","v_write_v"],["write READY","v_write_r"],["vec_done","vec_done"]];
    if(chosen.name==="UPSAMPLE2X") return [...base,["UP state","u_state","upsample_state_e"],["ARVALID","ar_v"],["RVALID","r_v"],["RREADY","r_r"],["AWVALID","aw_v"],["WVALID","w_v"],["WREADY","w_r"],["BVALID","b_v"],["up_done","up_done"]];
    return [...base,["command VALID","fetch_v"],["command READY","fetch_r"],["dma_busy","dma_busy"],["conv_busy","conv_busy"],["event_set","event_set","hex"],["core_done","core_done"],["irq_o","irq"]];
  }
  function renderWave() {
    const svg=$("wave"), count=Number($("wave-width").value), left=155, cell=30, top=40, height=29;
    const windowStart=Math.max(chosen.start,Math.min(position-Math.floor(count/3),chosen.end-count+1));
    const windowEnd=Math.min(chosen.end,windowStart+count-1),p=currentStep();
    const samples=stepMode()?[...new Set([...p.moments,position-chosen.start].flatMap(j=>[j-1,j,j+1]).filter(j=>j>=p.start&&j<=p.end))].sort((a,b)=>a-b).map(j=>chosen.start+j):Array.from({length:windowEnd-windowStart+1},(_,j)=>windowStart+j);
    const length=samples.length,rows=waveRows(),cut=j=>j>0&&samples[j]!==samples[j-1]+1;
    const width=left+length*cell+15,totalHeight=top+rows.length*height+12;
    svg.setAttribute("viewBox",`0 0 ${width} ${totalHeight}`);svg.setAttribute("height",totalHeight);svg.replaceChildren();
    for(let j=0;j<=length;j++){
      const x=left+j*cell;svg.append(element("line",{x1:x,y1:25,x2:x,y2:totalHeight,class:"wave-grid"}));
      if(j<length&&(stepMode()||length<12||j%4===0))svg.append(element("text",{x:x+2,y:15,class:"wave-text"},"+"+(samples[j]-chosen.start)));
    }
    svg.append(element("rect",{x:left+samples.indexOf(position)*cell,y:24,width:cell,height:totalHeight-25,class:"wave-cursor"}));
    rows.forEach(([label,key,type],r)=>{
      const y=top+r*height;svg.append(element("text",{x:10,y:y+14,class:"wave-text major"},label));
      if(type){
        let j=0;
        while(j<length){
          const f=frames[samples[j]],v=num(f,key);let end=j+1;
          while(end<length&&!cut(end)&&num(frames[samples[end]],key)===v)end++;
          const x=left+j*cell,w=(end-j)*cell;svg.append(element("rect",{x:x+1,y:y+1,width:w-2,height:22,rx:3,class:"wave-bus"}));
          const text=type==="hex"?hex(v):state(f,key,type).replace(/^(CP_|POST_|VEC_|UPSAMPLE_)/,"");
          if(w>text.length*6+6)svg.append(element("text",{x:x+5,y:y+16,class:"wave-text"},text));
          const title=element("title",{},text);svg.lastChild.append(title);j=end;
        }
      }else{
        let d="",previousY=null;
        for(let j=0;j<length;j++){
          const x=left+j*cell,v=key==="clock"?1:num(frames[samples[j]],key);
          if(cut(j))previousY=null;
          if(key==="clock"){d+=`M${x} ${y+21} V${y+4} H${x+cell/2} V${y+21} H${x+cell}`;continue;}
          const level=v===1?y+4:y+21;
          if(v===null){svg.append(element("text",{x:x+9,y:y+16,class:"wave-text"},"X"));previousY=null;continue;}
          d+=previousY===null?`M${x} ${level}`:`V${level}`;d+=`H${x+cell}`;previousY=level;
        }svg.append(element("path",{d,class:"wave-signal"}));
        const readyKey=key.endsWith("_v")?key.slice(0,-1)+"r":key.endsWith("v")?key.slice(0,-1)+"r":null;
        if(readyKey&&signals[readyKey]!==undefined)for(let j=0;j<length;j++){
          const f=frames[samples[j]];if(one(f,key)&&num(f,readyKey)===0)svg.append(element("rect",{x:left+j*cell+1,y:y+1,width:cell-2,height:23,fill:"none",stroke:"#fb8da6","stroke-dasharray":"3 2"}));
        }
      }
    });
    for(let j=0;j<length;j++){
      if(cut(j)){const x=left+j*cell;svg.append(element("rect",{x:x-5,y:22,width:10,height:totalHeight-22,fill:"#0d1723",class:"wave-break"}));svg.append(element("text",{x:x-6,y:33,class:"wave-text"},"//"));const title=element("title",{},`省略 ${samples[j]-samples[j-1]-1} 个时钟`);svg.lastChild.append(title);}
      const hit=element("rect",{x:left+j*cell,y:24,width:cell,height:totalHeight-24,class:"wave-hit",role:"button","aria-label":"跳到相对周期 "+(samples[j]-chosen.start)});
      hit.addEventListener("click",()=>{pause();position=samples[j];render();});svg.append(hit);
    }
    $("wave-title").textContent=stepMode()?"本步骤的关键时刻波形":"握手与状态波形";
    $("wave-width").hidden=stepMode();$("wave-width").previousElementSibling.hidden=stepMode();
    $("wave-note").textContent=stepMode()?`本步覆盖 K+${p.start}…K+${p.end}；只显示关键时刻及相邻采样。// 表示中间时钟已省略，每列仍是一个真实周期，不能按图上宽度比较耗时。光标与框图是同一拍。`:"信号采样于上升沿前；光标对应的 VALID ∧ READY 在本上升沿被接收。虚线标记握手等待，X 表示尚未定义的信号。";
  }

  function render() {
    const f=frames[position],d=describe(f),after=frames[position+1]||f,relative=position-chosen.start;
    const p=currentStep(),steps=stepLessons[chosen.index];
    $("clock-caption").textContent=stepMode()?"当前教学步骤":"当前上升沿";
    $("clock").textContent=stepMode()?`步骤 ${p.index+1} / ${steps.length}`:"K + "+relative;
    $("absolute-clock").textContent=stepMode()?`框图真实时刻 K+${relative} · #${f[0]}`:"复位后 #"+f[0]+" · "+f[0]*T.clockNs+" ns";
    $("progress-label").textContent=stepMode()?`第 ${p.index+1} / ${steps.length} 步`:relative+" / "+(chosen.end-chosen.start);
    $("time-range").textContent="K = #"+frames[chosen.start][0]+" · "+chosen.label;
    $("duration").textContent=(chosen.end-chosen.start+1)+" 个真实上升沿 · "+((chosen.end-chosen.start)*10)+" ns 观察跨度";
    $("seek").max=stepMode()?steps.length-1:chosen.end-chosen.start;$("seek").value=stepMode()?p.index:relative;
    $("cp-state").textContent=$("show-tech").checked?state(f,"cp_state","cp_state_e"):cpPlain[state(f,"cp_state","cp_state_e")];
    $("after-state").textContent=state(after,"cp_state","cp_state_e");
    const [title,detail]=phaseText(f);$("edge-title").textContent=d.notes.length ? (d.active.has("fetch_cp") ? "接收指令与并行活动" : d.active.has("conv_event") ? "卷积完成事件到达" : d.active.has("fifo_o") ? "卷积结果写回" : "本沿发生数据传输") : title;
    const paragraph=document.createElement("p");paragraph.textContent=detail;
    $("edge-detail").replaceChildren(paragraph);
    if(d.notes.length){const ul=document.createElement("ul");ul.className="transfer-list";d.notes.forEach(note=>{const li=document.createElement("li");li.textContent=note;ul.append(li);});$("edge-detail").append(ul);}
    $("handshakes").replaceChildren(...d.handshakes.slice(0,5).map(h=>{const div=document.createElement("div");div.className="handshake"+(h.r?"":" wait");div.textContent=h.name+" = "+h.v+" / "+h.r+(h.r?" ✓":" · 等待");return div;}));
    $("prev").disabled=stepMode()?p.index===0:position===chosen.start;$("next").disabled=stepMode()?p.index===steps.length-1:position===chosen.end;$("key-next").disabled=position===chosen.end;
    $("next").textContent=stepMode()?"下一步骤 →":"下一时钟 →";$("prev").setAttribute("aria-label",stepMode()?"上一步骤":"上一时钟");
    $("key-next").hidden=stepMode();$("wave-panel").hidden=!(stepMode()||$("show-tech").checked);
    $("speed").options[0].textContent=stepMode()?"每步 4 秒":"2 周期 / 秒";$("speed").options[1].textContent=stepMode()?"每步 2 秒":"8 周期 / 秒";$("speed").options[2].textContent=stepMode()?"每步 1 秒":"30 周期 / 秒";
    renderLesson(f);updateHardware(f,d);updatePipeline(f);renderWave();
    const currentPc=one(f,"capture")?num(f,"fetch_pc"):num(f,"cp_state")===0||num(f,"cp_state")===1?null:num(f,"dispatch_pc");
    document.querySelectorAll(".program-command").forEach((b,i)=>{b.classList.toggle("chosen",i===chosen.index);b.classList.toggle("current",currentPc===i*16);});
  }
  function choose(index){
    pause();chosen=cases[index];position=chosen.start;
    $("instruction").value=chosen.name;fillCases();$("case").value=index;
    $("seek").max=chosen.end-chosen.start;
    const meta=instructionMeta[chosen.name];
    $("instruction-info").innerHTML="";
    const p=document.createElement("p");const strong=document.createElement("strong");strong.textContent=chosen.name+" · "+meta[1]+" · ";p.append(strong,meta[2]+"。"+meta[3]);
    const fields=document.createElement("p");fields.textContent=`实例 PC=${hex(chosen.index*16,4)} · imm=${hex(chosen.imm,4)} · dst=${chosen.dst===65535?"NONE":chosen.dst} · src0=${chosen.src0===65535?"NONE":chosen.src0} · src1=${chosen.src1===65535?"NONE":chosen.src1} · op=${chosen.op===65535?"NONE":chosen.op} · quant=${chosen.quant===65535?"NONE":chosen.quant}`;
    const p2=document.createElement("p");p2.textContent="16 字节编码（内存中的小端顺序）：";const code=document.createElement("code");code.textContent=chosen.hex.match(/../g).join(" ");p2.append(code);
    $("instruction-info").append(p,fields,p2);
    history.replaceState(null,"","#"+chosen.name+"-"+index);render();
  }
  function fillCases(){
    $("case").replaceChildren(...cases.filter(c=>c.name===$("instruction").value).map(c=>{const o=document.createElement("option");o.value=c.index;o.textContent="PC "+hex(c.index*16,2)+" · "+c.label;return o;}));
  }
  function pause(){if(timer)clearInterval(timer);timer=null;$("play").textContent="▶ 播放";$("play").setAttribute("aria-pressed","false");}
  function startPlayback(){
    if(stepMode()?currentStep().index===stepLessons[chosen.index].length-1:position===chosen.end)position=chosen.start;
    $("play").textContent="Ⅱ 暂停";$("play").setAttribute("aria-pressed","true");
    const delay=stepMode()?({2:4000,8:2000,30:1000}[$("speed").value]):1000/Number($("speed").value);
    timer=setInterval(()=>{
      if(stepMode()){const steps=stepLessons[chosen.index],index=currentStep().index;if(index===steps.length-1){pause();return;}position=chosen.start+steps[index+1].representative;render();if(index+1===steps.length-1)pause();}
      else{if(position>=chosen.end){pause();return;}position++;render();if(position===chosen.end)pause();}
    },delay);
  }
  function step(delta){pause();if(stepMode()){const steps=stepLessons[chosen.index],index=Math.max(0,Math.min(steps.length-1,currentStep().index+delta));position=chosen.start+steps[index].representative;}else position=Math.max(chosen.start,Math.min(chosen.end,position+delta));render();}
  function nextKey(){
    if(stepMode()){step(1);return;}
    pause();const lesson=lessons[chosen.index],old=position,current=lesson.stories[position-chosen.start];let next=position+1;
    while(next<chosen.end){const s=lesson.stories[next-chosen.start];if(s.id!==current.id||s.significant)break;next++;}
    position=Math.min(next,chosen.end);render();if(position-old>1)toast(`跳过 ${position-old-1} 个中间时钟；时间轴保留真实周期。`);
  }
  function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);toast("已导出 "+name);}
  function toast(message){$("toast").textContent=message;$("toast").classList.add("visible");setTimeout(()=>$("toast").classList.remove("visible"),2500);}
  $("instruction").replaceChildren(...Object.entries(instructionMeta).map(([name,[label,opcode]])=>{const o=document.createElement("option");o.value=name;o.textContent=opcode+" · "+name+" / "+label;return o;}));
  $("program").replaceChildren(...cases.map(c=>{const b=document.createElement("button");b.className="program-command";const strong=document.createElement("b");strong.textContent=c.name;const span=document.createElement("span");span.textContent=hex(c.index*16,2)+" · "+c.label;b.append(strong,span);b.addEventListener("click",()=>choose(c.index));return b;}));
  $("instruction").addEventListener("change",()=>choose(cases.find(c=>c.name===$("instruction").value).index));
  $("case").addEventListener("change",()=>choose(Number($("case").value)));
  $("reset").addEventListener("click",()=>{pause();position=chosen.start;render();});
  $("prev").addEventListener("click",()=>step(-1));$("next").addEventListener("click",()=>step(1));$("key-next").addEventListener("click",nextKey);
  $("play").addEventListener("click",()=>timer?pause():startPlayback());
  $("speed").addEventListener("change",()=>{if(timer){pause();startPlayback();}});
  $("seek").addEventListener("input",()=>{pause();const value=Number($("seek").value);position=chosen.start+(stepMode()?stepLessons[chosen.index][value].representative:value);render();});
  $("wave-width").addEventListener("change",renderWave);
  $("show-tech").addEventListener("change",()=>{$("technical-detail").open=$("show-tech").checked;render();});
  $("mode").addEventListener("change",()=>{pause();if(stepMode())position=chosen.start+currentStep().representative;render();});
  $("expand-step").addEventListener("click",()=>{pause();$("mode").value="cycles";$("show-tech").checked=true;$("technical-detail").open=true;render();});
  $("diagram-zoom").addEventListener("change",()=>{
    const zoom=$("diagram-zoom").value;
    $("diagram").style.width=zoom==="fit"?"100%":1400*Number(zoom)/100+"px";
  });
  $("help-button").addEventListener("click",()=>{$("guide").hidden=!$("guide").hidden;$("help-button").setAttribute("aria-expanded",String(!$("guide").hidden));});
  $("export-trace").addEventListener("click",()=>{
    const csv=["relative_cycle,task_cycle,"+T.signals.join(","),...frames.slice(chosen.start,chosen.end+1).map((f,i)=>i+","+f[0]+","+f.slice(1).map(v=>v===null?"X":"0x"+v).join(","))].join("\r\n");
    download(new Blob([csv],{type:"text/csv;charset=utf-8"}),chosen.name+"-pc"+chosen.index*16+".csv");
  });
  $("download-svg").addEventListener("click",()=>{
    const svg=$("diagram").cloneNode(true);svg.setAttribute("xmlns",ns);svg.setAttribute("width","1400");svg.setAttribute("height","810");
    // Inline computed styles so exports also work when opened via file://.
    const original=[$("diagram"),...$("diagram").querySelectorAll("*")],copies=[svg,...svg.querySelectorAll("*")];
    const properties=["fill","stroke","stroke-width","stroke-dasharray","opacity","font-family","font-size","font-weight","letter-spacing","rx","ry"];
    original.forEach((node,i)=>{const style=getComputedStyle(node);properties.forEach(property=>copies[i].style.setProperty(property,style.getPropertyValue(property)));});
    svg.style.width="1400px";svg.style.height="810px";svg.style.minWidth="0";
    const bg=element("rect",{width:1400,height:810,fill:"#0f1725"});svg.prepend(bg);
    download(new Blob([new XMLSerializer().serializeToString(svg)],{type:"image/svg+xml;charset=utf-8"}),"NPU-"+chosen.name+"-K"+(position-chosen.start)+".svg");
  });
  document.addEventListener("keydown",event=>{
    if(/INPUT|SELECT|TEXTAREA|BUTTON/.test(event.target.tagName)||event.ctrlKey||event.altKey||event.metaKey)return;
    if(event.key==="ArrowRight"){event.preventDefault();step(1);}else if(event.key==="ArrowLeft"){event.preventDefault();step(-1);}else if(event.key===" "){event.preventDefault();timer?pause():startPlayback();}else if(event.key.toLowerCase()==="n"){event.preventDefault();nextKey();}
  });
  document.addEventListener("visibilitychange",()=>{if(document.hidden)pause();});
  $("trace-meta").textContent=T.provenance+" · 输出已比对通过 · "+frames.length+" 周期";
  buildDiagram();inspect(selectedModule);
  const hashIndex=Number(location.hash.match(/-(\d+)$/)?.[1]);choose(Number.isInteger(hashIndex)&&cases[hashIndex]?hashIndex:7);
  // Read-only access supports reproducible browser validation and inspection.
  window.NPULab={cases,frames,signals,describe,lessons,stepLessons,get mode(){return $("mode").value;},get teachingStep(){return currentStep();},get selection(){return chosen;},get position(){return position;}};
})();
