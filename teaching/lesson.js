"use strict";
// Educational descriptions are attached to recorded states, never generated
// by a pretend clock. Ownership tags keep later/earlier work out of the lesson.
window.createNPULessons = ({frames, cases, num, one, fire, state, signedLow}) => {
  const name = (f,key,type) => state(f,key,type);
  const goals = [
    "什么也不计算，直接继续下一条指令。", "把输入 3 从 DDR 搬到 A0，供后面的卷积读取。",
    "把权重 2 和其他 0 权重搬到 W0，供卷积使用。", "把 8 个通道的偏置搬到 W0；本例偏置全是 0。",
    "把卷积的 8 份缩放参数搬到 W0；本例缩放比例都为 1。", "把残差相加使用的缩放参数搬到 W0；本例比例为 1。",
    "等输入和全部计算参数准备好，再开始卷积。", "按权重计算：3 × 2 = 6，再把结果保存到 O0。",
    "等卷积结果真正写入 O0，才允许后面的残差相加。", "把 O0 中的 6 与 A0 中保留的 3 相加，得到 9。",
    "把 O0 中的结果 9 写回 DDR，供上采样读取。", "等结果写回 DDR 后，再允许上采样读取它。",
    "把一个像素复制成 2 × 2 的四个像素，通道 0 都为 9。", "确认所有工作完成，再通知处理器收工。",
  ];
  const events = [[1,"输入 A0"],[2,"输入 A1"],[4,"参数 W0"],[8,"参数 W1"],[16,"卷积 O0"],[32,"卷积 O1"],[64,"写回 O0"],[128,"写回 O1"]];
  const df = {
    DF_REQUEST_OPERATOR:["申请读取计算说明", "让内存接口取回这次计算的尺寸、步长和分块信息。"],
    DF_WAIT_OPERATOR:["等计算说明回来", "正在等待尺寸与步长信息；这里读的是说明书，还没搬输入数据。"],
    DF_REQUEST_SOURCE:["申请读取输入说明", "询问输入放在 DDR 的哪里、形状是什么、每行间隔多少字节。"],
    DF_WAIT_SOURCE:["等输入说明回来", "等输入的位置与形状信息返回，才能确定搬运范围。"],
    DF_REQUEST_DESTINATION:["申请读取输出说明", "询问输出在 DDR 的位置与形状。"],
    DF_WAIT_DESTINATION:["等输出说明回来", "等目标位置和可用空间的信息返回。"],
    DF_REQUEST_QUANT:["申请读取缩放参数说明", "查询缩放参数放在哪里、共有多少份。"],
    DF_WAIT_QUANT:["等参数位置回来", "正在等缩放参数的地址说明，还没读取参数本身。"],
    DF_REQUEST_SEGMENT:["申请读取分段说明", "查看这次输入由哪些存储片段组成。"],
    DF_WAIT_SEGMENT:["等分段说明回来", "等待各个数据片段的存放位置。"],
  };
  // Exact purposes of the address states in npu_dma_agu.sv.
  const address = {
    AGU_ACT_LOCAL_H_STRIDE:"算卷积窗口在高度方向跨过多少行", AGU_ACT_LOCAL_H_DILATION:"算卷积核在高度方向还需要覆盖多少行",
    AGU_ACT_LOCAL_W_STRIDE:"算卷积窗口在宽度方向跨过多少列", AGU_ACT_LOCAL_W_DILATION:"算卷积核在宽度方向还需要覆盖多少列",
    AGU_ACT_LOCAL_C:"算一个像素的全部输入通道占多少字节", AGU_ACT_RAW_H:"算输入块从哪一行开始", AGU_ACT_RAW_W:"算输入块从哪一列开始",
    AGU_ACT_CLAMP:"检查输入范围有没有超出原始特征图", AGU_ACT_GEOMETRY:"整理有效数据和填充区域的尺寸",
    AGU_ACT_CLEAR_PLANE:"算一行输入占多少字节", AGU_ACT_CLEAR:"算目标输入区域一共需要清零多少字节",
    AGU_ACT_X:"算每次搬运的有效通道占多少字节", AGU_ACT_PAD_ROW:"算顶部填充在工作台中占了多少位置",
    AGU_ACT_PAD_BYTES:"把填充位置转换成字节偏移", AGU_ACT_DST_CHANNEL:"算目标通道在工作台中的位置",
    AGU_ACT_SPAD_OFFSET:"把填充和通道位置相加，确定片上起点", AGU_ACT_EXT_H:"算 DDR 中起始行的字节偏移",
    AGU_ACT_EXT_W:"算 DDR 中起始列的字节偏移", AGU_ACT_EXT_C:"算 DDR 中起始通道的字节偏移",
    AGU_ACT_END_Z:"算最后一行在 DDR 中的位置", AGU_ACT_END_Y:"算最后一个像素在 DDR 中的位置",
    AGU_ACT_SPAD_END_Z:"算最后一行在工作台中的位置", AGU_ACT_SPAD_END_Y:"算最后一个像素在工作台中的位置",
    AGU_ADD_SPAD_OFFSET:"把目标起点加到片上范围里", AGU_ADD_SPAD_X:"加上最后一块数据的长度，确定结束位置",
    AGU_CHECK_EXTERNAL:"检查 DDR 中的读取或写入范围是否越界", AGU_FINAL:"检查片上范围是否放得下，整理搬运请求",
    AGU_RETURN:"把算好的搬运路线交给 DMA 引擎",
    AGU_WEIGHT_KERNEL_H:"算权重在卷积核高度方向有多少块", AGU_WEIGHT_KERNEL_W:"算权重在卷积核宽度方向有多少块",
    AGU_WEIGHT_TILE:"算这一块卷积权重占多少字节", AGU_WEIGHT_ORIGIN:"算这一块权重在 DDR 中的起点",
    AGU_LAYOUT_BIAS:"安排偏置在 W0 中的对齐位置", AGU_LAYOUT_BIAS_RETURN:"确定偏置的起点与长度",
    AGU_LAYOUT_CONV:"安排卷积缩放参数在 W0 中的位置", AGU_LAYOUT_CONV_RETURN:"确定卷积缩放参数的起点与长度",
    AGU_LAYOUT_VECTOR:"安排残差缩放参数在 W0 中的位置", AGU_LAYOUT_VECTOR_RETURN:"确定残差缩放参数的起点与长度",
    AGU_STORE_X:"算每个输出像素需要写回多少字节", AGU_STORE_SPAD_Y:"算 O0 中相邻输出像素的间隔",
    AGU_STORE_SPAD_Z:"算 O0 中相邻输出行的间隔", AGU_STORE_EXT_H:"算输出在 DDR 中起始行的偏移",
    AGU_STORE_EXT_W:"算输出在 DDR 中起始列的偏移", AGU_STORE_EXT_C:"算输出在 DDR 中起始通道的偏移",
    AGU_STORE_END_Z:"算最后一行输出在 DDR 中的位置", AGU_STORE_END_Y:"算最后一个输出像素在 DDR 中的位置",
    AGU_STORE_SPAD_END_Z:"算最后一行输出在 O0 中的位置", AGU_STORE_SPAD_END_Y:"算最后一个输出像素在 O0 中的位置",
  };
  const post = {
    POST_MULTIPLY:["乘上缩放系数", "用整数乘法实现数值缩放；本例的整体比例为 1。", "得到用于后续缩放的中间乘积。"],
    POST_MAGNITUDE:["整理符号与绝对值", "把正负号与大小分开，方便对负数也正确舍入。", "准备按统一规则缩小数值。"],
    POST_SHIFT:["缩回正确的数值尺度", "前面把比例表示成一个大整数，这一步用移位除回去。", "本例除以 2³⁰，与前面的系数配合得到比例 1。"],
    POST_ROUND_DECIDE:["判断余数该不该进位", "缩放可能留下小数，需要决定保留哪个整数。", "按四舍六入、正好一半取偶数的规则作决定。"],
    POST_ROUND_APPLY:["执行整数舍入", "按上一拍的决定保留或进位。", "得到舍入后的整数结果。"],
    POST_CLAMP:["把数值限制在可保存的范围", "这个 NPU 的激活范围是 −2048 到 2047，超出就夹在边界上。", "本例 6 或 3 都在范围内，不会被改变。"],
    POST_ACTIVATE:["执行选定的激活规则", "激活用于调整特征数值。本例没有启用激活，直接保留结果。", "当前通道处理完成，再处理下一个通道。"],
    POST_TANH_CAPTURE:["接收激活查表结果", "若使用 tanh 激活，需要等待查表结果。", "把查到的结果保存到当前通道。"],
  };
  function entry(chapter,chapterTitle,id,title,action,why,result,focus,extra={}) {
    return {chapter,chapterTitle,id,title,action,why,result,focus,significant:false,...extra};
  }
  function multiplication(f,prefix,title,chapter,chapterTitle,id,focus) {
    const a=num(f,prefix+"_mul_a"),b=num(f,prefix+"_mul_b");
    const done=one(f,prefix+"_mul_done"),busy=one(f,prefix+"_mul_busy");
    const expression=a!==null&&b!==null?`${a} × ${b}`:"行长度";
    const detail=done?"乘法结果已准备好，本沿把它交给地址计算器。":busy?`正在做第 ${num(f,prefix+"_mul_step")+1} / 16 轮移位与相加。`:"本沿启动这次地址乘法，下一拍开始逐轮计算。";
    return entry(chapter,chapterTitle,id,title,`${title}。当前在算 ${expression}；${detail}`,
      "搬运或读取之前，必须把行、列、通道的位置换算成字节地址。地址乘法器逐轮移位、相加，节省 FPGA 乘法器资源。",
      done?`这次计算得到 ${num(f,prefix+"_mul_result")}，接着计算下一项地址或范围。`:"16 轮计算结束并交接结果后，才能继续下一项。",focus,
      {significant:done,multiplication:true,calculation:expression,mulPrefix:prefix,mulRound:busy?num(f,prefix+"_mul_step")+1:done?16:0});
  }
  function computePost(f,key,laneKey,chapter,chapterTitle,focus,vector=false) {
    const st=name(f,key,"post_state_e"),lane=num(f,laneKey),description=post[st];
    if(!description)return null;
    return entry(chapter,chapterTitle,st+"-"+lane,`处理第 ${lane+1} / 8 个通道：${description[0]}`,
      `${description[1]} 当前是通道 ${lane}，本例${lane===0?(vector?"残差值为 3":"卷积累加值为 6"):"这个通道的数值为 0"}。`,
      vector?"两条路径的数值尺度要对齐，才可以把残差和卷积结果相加。":"卷积的累加值需要转换成后面模块能保存和使用的特征值。",
      description[2],focus,{significant:true,lane});
  }
  function story(c,i,dispatch,startUnit,upWrites,writeback) {
    const f=frames[i],isLoad=c.name==="DMA_LOAD",isDMA=isLoad||c.name==="DMA_STORE";
    const done=entry("done","完成","done","这条指令已经完成",goals[c.index],"本例的结果和完成信号均来自实际 RTL 仿真。","后续指令现在可以使用它的结果。",["events"],{significant:true});
    if(i===c.end){
      const ending={...done,action:"已完成："+goals[c.index],copied:c.name==="UPSAMPLE2X"?4:undefined,focus:c.name==="END"?["irq"]:isDMA?["dma","events"]:c.name==="UPSAMPLE2X"?["up"]:c.name==="VEC_ADD"?["vec"]:["cp","events"]};
      if(c.name==="WAIT"){ending.action="指定工作的完成指示灯全亮，等待检查已经通过。";ending.result="继续下一条指令；WAIT 不会关掉这些灯。";}
      if(c.name==="NOP"){ending.action="空操作已经处理完，没有启动计算或搬运。";ending.result="继续接收下一条指令。";ending.focus=["cp"];}
      if(c.name==="END"){ending.title="整个任务已结束，处理器收到通知";ending.action="14 条指令全部处理完，完成中断已经置位。";ending.result="处理器可以读取完成状态和输出，或启动新任务。";}
      return ending;
    }
    if(i===c.start)return entry("accept","接收命令","capture","控制器读到你的这条命令",`刚收到“${c.label}”这项工作，先把命令保存下来。`,"命令必须先被接收并检查，相关硬件才能开始执行。","下一拍尝试把这项工作交给对应的单元。",["fetch","cp"],{significant:true});
    if(dispatch!==null&&i<=dispatch){
      const prefix=isDMA?"dma":c.name==="CONV2D"?"conv":"vec";
      const target=isDMA?"DMA 前端":"执行前端";
      return entry("accept","接收命令",fire(f,prefix)?"dispatch":"wait-dispatch",fire(f,prefix)?`把命令交给${target}`:`等${target}腾出位置`,
        fire(f,prefix)?`这条命令在本沿被${target}接收。`: `${target}还没有接单，控制器暂时保留这条命令。此时图上其他单元可能正在处理前一条指令。`,
        "一个单元只有准备好接收新工作时，命令才能交过去。",fire(f,prefix)?(isDMA||c.name==="CONV2D"?"控制器可以继续处理后续指令，当前工作在后台继续。":"控制器将等待这项同步工作完成。"):`等它接单后，才开始准备“${c.label}”。`,["cp",isDMA?"df":"ef"],{significant:fire(f,prefix)});
    }
    if(isDMA){
      const engine=name(f,"de_state","dma_engine_state_e");
      if(num(f,"dma_engine_tag")===c.index&&engine!=="DE_IDLE"){
        const target=c.index===1?"A0 输入区":"W0 参数区";
        const dest=isLoad?target:"DDR 外部内存";
        if(engine==="DE_DONE")return entry("done","完成","dma-done","搬运引擎确认完成",`“${c.label}”已经搬完。`,"最后一份数据完成之后，控制器才能确认这次搬运结束。",c.imm&255?"设置这条命令指定的完成指示灯，供 WAIT 检查。":"这条命令没有指定完成事件；引擎回到可接单状态。",["dma","events"],{significant:true});
        if(engine==="DE_CLEAR")return entry("transfer","实际搬运","clear","先把输入工作台清零",`向 A0 写入 0，为稍后搬入的输入留出干净的区域。`,"卷积可能需要边界填充，先清零可以保证没有有效输入的位置都是 0。", "随后把真实输入 3 覆盖到有效位置。",["dma","a"],{significant:fire(f,"sp_w")});
        if(engine==="DE_PLAN_SIZE"||engine==="DE_PLAN_BURST")return entry("transfer","实际搬运",engine,"安排这趟搬运如何分批", "确定一拍搬多少字节，以及这次连续搬多少拍。", "内存接口对传输大小、地址对齐和边界有要求。", "准备好之后，发出本次读取或写入申请。",["dma"]);
        if(engine==="DE_READ_ADDRESS")return entry("transfer","实际搬运","read-address",fire(f,"ar")?"内存接收了读取申请":"向 DDR 申请读取数据",`申请取回“${c.label}”对应的数据；本例每个返回数据块是 8 字节。`,"先告诉内存从哪里读、读多少，数据才会返回。",fire(f,"ar")?"下一步等数据返回。":"申请被接收后，才能等待数据返回。",["dma","read","ddr"],{significant:fire(f,"ar")});
        if(engine==="DE_READ_DATA")return entry("transfer","实际搬运","read-data",fire(f,"sp_w")?`把返回的数据放进${dest}`:"等 DDR 返回下一块数据",fire(f,"sp_w")?`收到 8 字节数据并写入${dest}。${c.index===1&&num(f,"sp_waddr")===0?"输入通道 0 的值 3 在本沿落到 A0。":"这次搬的是数据或计算参数本身。"}`:"读取申请已经发出，等待内存准备好下一个 8 字节块。",
          "说明书和地址已准备好，现在才是在搬运实际数据。",one(f,"r_last")&&fire(f,"r")?"最后一块已经收到，接着报告完成。":"继续接收剩余数据块。",["ddr","read","dma",c.index===1?"a":"w"],{significant:fire(f,"sp_w")});
        const storeStates={
          DE_WRITE_ADDRESS:["向 DDR 申请写入位置","把结果在 DDR 中的目标地址和写入长度交给内存接口。","申请被接收后，再发送结果数据。"],
          DE_WRITE_SP_REQUEST:["向 O0 要一块计算结果","请求从成品区读取下一块 8 字节数据。","同步存储器将在后续周期返回结果。"],
          DE_WRITE_SP_RESPONSE:["等 O0 读出结果","等片上存储器把请求的输出数据送回来。","把结果存进 DMA 缓冲，准备向 DDR 发送。"],
          DE_WRITE_DATA:[fire(f,"w")?"把结果数据送到 DDR":"等内存接口接收结果",fire(f,"w")?`本沿发送 8 字节结果，低 16 位为 ${signedLow(f,"w_data",16)}。其中第一块包含通道 0 的值 9。`:"DMA 已备好数据，等接收方允许交接。","全部结果数据发送后，还要等写入确认。"],
          DE_WRITE_RESPONSE:[fire(f,"b")?"收到写入确认":"等 DDR 的写入确认",fire(f,"b")?"内存接口在本沿返回确认，这次写事务完成。":"数据已发出去，但内存接口还没返回这次写事务的确认。","收到确认后，DMA 才报告搬运完成。"],
        };
        const s=storeStates[engine];
        if(s)return entry("transfer","实际搬运",engine,...s.slice(0,2),"输出数据要经过片上读取、发送和确认，几个步骤不能在一拍里全部完成。",s[2],["dma","o","write","ddr"],{significant:fire(f,"sp_r")||fire(f,"sp_res")||fire(f,"w")||fire(f,"aw")||fire(f,"b")});
      }
      if(num(f,"dma_frontend_tag")===c.index){
        const agu=name(f,"agu_state","agu_state_e");
        if(agu!=="AGU_IDLE"){
          const title=address[agu]||"整理搬运位置与范围";
          if(agu in address&&num(f,"agu_mul_a")!==null&&(one(f,"agu_mul_busy")||one(f,"agu_mul_done")||/^AGU_(ACT_(LOCAL_|RAW_|CLEAR|X$|PAD_|DST_CHANNEL|EXT_|END_|SPAD_END_)|STORE_(X|SPAD_[YZ]|EXT_|END_|SPAD_END_)|WEIGHT_)/.test(agu)))return multiplication(f,"agu",title,"address","计算搬运位置",agu,["df"]);
          return entry("address","计算搬运位置",agu,title,`${title}，把各项地址信息整理好。`,"搬运器需要明确的起点、长度与结束边界，不能直接把“一个像素”当作字节地址。", agu==="AGU_RETURN"?"算好的请求可交给 DMA 引擎。":"继续检查或计算下一项搬运信息。",["df"]);
        }
        const s=df[name(f,"df_state","dma_frontend_state_e")];
        if(s)return entry("descriptors","读取任务说明",name(f,"df_state","dma_frontend_state_e"),s[0],s[1],"一条命令只给出了描述符编号，需要先查到数据的形状和存放位置。","拿齐说明后，地址计算器才能规划搬运路线。",["df","mem","block"]);
      }
      return entry("descriptors","读取任务说明","handover","整理并交接搬运信息","前端正在接收说明或等待搬运引擎接收算好的请求。","前端负责准备路线，DMA 引擎负责真正搬运，它们之间也需要交接。","接单后开始执行搬运请求。",["df","dma"]);
    }
    if(c.name==="WAIT"){
      const mask=c.imm&255,bits=(num(f,"events")||0)|(num(f,"event_set")||0),ready=(bits&mask)===mask;
      const lamps=events.filter(([bit])=>mask&bit).map(([bit,label])=>({label,ready:!!(bits&bit)}));
      return entry("waiting","检查完成指示灯",ready?"wait-ready":"wait-lamps",ready?"需要的指示灯全亮了":"停在这里，等指定工作完成",lamps.map(l=>`${l.label}：${l.ready?"已完成":"还没完成"}`).join("；")+"。",
        "后面的指令要使用这些结果，提前执行可能读到旧数据。WAIT 自己不计算，也不搬数据。",ready?"本沿通过检查，允许继续下一条指令；这些灯保持亮着。":"其他执行单元继续工作，WAIT 在后续时钟重新检查。",["cp","events"],{lamps,significant:ready});
    }
    if(c.name==="NOP")return entry("done","直接继续","nop","这一步不做运算","这条空操作完成检查后，直接算作处理完毕。","程序允许安排一个不启动运算单元的命令。","继续接收下一条指令，不是让 NPU 休眠。",["cp"],{significant:true});
    if(c.name==="END")return entry("done","结束任务",one(f,"core_done")?"core-done":"end-wait",one(f,"core_done")?"把完成消息交给状态寄存器":"检查是否可以收工",one(f,"core_done")?"运算核心发出“任务已完成”的消息。":"确认搬运器、卷积、残差相加和上采样单元都已空闲。","最后一条命令必须等所有在途工作结束，才能宣布整个任务完成。",one(f,"core_done")?"状态寄存器更新完成标志，并根据配置通知处理器。":"确认完成后，发出任务完成消息。",["cp","csr","irq"],{significant:one(f,"core_done")});
    if(startUnit!==null&&i<=startUnit){
      const ef=name(f,"ef_state","execution_frontend_state_e");
      return entry("descriptors","读取任务说明",ef,i===startUnit?"把计算任务交给运算单元":"查清这次计算要用的数据",i===startUnit?"命令与尺寸信息在本沿被对应运算单元接收。":ef.includes("SOURCE")?"查输入数据在 DDR 中的位置和形状。":ef.includes("DEST")?"查输出数据在 DDR 中的位置和形状。":"查询卷积或残差计算的尺寸、步长与分块说明；如果缓存已有说明，可直接使用。",
        "运算单元需要知道数据的尺寸与存放位置，才能正确读取和计算。","接下来做地址准备，随后开始读取实际数据。",["ef","mem"],{significant:i===startUnit});
    }
    if(c.name==="CONV2D"){
      if(num(f,"conv_event"))return {...done,title:"卷积结果已存好，点亮完成指示灯",action:"O0 已收到结果 6，写回队列也已排空。",why:"先存好结果，再宣布完成，后面的残差相加才能安全读取。",focus:["o","fifo","events"]};
      if(writeback!==null&&i>writeback)return entry("writeback","保存结果","completion-pending","结果已经存好，完成消息正在返回","O0 已收到结果 6，等待流水线和写回队列的完成消息汇总。","数据写入和控制器收到完成事件之间还有寄存器交接。","接着点亮卷积完成指示灯，让 WAIT 继续。",["fifo","events"]);
      if(fire(f,"o"))return entry("writeback","保存结果","conv-write","把卷积结果 6 存进 O0","写回队列将 8 个通道的结果写入成品区；通道 0 为 6，其余通道为 0。","后面的残差相加要从 O0 读取这份结果。","等流水线与写回队列全部结束，再点亮卷积完成指示灯。",["fifo","o"],{significant:true});
      if(fire(f,"out"))return entry("writeback","保存结果","conv-fifo","把处理好的结果放进写回队列","后处理完成，先将 8 个通道的结果交给两项写回队列。","计算结果和存储写入之间用小队列衔接，接收结果与真正写入可以分开。","接着由队列把结果写入 O0。",["post","fifo"],{significant:true});
      const p=computePost(f,"post_state","post_lane","post","调整数值",["post"]);if(p)return p;
      if(fire(f,"post"))return entry("post","调整数值","post-start","把累加结果交给后处理","接收 8 个通道的整数累加结果，准备逐通道加偏置、缩放和限制范围。","累加结果还需要转换成模型使用的特征数值。","本例偏置为 0、缩放比例为 1、无激活，因此通道 0 最终仍是 6。",["buffer","post"],{significant:true});
      if(fire(f,"mac_out"))return entry("calculate","相乘并相加","mac-result","接住算好的结果 6","乘法与加法树已经得到 3 × 2 = 6，把结果保存到寄存缓冲中。","缓冲先接住结果，慢速后处理可以稍后再取走。","下一步送入后处理，把结果变成可保存的特征值。",["mac","buffer"],{significant:true});
      const pipeline=[["mac_dot","累加这一组乘积的总和","得到 8 个输出通道的整数累加结果。"],["mac_l2","合并加法树的最后两组部分和","把一个输出通道对应的 8 项乘积合成总和。"],["mac_l1","继续合并上一层的部分和","把已经两两相加的结果继续合并。"],["mac_product","把乘积两两相加","乘法结果进入第一层加法树，开始合并。"]];
      for(const [key,title,action]of pipeline)if(one(f,key))return entry("calculate","相乘并相加",key,title,action+" 本例只有 3 × 2 这一项非零，其余乘积都是 0。","8 项乘积通过多级加法合成结果，每一级占一个时钟。","下一拍交给后面的流水级；最后得到通道 0 的结果 6。",["mac"],{significant:true});
      if(fire(f,"mac"))return entry("calculate","相乘并相加","mac-input","把输入 3 和权重 2 送入乘法器","激活与权重都已读出，并与“是否第一组、是否最后一组”的信息对齐。","卷积需要把对应的输入和权重相乘，然后汇总。","本沿接收后，乘法、加法树与累加依次处理，随后得到 6。",["controller","mac"],{significant:true});
      const parameter=name(f,"param_state","param_state_e");
      if(parameter!=="PARAM_IDLE"&&parameter!=="PARAM_READY")return entry("parameters","读取计算参数",parameter,parameter.includes("BIAS")?"先准备后处理用的偏置":"先准备后处理用的缩放参数",parameter.includes("RESPONSE")?"等 W0 读出参数；这些是计算用的配方，不是输入特征。":"从 W0 请求读取偏置或各通道的缩放参数。","卷积结果产生后要马上做后处理，提前把参数准备好。","本例偏置全为 0、缩放比例全为 1。",["w","controller"],{significant:fire(f,"w_req")||fire(f,"w_res")});
      const setup=name(f,"ctrl_state","setup_state_e");
      if(setup!=="SETUP_IDLE"&&setup!=="SETUP_RUN")return multiplication(f,"ctrl",{SETUP_ROW_BYTES:"算输入一行占多少字节",SETUP_CAPACITY:"算这一块输入一共占多少空间",SETUP_ROW_ADVANCE:"算卷积换到下一行时地址要跳多远"}[setup]||"准备卷积读取地址","address","计算读取位置",setup,["controller"]);
      if(fire(f,"a_res")||fire(f,"w_res"))return entry("inputs","读取输入与权重","conv-response","取回卷积要用的输入和权重","从 A0 读 8 个输入通道，从 W0 读 8 × 8 个权重；等两边都到齐后一起送入乘法器。","输入和权重必须对应到同一组通道，不能拿错配方。","通道 0 的输入为 3，对应权重为 2。",["a","w","controller"],{significant:true});
      if(one(f,"post_v"))return entry("post","调整数值","buffer-wait","缓冲中的结果等后处理接收","结果已由缓冲保存，等待后处理准备好。","后处理逐通道工作，接收新结果前需要有空位。","允许交接后，逐通道调整这份结果。",["buffer","post"]);
      return entry("inputs","读取输入与权重","conv-read","申请或等待本地输入与权重","请求 A0 和 W0 返回当前卷积位置的数据。","片上存储器也是同步读取，需要经历请求与响应周期。","两边数据到齐后，进入相乘与相加的流水线。",["a","w","controller"],{significant:fire(f,"a_req")||fire(f,"w_req")});
    }
    if(c.name==="VEC_ADD"){
      const v=name(f,"v_state","vec_state_e");
      if(one(f,"vec_done"))return {...done,title:"残差相加完成",action:"O0 中已保存 6 + 3 = 9，执行前端收到完成消息。",focus:["vec","ef"]};
      if(v.startsWith("VEC_SETUP"))return multiplication(f,"v",{VEC_SETUP_ROW_BYTES:"算输出一行的字节长度",VEC_SETUP_PAD_ROWS:"算上方填充占的空间",VEC_SETUP_PAD_COLUMNS:"算左侧填充占的空间",VEC_SETUP_ROW_ADVANCE:"算残差换行时的地址间隔"}[v],"address","计算读取位置",v,["vec"]);
      if(v.includes("PARAM"))return entry("inputs","读取两份特征",v,"取出残差的缩放配方","从 W0 读取残差专用的量化参数。本例的缩放比例为 1。","两条路径的数值尺度一致后，才可以直接相加。","随后读取 O0 中的卷积结果与 A0 中保留的残差。",["w","vec"],{significant:fire(f,"v_read")||fire(f,"v_res")});
      if(v.includes("SRC0")||v.includes("SRC1")){const original=v.includes("SRC1");return entry("inputs","读取两份特征",v,original?"取回保留的残差 3":"取回卷积结果 6",original?"从 A0 读取旁路保留的输入，通道 0 为 3。":"从 O0 读取刚算好的卷积结果，通道 0 为 6。","把两份对应位置的特征都读到本地，才能相加。","两份数据准备好后，先把残差缩放到合适的尺度。",[original?"a":"o","vec"],{significant:fire(f,"v_read")||fire(f,"v_res")});}
      if(fire(f,"v_write"))return entry("add","相加并保存","vec-write","把 6 与 3 相加，保存结果 9","缩放后的残差与卷积结果按对应通道相加，并写回 O0。","保留前面路径的特征信息，让它继续参与后面的计算。","通道 0 得到 9；如果超出 −2048…2047，会夹到边界。",["vec","o"],{significant:true});
      const p=computePost(f,"v_post","v_post_lane","scale","对齐残差尺度",["vec"],true);if(p)return p;
      return entry("scale","对齐残差尺度",v,"把残差交给独立的缩放单元","准备逐通道缩放残差；本例 3 缩放后还是 3。","残差相加单元有自己的后处理电路，与卷积后处理独立。","缩放结果到齐后，与卷积结果 6 相加。",["vec"]);
    }
    if(c.name==="UPSAMPLE2X"){
      const u=name(f,"u_state","upsample_state_e"),sent=upWrites[i-c.start];
      if(one(f,"up_done"))return {...done,title:"四个像素全部复制完成",action:"DDR 中得到 2 × 2 输出；四个像素的通道 0 都是 9。",focus:["up","ddr"],copied:4};
      if(u==="UPSAMPLE_SETUP_ROW")return multiplication(f,"u","算源数据一行需要读取多少字节","address","计算行长度",u,["up"]);
      if(u==="UPSAMPLE_PLAN_READ"||u==="UPSAMPLE_READ_ADDRESS")return entry("read","读入原始像素",u,"向 DDR 申请读入原来的一行","原来只有一个像素，8 个通道占 16 字节，将分成两块 8 字节读取。","先把一行存到本地缓冲，复制时就不用反复去 DDR 读取。","读取申请被接收后，等这一行的数据返回。",["read","up","ddr"],{significant:fire(f,"ar"),copied:0});
      if(u==="UPSAMPLE_READ_DATA")return entry("read","读入原始像素",u,fire(f,"r")?"把原始像素存进行缓冲":"等原始像素从 DDR 返回",fire(f,"r")?"收到一个 8 字节块，把它保存到上采样单元的行缓冲。":"等内存返回剩余数据块；原像素的通道 0 为 9。","上采样要复制同一个像素，先在本地保存一份。","一行读齐后，开始横向和纵向复制。",["read","up"],{significant:fire(f,"r"),copied:0});
      if(u==="UPSAMPLE_DONE")return {...done,title:"上采样已写完，准备报告完成",action:"两行输出都已写完并收到内存确认。",focus:["up"],copied:4};
      const row=num(f,"u_row_copy")+1,pixel=Math.min(4,Math.floor(sent/2)+1);
      const title=u==="UPSAMPLE_WRITE_RESPONSE"?(fire(f,"b")?`收到第 ${row} 行的写入确认`:`等第 ${row} 行的写入确认`):fire(f,"w")?`复制第 ${pixel} / 4 个输出像素`:u==="UPSAMPLE_WRITE_PREFETCH"?"从行缓冲取出要复制的数据":"安排复制后的输出写入位置";
      return entry("copy","复制并写回",u+"-"+num(f,"u_row_copy"),title,fire(f,"w")?`本沿发送第 ${sent+1} / 8 个数据块。一个完整像素需要两块；当前复制到输出第 ${Math.floor((pixel-1)/2)+1} 行、第 ${(pixel-1)%2+1} 列。`:u==="UPSAMPLE_WRITE_RESPONSE"?"数据已发送，正在等这一行的写入确认。":"把同一个像素重复写两次，再把这一行重复写一次。",
        "最近邻上采样直接复制特征，不做卷积；这里扩大的是特征图的高和宽。","写出 4 个像素并收到两行的写入确认后，才报告上采样完成。",["up","write","ddr"],{significant:fire(f,"w")||fire(f,"aw")||fire(f,"b"),copied:Math.floor((sent+(fire(f,"w")?1:0))/2),activePixel:pixel-1});
    }
    return done;
  }
  return cases.map(c=>{
    const prefix=c.name.startsWith("DMA_")?"dma":c.name==="CONV2D"?"conv":c.name==="VEC_ADD"||c.name==="UPSAMPLE2X"?"vec":null;
    let dispatch=null,startUnit=null,writeback=null;
    for(let i=c.start;i<=c.end;i++){
      if(prefix&&dispatch===null&&fire(frames[i],prefix))dispatch=i;
      const unit=c.name==="CONV2D"?"conv_start":c.name==="VEC_ADD"?"vec_start":c.name==="UPSAMPLE2X"?"up_start":null;
      if(unit&&startUnit===null&&fire(frames[i],unit))startUnit=i;
      if(c.name==="CONV2D"&&writeback===null&&fire(frames[i],"o"))writeback=i;
    }
    const upWrites=[];let sent=0;
    for(let i=c.start;i<=c.end;i++){upWrites.push(sent);if(c.name==="UPSAMPLE2X"&&fire(frames[i],"w"))sent++;}
    const stories=[];
    for(let i=c.start;i<=c.end;i++)stories.push(story(c,i,dispatch,startUnit,upWrites,writeback));
    const runs=[];
    stories.forEach((s,j)=>{
      if(j===0||s.id!==stories[j-1].id)runs.push({id:s.id,start:j,end:j});else runs.at(-1).end=j;
      s.run=runs.at(-1);
    });
    const chapters=[];
    stories.forEach((s,j)=>{
      let chapter=chapters.find(p=>p.id===s.chapter);
      if(!chapter){chapter={id:s.chapter,title:s.chapterTitle,start:j,cycles:0};chapters.push(chapter);}
      chapter.cycles++;s.chapterIndex=chapters.indexOf(chapter);
    });
    return {goal:goals[c.index],stories,runs,chapters,dispatch,startUnit};
  });
};
