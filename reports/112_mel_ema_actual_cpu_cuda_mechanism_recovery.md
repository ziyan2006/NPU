# EMA 真实首步 CPU–CUDA 机制恢复进展

更新时间：2026-10-04T19:24Z。NONRELEASE；发布选择 NONE；正式学生训练仍为4500步，未启动4501..5000正式500步续训。

## 本轮实际完成

- 215：一份真实六槽音频输入、完整原模型六次微批前向、完整原损失路径，零 autograd/Adam.step/CUDA；全 raw/原Adam/EMA/日程/语义游标/CPU-Python-NumPy RNG逐位类型恢复。实际CPU2线程。5次真实受守卫解码子调用。21新检查、实际审计及独立只读复核均真实native exit0。
- 216：真实CUDA初始化，安装运行时与不可变4500源严格FP32身份相同；完整原CPU/Python/NumPy/CUDA RNG逐位类型恢复。没有模型/Adam构造或更新。14新检查、实际观察及独立只读复核均真实native exit0。17新增原生文件当时只是映射后观察，不倒推认证。
- 218：在新进程启动前持有此前实际观察的246个原生物理文件，逐实际加载事件验文件ID/字节/SHA并持有到完整EXIT；CPU和CUDA各266次文件加载事件，各3个实际进程EXIT0。没有用basename猜加载路径，也不代表所有未来输入条件分支已穷尽。
- 218 CPU2与源CUDA4线程分别完成真实4501首步原Adam、固定FP32 EMA、完整磁盘恢复及逐位重放、在真实Adam+EMA计数+CPU/Python/NumPy/CUDA RNG变异后注入BaseException并整组逐位回滚。CUDA不改原Adam健康检查或算术。
- 每设备仅一个唯一counter4500/六槽（逻辑下一更新4501），使用215已经认证的完整输入和目标PCM，不重新解码旧批。主更新、独立原194参考、磁盘重放和故障重复各6槽：每设备24次实际前后向、4次实际Adam.step；不是4个独立输入或正式训练步。
- 两设备原194独立首raw/完整原Adam参考逐位一致，EMA真实非零且无别名；输入与目标显式设备往返PCM逐位一致，完整typed元数据及输入digest跨设备相同。CPU是明确迁移机制，不宣称CPU与CUDA更新数值相同。
- 218独立reader每设备只读3份新完整状态，核各自own seal/全字段类型位、原Adam所有选项、累计日程/旧stop4500、完整原CUDA RNG、独立分开FP32 EMA算术、全部加载文件绑定和实际EXIT。reader无模型/Adam/前后向/音频/CUDA初始化。

## 恢复中保留的失败/边界

217的21新检查与CPU尝试真实exit0；但原 common.py 在损失模块首次导入时设20线程，覆盖早期配置。217 CPU证据实际20线程，不能当指定CPU2或源CUDA4机制门槛。检测后以独立拒绝标记让新217 CUDA诊断安全返回；worker记录forward/backward/Adam started/completed均0，真实exit1。host另有拒绝标记已存在的FileExistsError，全部日志和结果保留，不回写成功。

218的13新检查只修正两件事：所有原损失导入之后再设CPU2/CUDA4并重验源完整运行时；完整输入/目标显式设备往返。其余原194损失、原Adam、EMA、事务和硬4501机制限制保持。原217文件不修改。

独立reader v1误从事件行读取exit_code而KeyError/exit1；该字段实际保存于sealed进程表。新的v2仅SHA绑定并替换两处字段访问；失败原reader和日志永久保留，不捏造退出码，不重跑模型或机制。

## 关键归档

| 归档 | SHA256 |
| --- | --- |
| 218实现 | 2652de693f6300e274feb102379e095f4d0a747755a38c2a2c4908d3a76eba4a |
| 218测试 | 09e9d509b9399c1ce25e138953a9e80eb443143da5a000e7ce583169d58fee1d |
| 218单位日志 | 49e82e3ee4eb0749aad5d7a759b2b99151efcc7f5d64190fc96065bb61cfc0ee |
| 218 CPU实际result | 35973a31bc88b6c591a93e9beb8a96a1d7a2e18a509e67366575df632e4a81fb |
| 218 CUDA实际result | 6cc0734a3b23d7b46182e7739d18cb7b4135ccdd029db6394f94e83d6404e8ad |
| CPU独立review | ef7d20c28d7a4da6410f7aada441916deccdc7559ba5fb80ee936f3cf758c7e2 |
| CUDA独立review | 52bc9186e893900a4a926bc124f0f4696e4a10c0cda55fd4d8c78df97362f320 |
| 新v2只读reader | d67ec15a1bbb7a3e0573c15f06f014d285d355bec3539773861677dc8b88499e |

CPU run：ae0ed3/session26693/817290，实际native0。CUDA run：fe1723/session24316/700e71，实际native0。CPU reader：7e6b8c/session56651/a19f20，实际native0。CUDA reader v2：2a1440/session54837/ad2eea，实际native0。前台调用，非WMI，无detached_exit。

## 下一项工程与明确未完成项

真实输入模型审计、严格原CUDA运行时/RNG、真实原Adam首步/EMA/磁盘重放/完整回滚机制现在已有真实证据。下一项是新正式单raw/单原Adam训练入口，把实际音频后端/逐draw原生守卫/原采样、完整原DEV政策、raw与EMA阶段评估/完整状态receipt接入，并完成其新单位和完整协议/持续授权记录/plan。不得再重复203..218关闭单位/旧实际draw/已成功机制或独立reader。

新正式trainer/CLI/协议/训练plan/worker尚未创建；不能把一输入诊断当500步正式训练或音质证据。剩余training-only实际依赖语义阅读仍未宣称全部完成，不提前封存缺工具的计划。原报告102单raw/原Adam500更新4501..5000、新4750和5000/硬5000/唯一EMA评估变量及所有旧stop和累计资格不变。当前人听和独立真实验收仍PENDING。

原20 tracked dirty/469 insertions/44 deletions保持，没有提交、推送、音乐上传、驱动/PATH/注册表/精度修改或终止用户程序。自动化实际观察为PAUSED；本轮不擅自更改。

