# EMA影子状态组件：46项合成CPU门槛通过，完整训练准备未完成

2026-10-04，NONRELEASE。接续报告102，不改变其固定500共同更新设计，也没有启动GPU。报告101的新202完整诊断归档保持封存；旧197失败、199.verify失败、201.prepare失败全部保留，不重复单位/prepare/run/verify/真实前向。

## 本轮已实现及真实执行

新增 `scripts/203_ema_shadow_state.py`、`scripts/_test_ema_shadow_state.py`、`docs/ema_shadow_state_unit_scope_20261004.json`。203仅管理独立脱离梯度的张量，不是nn.Module、训练器、优化器或完整恢复容器。按原参数名字/顺序，以同设备FP32分别计算前影子乘.99、更新后原参数乘.01，再相加；不使用lerp、alpha合并、FP64平均或改变运算顺序。固定源4500/硬5000/计数、源哈希、名字/形状/dtype/trainability/模块与状态身份严格检查。所有缓冲区及不训练参数按对应原状态类型敏感复制；非持久缓冲区保存在影子恢复组件中，不凭空加到model.state_dict。此适配拒绝共享张量存储、共享模块、自定义extra_state或序列化钩子，不假装通用支持。

46项新合成CPU单位前台exec chunk `fefb27`，真实native exit0，完整消费输出并记录日志 `results/mel_ema_shadow_component_monitor_20261004/unit_tests_attempt01.log`。无持续session，无WMI启动，不能生成detached_exit。只有一次新单位运行，没有重复关闭的历史单位。包含独立NumPy逐标量FP32递推/合成500次影子计数与硬限、E0逐位身份（含有符号零）、缓冲区/冻结参数类型复制、无别名、导出/载入输入不变、原模型/全模式/已有grad不变、CPU/Python/NumPy RNG不变、fresh CPU未初始化CUDA、非法状态拒绝及不提交半个影子状态。

整个新单位执行中模型调用/autograd/Adam构造/CUDA初始化API被拦截；专门测试这些拦截确实会先拒绝调用，没有执行前向、梯度引擎、Adam或CUDA。真实音频输入0、真实学生加载0、真实学生前向0、训练更新0、DEV/试听0。合成fixture参数的手动改值和500次影子递推不是500次真实训练，也不是4501..5000正式暴露。

组件BytesIO序列化和合成raw+shadow回卷已测，但**不是磁盘完整容器位重放、完整Adam/RNG/采样/日程事务、跨设备数值恢复或真实下一步机制通过**。模块只检查caller传入raw_step；这个整数不能独立证明实际Adam更新。构造器检查固定源哈希声明，尚未实测完整真实源4500 ARMS0接入；不得把声明当文件/参数来源验证。EMA参数没有匹配训练Adam，影子组件不能单独resume。

## 封存与后续

unit_gate.json封存5个实际文件绑定及单位native0事实，content seal `8a06f8e056c9b48796050b7d09533f516e7e2cdfaaa6a5898d2e4ace2d325faa`。203/测试/组件scope/单位日志/单位门槛自本报告起保持不变，不再重复此46单位或修补封存203。未来完整新集成单位可调用该模块，但不能重复本套单位冒充新进展；若发现新接口缺陷，另立隔离恢复工具和证据，不回写。

工具SHA `7fb5fd1b3c5f66f3e9a2ebf0b020997c3454c8cf334e90de03282ebe78b939ae`；测试SHA `d5a9adafdfb2d324e346cc8d5941e11ec0e5b6b8da5eea2e212d41427aadfc7a`；scopeSHA `76695a9b6608f98ff98caee619d3e4e5cca6597918a855d6d7795a9ee856f057`；单位日志SHA `c98538954b27b279271a93e58bc051c0a9be1313429e5f029e882f5ae68513ca`。报告102实际SHA仍 `39256e70d7e8ce4675102711c3b4595b0b81f315410bc3602df5677c16f784c8`。

下一步继续报告102的完整单原始训练轨迹集成：先完整读取实际193/194/143/147/150及stream/损失/前端矩阵/输入锁cache等真正依赖和真实源字段，核查新文件/任务/产物，另立新importer/trainer/完整集成测试/协议/审批/目录。原始model/Adam参数顺序/全模式/已有grad/更新曝光/CPU-CUDARNG/采样游标/原LR日程/累计best-stale-anchor/旧4500及legacy stop均须完整保留，独立影子递推状态/计数加入新的容器和全事务回滚，不能套用原194双训练臂或把EMA塞到原Adam。不得预先封存缺训练器的计划。

完整新单位、真实零更新审计、最多3唯一4501..4503 CPU/CUDA原194首步位一致/源runtime-RNG/影子非零/无别名/各完整容器磁盘位重放/含影子计数与CUDARNG事务回滚/跨设备PCMmetadata门槛均为PENDING。全部真实通过且disk>=12GiB/freeVRAM>=2300MiB/no duplicate后才有资格按报告102 WindowsPS5.1/WMI独立启动一次。本轮没有完整训练协议/审批/封存训练plan/worker，不启动新GPU。

05:19:13.9105426Z动态CIM无仓库Python任务，D盘24.62750244140625GiB；源历史与用户未提交修改保留。组件进展不产生音质结论或eligible/rank，releaseNONE、人听与独立真实验收PENDING。20分钟同聊巡检保持ACTIVE，健康无可行动变化安静，不因本轮结束停用。
