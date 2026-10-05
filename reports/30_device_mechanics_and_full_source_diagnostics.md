# 下载空档期：设备训练核心与整段源音频诊断

日期：2026-10-02（Asia/Shanghai）。用户同意先补 GPU 机制，再诊断现有学生。
本轮只实现工具、运行合成数据 CPU 机制测试和旧 DEVELOPMENT 只读推理；
**没有正式蒸馏、模型选出、板端替换或 SD 写入**。

## 1. 设备训练核心：实现与实测必须分开

新增 `scripts/147_paired_device_mechanics.py`，独立于已经哈希绑定的 script143。
两臂 FP32 模型/Adam 可选择 CPU 或 CUDA，沿用原图、产品波形 + complex L1、
microbatch=1、共享 LR/预算；保存模型、Adam、参数名顺序、模式、Python/NumPy/
CPU-CUDA RNG、输入游标和共同调度。恢复拒绝更换代码、设备/数值设置、曝光步数、
缺失 Adam 矩、错误参数顺序或半完成配对。

- 两臂更新作为一个事务；第二臂失败时恢复两臂及游标、预算和 RNG。
- 恢复载入中途失败也回滚；若回滚本身失败，实例标记 poisoned，禁止继续。
- 使用严格确定性算法，关闭 TF32/cuDNN benchmark；CUDA 必须在初始化前设置
  workspace，并绑定 GPU/后端版本及数值开关。没有自动 CPU 回退或混合精度。
- CLI **只接受生成的合成波形**，没有音乐路径、教师标签入口；硬限三步，
  检查点标记 `NONRELEASE_SYNTHETIC_DEVICE_MECHANISM`。`--formal-train` 始终拒绝。
  此设备核心不是已经获准、接通真实标签和开发调度的完整正式训练器。
- 旧 script143、旧 CPU 检查点和配对快照不修改，不能通过新 purpose 绕过标签导入门槛。

在实际冻结 bott2 网络上，用六段生成波形完成 CPU 三步；从磁盘第 1 步新建实例
续跑第 2/3 步，输入、模型、Adam、RNG、模式、游标和预算完全相同。再模拟第二臂
异常，全状态回滚完全相同。两臂各执行了 3 次参考更新和 2 次恢复重放；异常测试中
仅第一臂另执行了 1 次随后回滚的更新，第二臂未提交。它们不增加正式训练曝光，
也不构成音质/收敛结论。

当前有效证据：

```text
results/paired_device_cpu_20261002_r2/device_mechanism.json
SHA256: 81ba309fd7a9ab855f6363f60000f9583ed4b274321556f1e6ffeb1f661b25b7
script147 SHA256:
9f3258439892060c8bab131d688ca6c1f41a015c30c7df83a5aee8f522b1c7b5
```

最初的 `paired_device_cpu_20261002` 保留为修订前机制证据，不能作为当前代码的
恢复绑定；r2 在参数名快照去别名和不可恢复错误锁定加固后独立生成，未覆盖旧目录。
三次参考 CPU 更新各约 1.36～1.60 s，不据此估算真实数据长训或 GPU 吞吐。
PyTorch 2.14 的 Adam 在 CPU step 中可能自行查询 accelerator availability；
本轮 CPU 测试验证的是没有 CUDA 初始化、CUDA RNG 使用/同步或 GPU 算子，
不夸称框架完全不读取 GPU 可用性。

### CUDA 真实恢复测试尚未运行

本机 GPU 有其他用户应用运行，未终止任何进程。只读 preflight 检查时：
**GPU 利用率 59%，空闲显存 1284 MiB**。入口要求利用率 <=15%、空闲 >=2300 MiB，
因此正常暂缓，未初始化 CUDA、创建 GPU optimizer 或执行 GPU 更新。
记录在 `results/paired_device_cuda_20261002/gpu_preflight.json`。
这不是 CUDA 算子失败，也不是教师/学生任务卡死；以后空闲时应使用新输出目录实测。
检查只是瞬时保护，不是 GPU 资源预留；仍需关注实际运行时的其他负载。

CPU 位一致不能证明 CUDA 位一致，也不承诺跨设备、PyTorch/驱动版本恢复一致。
数值设置与优化器恢复的实现依据：
[PyTorch 2.14 reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)、
[Optimizer.load_state_dict](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.load_state_dict.html)。

## 2. 完整已锁源音频：不是三首完整原曲的盲测

新增 `scripts/148_diagnose_full_source_audio.py`，先从三个旧 DEVELOPMENT 域各选
字母序第一条记录并封存计划，再解码评分，不按模型成绩选例子：

| 来源 | 本次完整解码时长 | 必须保留的限制 |
|---|---:|---|
| MUSDB — A Classic Education / NightOwl | 6.800 s | 现有文件只是原曲节选 |
| MIR-1K — abjones_2 | 82.932 s | 12 个原卡拉 OK 文件按旧规则拼接，不是自然连续录音 |
| mshoxx — Beneath_v1 | 196.886 s | 原锁定的合成立体声纯器乐负样本 |

共 286.618 s，评分排除最初 128 帧、结尾 512 样本和 MIR 11 个拼接处的
过渡区，共约 276.053 s；没有读取训练曲、新 Cambridge 验收音频或 MoisesDB。
三个样本不能估计整个电子音乐场景的效果。

同冻结权重、线性 legacy_log 前端及 reflect STFT，对照保留 128 帧输入历史的
连续参考、逐层缓存的 16 帧推理、每 16 帧重启；只改变上下文生命周期。
每源前 512 帧另与一次完整前向核对。输出统一使用原混音减估计人声。

### 较长序列再次证实上下文重启缺陷

| 来源 | 重启掩码 MAE | 重启边界跳变 / 连续参考 | 逐层缓存最大掩码误差 |
|---|---:|---:|---:|
| MUSDB 节选 | 0.19050 | 2.87 倍 | 2.38e-6 |
| MIR 拼接记录 | 0.16870 | 3.35 倍 | 2.44e-6 |
| 纯器乐 | 0.07893 | 4.30 倍 | 3.16e-6 |

逐层缓存与连续参考的最大波形差 <=2.84e-7。原每块重启改变了输出，并可能造成
周期调制；**边界跳变是掩码诊断，不是实际掉样数或已证实的扬声器卡顿比例**。
MIR 掩码跳变统计含原文件拼接，但两路径输入相同；波形真值评分已排除其过渡区。
本轮复核延长了既有状态保存数学方案的序列覆盖，不替代完整整数链路、RTL、DDR、
实时联合负载或实板模拟输出验收，也没有新建/替换候选 BOOT.BIN。

### 人声残留与低频保护是独立的学习取舍

下表为真实源轨 **error-SNR**（尺度相关参考/误差能量比，越高越好），不是试听分数：

| 只读诊断路径 | MUSDB 人声 | MIR 人声 | 纯器乐伴奏 |
|---|---:|---:|---:|
| 连续模型，当前 LF44 | -0.032 dB | 2.269 dB | 26.588 dB |
| 每 16 帧重启，LF44 | 0.885 dB | 2.268 dB | 22.983 dB |
| 连续模型，无低频保护 | -0.056 dB | 5.833 dB | 24.368 dB |
| 用答案构造的 513-bin 实掩码 | 8.812 dB | 11.267 dB | 不适用 |

1. 修复上下文不能自动使所有平均真值指标上升：MUSDB 这段重启的 SNR 反而更高，
   MIR 几乎不变；不能按一个平均指标否认已经复现的周期调制缺陷。
2. MIR 稳定区的人声 STFT 能量有 **41.77%** 落在 LF44 合成完全屏蔽的 bins，
   MUSDB 只有 **1.87%**。去掉保护在 MIR 上提升约 3.565 dB，却令纯器乐伴奏
   error-SNR 下降约 2.219 dB；MIR 投影伴奏保留系数也从 0.897 降到 0.803。
   不支持直接删除低频保护，也不能把“41.77%”推广到其他歌声。
3. 连续 LF44 的投影残留人声系数在 MUSDB/MIR 分别约 0.836/0.642，提示这两段
   仍有明显残留。它是源轨联合线性拟合诊断，不等于独立分离出的残留人声响度。
4. 全 513-bin 已知参考掩码利用了答案，不是学生已获得的分数或理论最优上限。
   同频带/LF44 的答案投影构造也保存：MUSDB 为 6.839 dB，MIR 为 2.088 dB，
   后者甚至低于当前模型。因此不能拿这种平均/插值构造当可达上限，或据此保证
   扩大模型必然改善。仍应先公平检验教师蒸馏，再单独评估布局/低频策略。

## 3. 留存、测试与下一步

保存 18 个 FLOAT32 WAV，共 606,714,192 B：每源原混音、参考人声/伴奏、连续/
缓存/重启三路伴奏。各路使用同一源增益，不独立响度归一化或削波；音频在忽略目录，
不是板上录音、不上传音乐，也没有改写原试听页。

```text
results/full_source_diagnostic_20261002/plan.json
SHA256: 6ed9734fa5f90477fa8f8f1c3004c6abf3c65cf8eb99b58b2f004697f870feac
results/full_source_diagnostic_20261002/diagnostic.json
SHA256: 3fc21f827d8f21b8477777c230df7df9eedb326cb119b2e1fd12751fd7b0aded
script148 SHA256:
0925eeda3217ac6926d328fa16e638c9ee74051b6e3f0d6fff23cf598f531b69
```

新增设备核心 16、整段诊断 9 项测试通过；原配对开发 14、配对机制 15、数据入口
16、模型选择 13 项通过，共 **83 项独立测试**。有效 CPU 机制、整段诊断和既有
script146 评估的独立 `--verify` 均通过，不重复生成原评估/教师标签。

下一步：GPU 空闲时先完成合成数据 CUDA 真实恢复与异常回滚实测；继续补批准导入
与真实标签/开发验证调度接线，但不开放正式训练。新数据到手后按报告 29 查重和
封存独立角色。当前 24 首听审与私人 MP3 用途声明沿用报告 27，不要求重复试听；
旧 HTDemucs 数值退出码替代证据选择、独立验收和正式训练授权仍未自动解决。
初次教师比较继续固定图/布局/LF44/损失，不同时把低频诊断改成新的部署配置。

```powershell
$taskPython = 'C:\Users\30519\.workbuddy\binaries\python\envs\stem-npu\Scripts\python.exe'
& $taskPython scripts/_test_paired_device.py
& $taskPython scripts/_test_full_source_audio.py
& $taskPython scripts/147_paired_device_mechanics.py --verify
& $taskPython scripts/148_diagnose_full_source_audio.py --verify
# GPU 空闲后，使用新目录，不能覆盖已有 deferred 回执：
# & $taskPython scripts/147_paired_device_mechanics.py --synthetic-smoke --device cuda --out results/paired_device_cuda_fresh
```

冻结学生、原协议/数据锁/教师分轨/快照、旧 CPU 机制与开发评分、板端镜像及
`vitis_journal.py` 保留；不影响正在下载的数据。没有 Git 提交/推送、音频上传、
SD 写入、后台正式训练、定时任务、新聊天或子代理。
