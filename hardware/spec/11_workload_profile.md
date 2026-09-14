# NPU v1 工作负载与算子画像

文档版本：`0.1-draft`

状态：网络 A 已生成；网络 B 待批准

## 1. 为什么先做工作负载画像

通用 NPU 不能只看总 MAC。架构是否合适取决于逐层通道、Tensor 生命周期、尾通道利用率、算子组合和 DDR 流量。本文件把模型事实与硬件提案分开：模型变化后重新生成画像，再判断现有 ISA 是否仍适用。

## 2. 网络 A：当前人声消除模型

数据来源：`hardware/generated/bott2_mir1k_v1/manifest.json`，checkpoint step 2855。

### 总体

| 项目 | 数值 |
|---|---:|
| 输入 | `C×F×T = 2×128×16` |
| 输出 | `4×128×16` |
| 参数 | 824,900 |
| 卷积层 | 11 |
| MAC/块 | 123,338,752 |
| 音频秒均 MAC | 1.3279 GMAC/s |
| 权重/bias/scale blob | 824,000 / 3,600 / 7,200 B |
| 64-lane 排程周期 | 1,986,560 cycle |
| 时间/块 | 92.88 ms |

### 逐层画像

shape 沿用 manifest 的 `C×F×T` 表达。周期只计 8×8 MAC 阵列排程，不含 DMA 和后处理。

| 层 | 输入 → 输出 | K/S | post-op | 参数 | MAC | 阵列周期 | lane 利用率 |
|---|---|---|---|---:|---:|---:|---:|
| enc0 | `2×128×16 → 32×128×16` | 3×3 / 1 | LeakyReLU | 608 | 1,179,648 | 73,728 | 25% |
| enc1 | `32×128×16 → 64×64×8` | 3×3 / 2 | LeakyReLU | 18,496 | 9,437,184 | 147,456 | 100% |
| enc2 | `64×64×8 → 96×32×4` | 3×3 / 2 | LeakyReLU | 55,392 | 7,077,888 | 110,592 | 100% |
| enc3 | `96×32×4 → 128×16×2` | 3×3 / 2 | LeakyReLU | 110,720 | 3,538,944 | 55,296 | 100% |
| bott | `128×16×2 → 128×16×2` | 1×3 / 1 | LeakyReLU | 49,280 | 1,572,864 | 24,576 | 100% |
| bott-res0 | `128×16×2 → 128×16×2` | 3×3 / 1 | LeakyReLU+ADD | 147,584 | 4,718,592 | 73,728 | 100% |
| bott-res1 | `128×16×2 → 128×16×2` | 3×3 / 1 | LeakyReLU+ADD | 147,584 | 4,718,592 | 73,728 | 100% |
| dec3 | `224×32×4 → 96×32×4` | 3×3 / 1 | LeakyReLU | 193,632 | 24,772,608 | 387,072 | 100% |
| dec2 | `160×64×8 → 64×64×8` | 3×3 / 1 | LeakyReLU | 92,224 | 47,185,920 | 737,280 | 100% |
| dec1 | `96×128×16 → 32×128×16` | 1×3 / 1 | LeakyReLU | 9,248 | 18,874,368 | 294,912 | 100% |
| out | `32×128×16 → 4×128×16` | 1×1 / 1 | tanh | 132 | 262,144 | 8,192 | 50% |

### 算子覆盖

网络 A 对 v1 的硬需求是：

- 普通卷积：`1×1`、`1×3`、`3×3`；
- stride：1 和 2；
- 非对称因果 padding；
- LeakyReLU 和 tanh；
- 两个 residual ADD；
- 三次 nearest-neighbor 2× upsample；
- 三次 encoder skip concat；
- per-output-channel weight/requant；
- INT12 激活和 INT32 partial sum。

它不验证 depthwise、pool、通道非 8 倍数的大量尾块，也不验证与音频无关的 H/W 范围。因此仅跑通网络 A 不能证明 NPU 通用。

## 3. 网络 A 的存储压力

按 16-bit activation 槽计算，不含 bank padding：

| Tensor | 连续容量 |
|---|---:|
| 输入 | 8 KiB |
| enc0 skip / dec1 output | 各 128 KiB |
| enc1 skip / dec2 output | 各 64 KiB |
| enc2 skip / dec3 output | 各 24 KiB |
| enc3/bott | 各 8 KiB |
| 最终 4-channel INT16 输出 | 16 KiB |
| dec3 物化 concat | 56 KiB |
| dec2 物化 concat | 160 KiB |
| dec1 物化 concat | **384 KiB** |

最后一个 concat 单独就超过 216 KiB activation-bank 初始分配，因此 v1 必须采用 segmented view/双源读取，或在 DDR 中物化；不能默认把整个 concat 放片上。encoder skip 总计 216 KiB，刚好占满提案中的 activation banks，未给 ping-pong 和输出留空间，所以至少部分 skip 需要 DDR spill 或更细粒度 tile。

## 4. 网络 B：通用性验证候选

推荐使用一个小型 MobileNet-like conformance CNN，不以分类准确率为目标，专门覆盖网络 A 缺少的硬件路径：

```text
input  [1,32,32,3]
conv3x3 s2, 3→16 + ReLU
depthwise3x3, 16→16 + ReLU
pointwise1x1, 16→24 + ReLU
depthwise3x3, 24→24 + ReLU
pointwise1x1, 24→24
residual ADD
average pool 2x2
pointwise1x1, 24→10
output [1,8,8,10]
```

它验证：输入 3 通道、输出 10 通道的尾 lane，depthwise，1×1，普通卷积，残差和 average pool；所有 shape 静态、batch=1，仍符合 v1 边界。

在批准网络 B 前，depthwise 和 pool 的 P0/P1 等级不能最终冻结。如果希望 v1 只服务音频 U-Net，可把 pool 保留 P1，并选择另一个不含 pool 的音频 CNN；代价是“通用性”范围会更窄。

## 5. 当前算力判断

64 lane 在 200 MHz 的理论峰值为 12.8 GMAC/s，在 100 MHz 为 6.4 GMAC/s。网络 A 平均只需要约 1.33 GMAC/s，但实时性由单块延迟、DDR stall 和片上 bank 冲突决定，不能简单用平均 GMAC 相除。

纯 MAC 排程为 1,986,560 cycle，完整任务预算为 4,000,000 cycle：留出的约 2.0 倍空间用于命令、DMA 未隐藏部分、post-op、bank stall 和流水线边界。P2 周期模型要证明这部分余量足够。

## 6. 模型准入报告

以后每个待部署模型必须由编译工具自动产生以下报告：

- 不支持的算子、dtype、shape 和动态行为；
- 每层逻辑/物理 shape、padding、布局和尾 lane；
- 每层 MAC、预计阵列周期和利用率；
- Tensor 生命周期、峰值片上空间和 DDR spill；
- 权重、激活、命令和描述符总字节数；
- 估计 DDR 读写流量与不可隐藏 stall；
- accumulator bound 与 requant 参数范围；
- 总周期、目标频率下毫秒数和 deadline 余量；
- capability/ISA 最低版本。

任何 P0 边界之外的模型必须由工具明确拒绝，而不是依赖上板后才发现错误。
