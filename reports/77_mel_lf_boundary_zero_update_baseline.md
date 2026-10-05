# 77 — LF32零更新起点直接效应与当前进度

2026-10-03 23:17（Asia/Shanghai）。188 summary3500前台session64953真实exit0(chunkf08f5a)，原31首177视图/逐曲聚合/原政策、完整model/Adam/参数顺序/模式/停止与LF身份独立复核通过。summary SHAa633df26cb4645f409e6d2a21adde96cb706e08b8808214042e7e481d5fbefe0，保存results/mel_lf_boundary_review_20261003/summary_step_3500。独立monitor baseline_review已记来源/真退出/对照；analyzed=[3500]、pending=[]。不重复summary3500/机制/审计/单元，不导出旧3500音频。

本3500没有训练更新：两个新臂model/Adam完全同源3500 full-control ARMS[0]，LF44控制评分与源逐项一致。候选32仅改变重建边界，因此下述是零更新直接效应，不是新训练收益。3750/4000才新增训练评分，旧开发不是盲测。

候选LF32对冻结44：按原探索资格政策eligible=true（控制false），平均残留投影减少1.47776dB、平均人声重建error-SNR高.59321dB；release_selection仍NONE。资格来自该旧DEV政策，人工听审/独立真实验收仍PENDING，不能宣称干净分离/发布/板端达标。

候选32对同权重LF44控制：平均残留少1.37747dB、平均重建高.23147dB，但不是全面胜出，原政策对此并行控制eligible=false。

| 域 | 残留投影变化dB（低好） | 人声error-SNR变化dB（高好） | 伴奏系数绝对误差增加 |
| --- | ---: | ---: | ---: |
| MUSDB native | -.27657 | -.04897 | .01297 |
| MUSDB weak | -.01685 | -.10258 | .00231 |
| MIR native | -4.28145 | +1.56138 | .06042 |
| MIR weak | -.93502 | -.48393 | .02359 |

改善均值明显由MIR native贡献，不说四域重建全好；MIR weak残留降低却重建回退。独立描述的低频伴奏error-SNR：MUSDB native低1.20896dB/weak低1.23822dB，MIR native高.84988dB/weak低2.71850dB；不是新资格阈值或波形低频质量上限。三器乐error-SNR对控制差均值-.04185dB，主要Maldito低.12555dB，另两首仅约3.6e-6/1.2e-7dB细微数值变化，不夸称均等损伤或据均值选模型。投影系数不是能量百分比。

当前真实worker41728已进入CUDA training，2026-10-03T15:17:27.2547522Z状态step3530/additional30/error=null，已记录33共同更新行；CPU累计1000.28125秒，stderr空、无detached_exit。前20成功共同更新平均9.530秒、范围4.712..19.222秒，共享GPU可波动，不承诺固定ETA；单进程WDDM显存N/A，不把整卡用量全部归本任务。

固定500更新/4000硬终止、原完整辅助与器乐权重不变，持续比较各自3500起点/冻结/并行控制，警惕伴奏和弱域重建代价。20分钟ACTIVE巡检继续，不机械下调LF或加lambda/器乐权重、不抬历史上限。全程NONRELEASE，无板端/部署/音乐上传/SD/Git提交授权，原历史绑定/负证据及其他未提交修改保持不变。
