# 固定 24 首配对试听包与本机听审入口

日期：2026-10-02（Asia/Shanghai）。承接[报告 25](25_paired_teacher_cpu_mechanics.md)。
本阶段让人工听审可操作，**不是正式训练、标签批准或板端模型更新**。

## 已完成

新增 `144_export_paired_listening.py`，读取原封存配对 bundle、两份 snapshot 和
pending 清单，不改变原标签/entry/协议或配对歌曲顺序。原曲完整解码，核对同一
FLOAT32 PCM 指纹后按清单的精确样本裁剪，不使用 MP3 时间快跳。
两教师的伴奏裁剪再次核对为输入逐样本减人声。

固定 24 首各导出原清单的 10%/50%/90% 三个约 18 s 位置，每处五路：

1. 原曲。
2. HTDemucs 人声、伴奏。
3. Mel-Band 人声、伴奏。

每首**所有三个窗口、所有五路共用一个非放大的安全播放增益**，保存 PCM24
立体声 44.1 kHz，峰值 <=0.95。源 FLOAT32 标签允许过满刻度，未削波或改写；
生成的是播放副本，不是训练标签。共同增益不等于严格感知响度匹配，也不是盲测。

每个副本保存后独立重载，对比期望波形与 PCM24 量化误差上限；包内 SHA256、
采样率/声道/subtype/长度/安全峰值再次复核通过。导出只保留一首完整混音及
三个窗口，使用 CPU，不运行任何教师或学生训练。

```text
试听文件/teacher_pairs_20261002/
  index.html / app.js / style.css
  manifest.json
  song_0006/w1/...   # w1～w3，每处五个 WAV
  ...               # 全部原固定 24 首

PAIRED_LISTENING VERIFIED songs=24 windows=72 audio=360;
review=pending; no training
manifest.json SHA256:
8e74e593c8e08270c76f2b72bc3988e596b2cf9a864110634873ab5bbaf0387a
```

共 360 个 WAV，整个目录 364 个文件、1,714,743,457 B（约 1.71 GB / 1.60 GiB）。
启动前预算 1,732,802,656 B，另保留 12 GiB；结束后 D 盘余量 31,808,405,504 B。
24 首的共同播放增益在 0.539622～0.835550 之间，不逐路放大弱人声。
音乐与本地包均被现有 Git ignore 规则排除，没有上传或提交。

## 页面与访问边界

新增 `145_serve_paired_listening.py`：先核验播放包，再仅监听
**127.0.0.1:62043**，给本次会话生成随机 URL 前缀。只允许清单中的三个 UI 文件
和 360 个播放文件；不开放目录浏览、原曲库、原 FLOAT 标签、manifest 或磁盘
其他文件，不监听局域网。支持 HEAD/单范围 Range 音频读取，拒绝其他路径及写入。
无上传端点、跨域放行、第三方脚本、遥测或浏览器权限请求。

页面可以搜索歌曲、切换三个位置、比较原曲与两套 stem；一条音频播放时暂停
其余播放器。computer-use 技能用于实际浏览器检查，验证了五路媒体加载、窗口
切换、播放进度推进，并留下未填写审核项的截图；没有由助手给出听审结论。

听审选项和备注仅保存在当前浏览器 localStorage，并可导出 JSON 草稿；它们
**不回写原 entry/snapshot，不代表正式签署的批准，也不会解锁训练入口**。
当前人工结论仍为 0/24，rights/listening pending，`training_authorized=false`。
换浏览器或端口前应导出备份。

## 验证证据

新增 12 项 CPU 测试通过：精确窗口/边界/拒绝补零、输出范围、封存身份与待审
状态、源/标签改变拒绝、残差一致性、磁盘预算、共同增益及量化重载、无 CUDA
查询/不写原始资料、HTML 数据转义、不覆盖既有包、文件白名单与 Range、只读
loopback 服务以及草稿不能自动授权。
原 pilot/播放 12 项、数据入口 16 项回归通过，合计本轮 **40 项**。
Python 语法与 `git diff --check` 通过。

导出后再次核对 96 个原始 FLOAT stem 文件、原曲文件和 60 个配对准备绑定文件，
全部哈希不变；冻结学生、旧正式协议/数据锁和新蒸馏协议均未变。

浏览器实测：五路均为 18 s、readyState=4、没有媒体错误；从前段切到中段后
窗口显示原曲 2:06～2:24。HTDemucs 伴奏播放进度到 7.41827 s，随后暂停；
其余四路没有同时播放，四个审核选择仍为 pending，草稿计数为 0/24。

```text
results/paired_listening_ui_20261002/browser.png
results/paired_listening_ui_20261002/browser_check.json
```

UI 导出按钮确实下载了 `C:/Users/30519/Downloads/paired_listening_draft.json`，
核对含 24 个原固定 ID、正确 bundle 指纹、0 条人工判断和训练授权 false。
这是流程检查用的空白草稿，不是人工审核结果。下载通知接口超时后没有重复点击，
改以实际下载文件核验，页面仍正常。

```text
空白导出草稿 SHA256:
ba5e236313fb62eac4579115e237cd19e7fc3db35b34ff925556fd465f588bc0
```

## 使用与下一步

已在本聊天旁保留试听页；本机服务运行期间可直接使用本次链接。
服务不是开机常驻程序或定时任务，应用关闭后可能需要重新启动。重开只读服务
会打印新链接，**不要重复生成试听包**：

```powershell
$taskPython = 'C:\Users\30519\.workbuddy\binaries\python\envs\stem-npu\Scripts\python.exe'
& $taskPython scripts/145_serve_paired_listening.py
# 端口已由本试听服务占用时直接使用现有链接，不再启动第二个实例。
# 只做独立包复核：
& $taskPython scripts/144_export_paired_listening.py --verify
```

先从前两首开始熟悉操作：原曲确认有没有人声，比较两套伴奏中的人声残留、
乐器误删、抽吸/断续，再用人声轨检查漏分/串音，在备注写清位置和问题。
随后按原名单完成逐首审核；少量试听的良好反馈不能自动批准全部 24 或 275 首。
页面的“使用资格已确认”只是一条待复核声明，需要独立依据。

原退出码 null 的完成证据选择、使用资格、独立真实开发/盲测门槛和正式训练
授权仍未解决；报告 25 的 CLOSED / NONE 状态不变。没有重训、GPU 推理、板端
镜像/SD 更新、Git 提交/推送或重新建立定时检查；`vitis_journal.py` 未触碰。
