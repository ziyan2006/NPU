# 独立真实分轨：下载入口与人工申请交接

日期：2026-10-02（Asia/Shanghai）。承接[报告 28](28_paired_development_adapter_and_truth_gap.md)。
本轮继续查找新的独立真实分轨来源，未生成新标签、修改模型或启动训练。

## 已核实的入口

1. [Cambridge 最新官方目录](https://www.cambridge-mt.com/ms3/mtk/)在搜索结果中可见，
   但本次正常 HTTP 访问返回 Cloudflare 403 / `Just a moment...`，要求 JavaScript/cookies。
   未取得目录中的实际下载链接；没有把旧目录的 TLS 握手故障归因于模型，也没有关闭
   证书验证、改安全设置或绕过访问检查。报告 28 的三个资料候选仍未下载。
2. [MoisesDB 官方仓库](https://github.com/moises-ai/moises-db)指向
   [Music AI 研究页面](https://music.ai/research/)。该页面介绍 240 首、47 位艺人、
   12 类风格及非商业研究用途；不据此推定有足够的合格电子人声验收曲目。
3. 使用 computer-use 技能在正常浏览器里检查上述研究页面。拒绝可选 cookies 后，
   点击 **MoisesDB** 卡片的 Download，实际打开下载申请表，而非直接下载文件。
   已观察到字段 **Full name、E-mail、Organization、Justification**，以及
   **Request download** 按钮和论文引用提示。
   其余两个 SDXDB23 的 Download 不作 MoisesDB 入口使用。

表单未填写、未提交，用户姓名/邮箱/单位未传送。没有通过隐藏接口跳过申请。
因此本轮新增下载曲目为 **0**，真实验收候选仍为 **1**，已审核合格为 **0**；
24 首教师伪标签不计入真实验收曲目。原 Cambridge 六首的角色不变。

## 给用户的最小操作

可在已经打开的官方表单中，填写自己的姓名、邮箱、真实单位/身份与研究用途，
核对网站的用途要求后自行提交申请。不要填写不存在的组织，也不要把此前私人
MP3 的训练许可声明扩展为其他数据集的使用资格。

若符合实际项目用途，可参考以下 Justification 草稿并自行修订：

> I am researching lightweight vocal/accompaniment separation for an FPGA-based NPU.
> I would like to use MoisesDB for non-commercial research, including independent
> evaluation with real multitrack recordings. I will cite the MoisesDB publication.

本草稿未发送给网站，也不是代用户签署许可。取得官方下载链接或本地数据包后，
告知助手实际路径/链接；不需要上传音乐到外部服务。

## 拿到数据后的顺序

- 先检查实际包大小、磁盘余量、适用用途要求及提供方公布的文件校验值，
  再决定完整包还是官方支持的子集；当前未取得包大小，不启动整包下载。
- 先读取曲目/风格/艺人/源轨元数据，查重旧训练、开发和验收名单。
  在新音频模型评分前封存角色；不为了凑数把已有训练候选改为验收。
- 按真实文件显式映射 vocal 和 accompaniment，核对共同起点、采样规格、
  活跃人声、串音/SFX及可用时长，再整理新的听查项。
- 独立验收至少 10 首电子人声歌曲的门槛不变。当前不能承诺 MoisesDB 能补齐
  全部缺口，也不据此选择/发布学生检查点。

## 留存与保留项

官方申请页截图：
`results/paired_development_eval_20261002/moisesdb_request_download_20261002.jpg`

```text
SHA256: a14809693a9498cc46e834b3431b522345b7ee290ad0feaf3d1d18035025cc7e
D: 剩余字节（本轮检查）: 31751024640
```

轻量检查显示原 Mel-Band 275 首和 HTDemucs 24 首状态均为 complete/error=null；
相应旧 worker PID 不在当前 Python 进程列表，现有试听服务器正常保留。
不重新扫描已核验音频，也不从状态文件推导 HTDemucs 原退出码；报告 24/25
记录的原退出码未知问题仍未解决。

冻结学生、封存协议、原数据锁/伪标签/配对包、已有开发评估和板端镜像未改写。
未启动教师/学生推理、正式训练或定时任务；未写 SD、上传音频、提交/推送 Git、
创建新聊天或子代理；其他工作区改动及 `vitis_journal.py` 保留。

本阶段卡点是数据访问需要用户提供真实身份并申请，不是训练进程卡死。
