# 2026-10-05：外部视觉教师对照与离线答案蒸馏

2026-10-21 最新：官方500题于02:33:21 UTC完成，03:43:03 UTC核验归档。
Qwen-Drive教师69.80%、保留CE55.40%、parent54.20%；教师相对CE修复128／退化56，
净+14.40个百分点，视频bootstrap 95%区间[+9.60,+19.00]。这是教师筛选结果，
尚无学生蒸馏收益。详见[官方结果、两套评测对照与结论](#qwen-drive-official-results)。

此前内部600题仍为教师56.83%、保留CE59.00%，净−2.17个百分点；不能用官方结果覆盖这一差异。
[600题历史结果](#qwen-drive-results)、[协议审核](#qwen-drive-protocol-audit)和
[Scenery全46题五帧仲裁](#qwen-drive-scenery-adjudication)均保留。
已实现冻结教师、从保留CE开始的4k训练候选／500步离线答案蒸馏与匹配CE入口，
详见[实施说明与后台命令](qwen_drive_distill_20261021.md)。CPU检查通过，GPU短测因显存争用
中断而未验证，正式入口先自动完成小规模GPU短测，成功后才生成4000候选；正式训练未启动。
原方案见[下一步试验安排](#qwen-drive-transfer-pilot-plan)。
安装及模型下载见[教师接入](#qwen-drive-20261020)，并行命令见[四卡入口](#qwen-drive-four-gpu)。
以下按日期保留历史状态；旧教师、旧训练起点及“尚未运行”不能当作最新Qwen-Drive状态。

## 2026-10-05 历史状态

代码已实现；600条教师对照于2026-10-05 10:33 UTC完成，学生58.83%、教师42.33%，
修复82/退化181，未通过原总体优势门槛。10k训练候选推理和两组正式训练尚未运行。
后续使用[任务说明与原生视频对照](teacher_adaptation_20261005.md)检查教师任务适配。
本文末尾单条验证记录描述代码首次交付时状态，正式教师结果见时间线第37节。

学生初始化为`outputs/mpp_lingoqa_qformer_all_1epoch_v1`，同时读取语言LoRA、Projector、
完整Q-Former。该模型开发4320题60.05%、官方500题54.2%。续训检查点24000的对应结果
为61.78%/52.2%；本轮使用54.2%模型作为既定参考。官方集合已在历史实验中多次观察，
不将它描述为从未使用的盲测集。

默认候选教师是本地`chart_verification_vlm/checkpoints/Qwen3-VL-4B-Instruct`，使用现有
`chartvlm` Python3.11/Transformers4.57.6环境。只读复用权重与环境，不修改该项目。
4B是参数规模，不构成教师强于MPP的证据；必须通过开发对照。可用`TEACHER_MODEL`
和`TEACHER_PYTHON`切换其他已下载、兼容的HF教师，切换时使用新的`--output-dir`。

## 方法与数据

外部教师读取同一视频五帧完整图像和原问题，按时间顺序组成多图输入；没有参考答案提示、
中央裁剪或人工选择区域。每帧教师处理预算为50176至1003520像素；学生保留已有224输入
的EVA缓存。教师和学生的视觉处理分辨率不同，报告中有记录；教师能看到的细节不一定都能
通过蒸馏传给学生，需要用训练对照检验。

来源为`datasets/processed/lingoqa_qformer_all_v1`。新样本在
`outputs/teacher_distillation_v1/data`，不覆盖原始数据：

| 用途 | 问答 | 视频 | Action | Scenery |
|---|---:|---:|---:|---:|
| 教师开发对照 | 600 | 400 | 433 | 167 |
| 训练候选 | 10000 | 9858 | 6450 | 3550 |

seed44；按来源比例、不同视频轮转抽样，每视频最多2问，不根据已有正确/错误选开发题。
检查train/dev/eval的question_id、segment_id、frame路径隔离。完整开发集仍为4320题，
官方仍为500题。学生的600条开发预测从已核验的父模型结果中复用；epoch-01/final的
pt封装SHA不同但实际张量相同，沿用先前逐张量核验。

教师对照报告给出Judge、Token F1、fixed/regressed、来源分组和视频级bootstrap。
教师可用答案相对学生的95%区间下界大于0才允许进入训练准备，未正常结束的教师答案
在这个判定中按错误处理。原始Judge结果也单独保留。该门槛是实验规则，不能代替
视觉审查；`teacher_review/review.html`包含五帧、参考答案和双方回答，可抽查分歧。

训练候选在教师、学生都无参考提示的条件下生成，然后由相同官方Judge评分。
筛选教师正确且完整、学生错误的答案作为纠错目标；从学生原本正确的题中抽取约30%
原始答案回放，包括教师答错的题。最终有效样本数由实测决定，不宣称10k条全部进入训练。
`answers`始终保留原始GT；`teacher_answer`、`target_source`单独保存训练目标。

蒸馏组使用这个固定混合数据集，每个样本只做一次标准答案token交叉熵；约70%教师目标、
30%原始答案回放，相当于按实际比例混合两个CE期望。对照组使用完全相同的样本ID、顺序、
训练步数与初始化，全部读取原始`answers[0]`。训练目标来源是两组的区别。

默认3epoch、4卡、每卡batch1、累积4；语言LoRA/Projector LR2.5e-6，完整Q-Former
LR5e-6，FP32可训练参数、梯度检查点、seed44。每200步保存可恢复检查点。若最终N条，
默认更新数为`3 * ceil(ceil(N/4)/4)`。本轮预先固定比较两组final，开发选择baseline、
control或distill，平分保留较早者，然后仅评选中模型的官方集合。

这是离线答案蒸馏。教师字符串由学生tokenizer重新编码，无需词表对齐；旧`train_opd.py`
仍服务于同tokenizer的在线token分布蒸馏，没有把该流程称为OPD或RL。

## 完整命令

先在一张空闲GPU上运行教师对照。模型均已在本地，无需下载新数据或权重：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
CUDA_VISIBLE_DEVICES=0 bash scripts/run_teacher_distillation.sh teacher_compare
```

需要后台运行时，可使用：

```bash
systemd-run --user --unit=mpp-teacher-compare-v1 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=0 \
  /usr/bin/bash scripts/run_teacher_distillation.sh teacher_compare

journalctl --user -fu mpp-teacher-compare-v1.service -o cat
```

任意目录监视；状态以文件为依据，不冒充进程存活验证：

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_teacher_distillation.sh status'
```

对照完成查看`outputs/teacher_distillation_v1/teacher_report.json`中的
`teacher_qualified`与`usable_teacher_paired`。如果为false，训练准备会解释原因并退出，
此时应阅读纠错/退化案例或换教师，不会自动启动两组训练。

教师达标后的训练候选推理和筛选：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
CUDA_VISIBLE_DEVICES=0 bash scripts/run_teacher_distillation.sh prepare
```

该步骤要生成学生和教师各10000个回答并评分，比开发教师对照耗时；支持生成断点续跑。
教师、学生、Judge顺序运行，进程退出释放显存。观察最终`data/train_manifest.json`中
`teacher_targets`、`reference_replay`和实际来源分布。

两组训练顺序执行，选取四张空闲卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_teacher_distillation.sh train_control
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_teacher_distillation.sh train_distill
```

两组完成后：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_teacher_distillation.sh eval_dev
CUDA_VISIBLE_DEVICES=0 bash scripts/run_teacher_distillation.sh eval_selected
```

完整开发比较在`report.{json,md}`，官方比较在`official_report.{json,md}`。比较
`baseline_vs_distill`以及`control_vs_distill`，同时看净修复和区间；只有前者提高不能
单独证明教师监督优于同预算SFT。

断点恢复示例（换成实际存在的检查点）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_teacher_distillation.sh train_distill \
  --resume-from outputs/teacher_distillation_v1/train_distill/checkpoint-000200
```

`--output-dir`、`--epochs`等非默认参数在各相关阶段保持一致。新teacher模型与新像素预算
使用新实验目录。`--dry-run`仅打印GPU命令，数据准备/身份校验仍执行，状态显示
`dry_run_completed`，不能当作正式评测完成。

## 实现、核验与本次发现

- `scripts/generate_vlm_teacher.py`：独立HF教师推理、实际像素限制、EOS完成标记、逐条续跑。
- `mpp_video_qa/distillation.py`：原始划分隔离、预测对齐、教师纠错与原答案回放选择。
- `scripts/teacher_distillation.py`与Shell入口：阶段衔接、教师判定、同预算训练、开发选模、报告与监视。
- `mpp_video_qa/data.py`：默认原答案行为保持一致，增加显式distillation目标模式。
- `scripts/train_sft.py`：目标模式透传、保存与恢复检查；开发loss始终使用原始参考答案。
- `tests/test_teacher_distillation.py`：数据/答案隔离、截断过滤、真实缓存读取、流水线衔接、
  两组命令匹配、预算不同拒绝比较、官方结果不反向选模等测试。

本机Processor测试发现Transformers4.57.6的AutoProcessor未把顶层min_pixels/max_pixels
转给Qwen3图像处理器，仍保留16777216像素默认上限。已显式设置实际image_processor的size；
1920×1080五帧测试得到每帧46×82个16像素patch，即965632像素，低于1003520。

首次单条生成测试在沙箱内不可访问CUDA、自动回退CPU。已停止该验证进程，增加CUDA可用性
检查，在获准GPU环境重跑通过；不是正式训练失败。GPU验证产物：
`outputs/teacher_distillation_backend_smoke_v1`，1/1，16个completion tokens，finished=true。
回答是否正确没有在该单条检查中得出结论。

单条GPU验证、真实数据准备、父模型开发答案身份核验通过。新增10项CPU测试通过；
此前已运行的15项完整Q-Former/全量/续训回归测试通过。没有运行两组正式GPU训练，
没有新增教师对照准确率或蒸馏收益。

<a id="qwen-drive-20261020"></a>

## 2026-10-20：Qwen-Drive 教师筛选接入

本次先实现外部驾驶教师的能力筛选。输入仍为同一视频的完整五帧、原问题，不输入参考答案、
车辆状态或导航信息。没有启动训练，也没有将不同词表教师直接接入现有 token KL/JSD。
Qwen-Drive 的词表与当前旧 Qwen7B 学生不同，后续若采用答案蒸馏，需要把教师文本重新编码；
是否值得训练要先看筛选和事实审核结果。

官方依据：[仓库](https://github.com/QwenLM/Qwen-Drive-1.0)、
[VQA 接口](https://github.com/QwenLM/Qwen-Drive-1.0/blob/28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0/src/qwen_drive/modeling_qwen_drive.py)、
[模型文件](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B/tree/28484089a7cc8c335cf5089fb0745cf7c49b6eaa)。
固定源码 `28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0`，模型修订
`28484089a7cc8c335cf5089fb0745cf7c49b6eaa`。已取少量官方源码核对接口，权重未下载。
只下载根目录 VLM 文件，不下载 planner / perception 权重。

实现使用官方 `QwenDriveProcessor.encode_vqa`、`QwenDriveForPlanning` 的 VLM，
显式记录 completion token、EOS、实际图像网格与显存峰值。加载后移除未使用的 planner，
单卡 BF16 / SDPA，五帧每帧最大 921600 像素，默认最大生成512 token，greedy。
提示补充了时间顺序与自车视角，因此**这不是官方79.4分协议的逐项复现**。
保留 raw answer；闭合的 thinking 块不送入 Judge，未闭合 thinking、无 EOS 或空答案不能
作为完整教师纠错。长答案可能触及 Lingo-Judge 128 token 上限，查看 raw/completion 记录。
不静默减少帧数、截短输入、回退CPU或跳过失败样本。

准备的数据与对照：

| split | QA | 视频 | 用途 |
|---|---:|---:|---|
| smoke | 2 | 2 | Action/Scenery各1题，检查接口、完整输出与显存 |
| dev | 600 | 400 | seed44，同历史教师筛选ID；Action433 / Scenery167 |
| official | 500 | 100 | 仅显式执行 `official` 时运行 |

dev 同时复用两套已评分基线：最新 mechanism 的 SFT parent，以及历史保留的40k CE final。
逐题核验原题、参考答案、来源、ID、预测和官方评分协议；保存原始模型报告、预测与评分文件
SHA。没有重新生成学生答案或重新读取所有权重做张量审计。official 的 parent 复用历史
`feedback_opd_eval_v2/baseline_eval`，CE复用 `sft_control_final_eval`。

**数据来源限制：**内部 dev4320/600 来自公开 LingoQA train 的自划分，外部驾驶教师可能
已见过这些视频；当前没有它的逐视频训练清单。official500也已在本项目多次观察。
报告分开记录“在这批题上观测到优势”和“已验证可用于蒸馏的教师”：
前者由相对 retained CE 的视频级 paired bootstrap 决定；后者当前为未建立，不会仅因内部
dev分数更高自动启动长训。还需要核对教师训练来源、独立未见视频及关键事实。

### 用户执行的安装、下载与长评测

独立环境路径为本项目 `.venvs/qwen-drive`；不会升级原 `mpp_llava` 或 `chartvlm`。
环境创建和模型下载可能较长，本次交给用户执行：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
bash scripts/run_qwen_drive_teacher.sh setup
bash scripts/run_qwen_drive_teacher.sh download
```

setup 使用官方固定核心版本 Torch2.8 / torchvision0.23 / Transformers5.14.1。
VQA 使用 SDPA 和 FLA，causal-conv 走 Torch fallback，未要求编译 flash-attn / perception
扩展。新环境安装和整模型加载尚未在本次执行，最终以 GPU smoke 的实际结果为准。
下载支持 Hugging Face 缓存续传；过程日志在 `outputs/qwen_drive_teacher_20261020/logs/download.log`。
环境安装日志直接输出到当前终端。

选一张空闲GPU；以下 `0` 改成实际卡号。可先做不加载整模型的环境、配置及 tokenizer 检查：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_qwen_drive_teacher.sh preflight
```

后台执行正式教师筛选，自动先做2题 smoke，通过后600题生成、Judge评分和报告串行执行，
进程退出后释放显存，再进入下一阶段：

```bash
systemd-run --user --unit=mpp-qwen-drive-screen-20261020 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=0 \
  /usr/bin/bash scripts/run_qwen_drive_teacher.sh screen
```

监视进度和进程：

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_qwen_drive_teacher.sh status'
journalctl --user -fu mpp-qwen-drive-screen-20261020.service -o cat
```

需要看具体模型加载/生成错误：

```bash
tail -n 80 outputs/qwen_drive_teacher_20261020/logs/smoke_teacher.log
tail -n 80 outputs/qwen_drive_teacher_20261020/logs/dev_teacher.log
```

文件状态的 `completed/action=prepare` 只表示数据准备完成。`screen` 不运行 official，因此
`official report validated=False` 在此阶段正常。判断是否跑完看 dev 报告验证状态及服务日志。
意外中断后用原参数重跑 `screen`，跳过已保存答案；对被截断的 JSONL 末行会报错，
不自动删除已有记录。修改模型、代码、像素预算或生成预算必须换 `--output-dir`，防止混用结果。

结果文件：

- `outputs/qwen_drive_teacher_20261020/dev_report.md` / `.json`：两套基线、Judge、F1、
  teacher修复/退化、按视频bootstrap区间及Action/Scenery分项。
- `dev/review/review.html`：最多32例分层抽取的双对双错、纠错与退化案例，包含五帧与三方答案。
- `dev/review/audit_template.jsonl`：动作、原因/物体、方向、时序变化等人工审核空表；
  尚未填写的字段不能当成已审核。
- `smoke/teacher/generation_summary.json`：2题完整输出与GPU峰值，不能作为准确率结论。

需要另看已复用官方500时，独立执行：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_qwen_drive_teacher.sh official
```

该命令仍不训练；`official_report.md`记录结果，并保留“已重复使用、外部训练重合未核实”的说明。
本入口没有 `train` 动作，避免在教师筛选结果未知时消耗长训资源。

### 本次已执行的轻量核验

13项新增CPU测试通过，覆盖：无参考提示、五帧顺序、EOS/thinking/截断处理、续跑ID与元数据、
旧评分拒绝、基线产物改动拒绝、截断纠错过滤、事实审核队列，以及高分不自动认定未见泛化。
Shell语法和Python编译检查通过。官方图像处理器在1920×1080五帧合成输入的CPU检查通过：
5个视觉slot、每帧grid `[1,44,80]`、总4400个视觉token，像素预算生效。
这个检查使用测试tokenizer与配置，不等于真实checkpoint的GPU端到端验证。

600/500/2题文件及学生基线复制已准备并核验；真实教师环境、权重下载、GPU smoke、
600题教师评测和任何蒸馏训练均未执行。没有新增准确率或收益结论。

<a id="qwen-drive-four-gpu"></a>

## 2026-10-20：GPU 1、2、3、5 四卡并行

用户确认这四张卡空闲，并要求改为四卡。采用按题目分片的独立副本推理：每卡加载一份
Qwen-Drive，600题按原始行号模4分配，每卡150题；每张卡仍看到自己题目的完整五帧。
教师生成全部结束后，四个独立Lingo-Judge进程在相同四卡上各评分150题。
最后按原始题目顺序合并，使用原来的两套学生基线生成配对报告。
没有训练或模型参数同步，不使用tensor parallel / DDP。

每张卡先检查1题Action及1题Scenery，共8题；全部完成且输出完整后才执行600题。
本轮沿用已下载的同一份模型权重、独立教师环境和固定模型版本，无需重下。
默认输出目录改为 `outputs/qwen_drive_teacher_4gpu_20261020`，与原单卡目录分开，
原目录中的下载任务及历史数据不覆盖。

`--gpus`接受物理卡编号，控制器为每个子进程单独设置`CUDA_VISIBLE_DEVICES`；
若不传该参数，则读取当前`QWEN_DRIVE_GPUS`或`CUDA_VISIBLE_DEVICES`，未设置时使用0。
实验身份固定分片数和题目划分；同样分片数可换物理GPU续跑，运行日志记录实际绑定。
改变分片数需换输出目录。若显式指定`--output-dir`，之后各阶段须使用同一路径。

当前下载结束后，用以下新命令启动长评测；本次未代用户启动：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
systemd-run --user --unit=mpp-qwen-drive-screen-4gpu-20261020 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  /usr/bin/bash scripts/run_qwen_drive_teacher.sh screen --gpus 1,2,3,5
```

监视总进度及每卡进度：

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_qwen_drive_teacher.sh status --gpus 1,2,3,5'
journalctl --user -fu mpp-qwen-drive-screen-4gpu-20261020.service -o cat
```

单卡日志示例：`outputs/qwen_drive_teacher_4gpu_20261020/logs/dev_teacher_shard_000.log`，
对应GPU1；001/002/003对应GPU2/3/5。评分日志为`dev_judge_shard_000.log`等。
如果任何一个worker报错，控制器停止同阶段仍运行的其他worker，已落盘答案保留；
修复后用同参数重跑，逐分片续跑，完成的Judge分片也可复用。
生成中断后发生损坏末行时仍明确报错，不静默丢弃记录。

合并前检查：四份模型/运行环境/输入/生成预算一致；每片ID准确且完整；没有重复或越界题目；
生成摘要与预测SHA一致；评分协议、原题、答案、参考及完成标记匹配。合并产物保留各分片
manifest和预测/评分SHA，重复合并不追加重复行。总生成ETA使用最慢未完成分片的估计，
该ETA不包含尚未开始的Judge及CPU报告阶段；进程存活仍以服务日志为准。

结果在`outputs/qwen_drive_teacher_4gpu_20261020/dev_report.md`及`.json`。
需要单独运行已复用的official500时，可执行
`bash scripts/run_qwen_drive_teacher.sh official --gpus 1,2,3,5`，每卡125题。
`screen`仍不自动运行official或训练，数据重合限制沿用前述协议。

速度尚未实测。生成阶段理想吞吐接近单卡4倍，但模型加载、磁盘读取、首轮内核准备、
最慢分片及报告开销都会影响总耗时。交付时仅作约2.5–3.5倍的粗略估计，不作为性能结论；
例如单卡3小时，粗估四卡约50–70分钟，以实际分片计时和日志为准。

25项CPU测试通过（原13项＋新增12项），包含600→4×150的唯一覆盖、分片缺失/重复/错配拒绝、
预测/评分合并及原顺序恢复、重复合并不变、完成分片评分复用、最慢分片ETA、
真实CPU子进程的四卡环境绑定，以及一片失败后其他子进程退出。
轻量检查没有运行真实GPU教师推理；四卡显存、速度及准确率待用户长评测验证。

<a id="qwen-drive-results"></a>

## 2026-10-20：四卡600题筛选完成及结果核验

用户运行`screen --gpus 1,2,3,5`，于12:32:53 UTC完成；本次于13:22:20 UTC核验。
8题smoke通过，600题dev（400视频，Action433／Scenery167）的生成及Judge评分全部完成，
四片各150题，`dev report validated=True`。600份教师答案均有EOS、非空且无未闭合thinking；
因此本轮总体差距不能归因于答案截断或未完成。smoke只验证生成，界面中的Judge 0/8为预期。
official教师评测仍未运行；没有启动任何蒸馏训练。

| 模型 | Judge正确／600 | Judge | Token F1 | Exact |
|---|---:|---:|---:|---:|
| SFT parent，复用mechanism预测 | 350 | 58.33% | 0.6401 | 9.17% |
| 保留40k CE，复用历史预测 | 354 | 59.00% | 0.6407 | 9.00% |
| Qwen-Drive教师 | 341 | 56.83% | 0.4253 | 4.17% |

相对parent，教师修复103、退化112，净−9题（−1.50个百分点），报告的视频bootstrap
95%区间为[−6.32,+3.51]个百分点。相对保留CE，修复97、退化110，净−13题
（−2.17个百分点），区间[−6.92,+2.50]。区间跨零，不能据此宣布教师在总体上显著更差，
但本轮没有证明其优于学生，也不支持直接把当前配方升级为大规模蒸馏。

| 来源 | N | 保留CE | 教师 | 教师修复／退化 | 净差／百分点 |
|---|---:|---:|---:|---:|---:|
| Action | 433 | 54.50% | 56.12% | 84／77 | +1.62 |
| Scenery | 167 | 70.66% | 58.68% | 13／33 | −11.98 |

Action的配对区间为[−4.35,+7.39]个百分点，尚不能称为可靠的局部优势；Scenery为
[−18.54,−4.73]个百分点，是本次样本中总体差距的主要来源。上述区间来自固定模型的
视频重采样，不覆盖训练随机种子、反复开发选择或外部训练数据重合带来的不确定性。

教师回答平均39.30个空格分词，保留CE为16.83；教师F1和Exact也明显更低。
这表明输出长度和词面匹配存在差异，但尚未区分额外细节错误、回答风格和Judge口径的影响，
不能仅凭F1就判断视觉能力，亦不能假设缩短答案即可解决问题。32例分层复核页面及审核
空表已生成，尚未进行逐例视觉审核。97条修复来自开发集，不应直接拿去训练再用本开发集报提升。

`teacher_qualified_for_distillation=false`保持成立：当前没有总体优势证据，外部教师的
逐视频训练来源也未核实。内部dev来自公开LingoQA train，不能称为外部教师未见的测试集。
本轮只完成教师筛选，不能把这个分数理解成已经蒸馏后的学生性能。

实际耗时以落盘时间为依据：第一批smoke worker于12:13:37 UTC启动，至完成约19分16秒，
不含其前的安装与下载，也不含首个worker启动前的控制器准备。smoke启动至dev启动约5分钟；
dev启动至Judge启动约13分25秒（含模型启动和生成合并）；Judge启动至review启动约44秒；
review及最终报告约7秒。各dev worker记录的PyTorch峰值allocated显存约9.45 GiB，
不等于nvidia-smi显示的进程总显存。没有单卡同配置基准，不能把此前2.5–3.5倍预估改称实测加速比。
用户日志中的56分33.721秒是累计CPU时间，并非实际等待时间。

本次重新检查报告引用的33个来源文件SHA、实验及dev输入SHA、9个代码SHA；
核对三方600个唯一ID及问题／参考／segment身份、预测与评分匹配，并从逐题评分重算总体
Judge/F1/Exact、分来源Judge及配对修复／退化，均与报告一致。bootstrap区间引用原报告，
本次未另行重采样；未重新哈希完整模型权重，未重跑推理或修改原始报告。

结果见[报告](../outputs/qwen_drive_teacher_4gpu_20261020/dev_report.md)、
[完整JSON](../outputs/qwen_drive_teacher_4gpu_20261020/dev_report.json)及
[32例待审核页面](../outputs/qwen_drive_teacher_4gpu_20261020/dev/review/review.html)。
时间、核验、逐分片运行信息与产物指纹保存在
[结果快照](artifacts/qwen_drive_20261020/results_snapshot_20261020T132220Z.json)。

<a id="qwen-drive-protocol-audit"></a>

## 2026-10-20 13:32:22 UTC：官方成绩、评测口径与争议案例审核

用户质疑56.83%与公开成绩不符。本次查阅官方仓库、论文、VQA入口及LingoQA评分代码，
仅执行原始数据核对、CPU tokenizer长度审计和两例首末帧查看，没有新推理、重评分或训练。
**56.83%的计数仍正确，但其范围是本项目自定义接入及内部600题；不是官方标准成绩的复现，
也不能直接外推为教师本身在LingoQA上只有这一水平。**

官方表格报告Qwen-Plus评分77.8，并单独说明官方LingoJudge下为79.4；因此不能用
“作者用了不同Judge”单独解释差距。
来源：[官方结果](https://github.com/QwenLM/Qwen-Drive-1.0#driving-vqa)。

已确认的差异及限制：

- 本轮600题来自公开训练集的内部划分，未测本项目准备的官方500题；不能把不同集合的
  56.83与79.4当作直接复现差值。
- 论文附录C.1的LingoQA案例强调回答视频最终动态结果；本轮使用自行添加的驾驶视角、
  仅依据可见证据及简洁回答说明，没有这一限定。提示差异已确认，其影响未做因果对照；
  论文单个案例也不足以证明全部benchmark题目使用完全相同提示。
  来源：[论文附录C.1](https://arxiv.org/html/2609.00111v1#A3.SS1)。
- 上游VQA入口为近确定性采样（top-k=1、temperature=0.01、top-p=0.001），上限32768；
  本轮为greedy、上限512。600条均正常EOS结束，实测最大264 token，未出现生成预算耗尽。
  不能把512上限直接归为差距原因；top-k=1本身也接近greedy。
  来源：[固定版本VQA入口](https://github.com/QwenLM/Qwen-Drive-1.0/blob/28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0/scripts/run_vqa.py)。

LingoJudge会把拼接的问题、参考答案和待评回答限制为128 token。本次用实际Judge tokenizer
检查全部三方输出：教师82/600题超限，parent与保留CE均为0/600；超限教师回答分为Action69、
Scenery13，其中51题仍判正确、31题判错。**这是评分输入截断，不是教师生成中断；
82条不能被当成82个因此产生的错误。** 是否改变评分、改变多少尚未对照，不能直接将
Judge长度改为512后冒充原官方协议。当前评分的lower/strip、max-reference、logit>0及128上限
与公开实现一致。来源：[官方Judge代码](https://github.com/wayveai/LingoQA/blob/main/benchmark/judge.py)。

针对性查看两例（非随机抽样，只看首末帧，不代表完成32例审核）：

1. `e3e1b7bc852a7cf43c77dd33a8a8a15f`：题目问有无交通标志，原始参考为“没有”；
   教师回答交通岛上有蓝底白箭头的靠左行驶标志，Judge判错。首末帧中该标志均清晰可见。
   parent与保留CE均回答“没有交通标志”，Judge均判对。
   这是至少一例参考与可见证据冲突，不能直接归为教师看错；仍不足以推断此类问题占比。
2. `04feb048f6f16f23a66d8fcbace33fc4`：参考为无人行走，教师称右侧有人。首帧右侧近处
   确有一人，末帧该近处行人已离开视野。此例提示“任意帧”与“最终场景”的时间指代歧义，
   本次不将其裁定为标注错误。

另外将全部600题逐项回查原始Action／Scenery `train.parquet`：question_id、segment_id、
问题、答案、五帧路径及顺序全部匹配，未发现数据转换串题。两例问题也直接存在于原始表格。
检查使用官方VQA processor、五张RGB图和原有完成记录，未发现漏帧或空输出；本次没有重新
加载完整模型、对不同attention实现作数值等价测试或重跑上游端到端基线。

论文明确将LingoQA列为训练来源之一，但未公开到本次600题的视频级重合清单；不能称本次
600题是外部教师未见集，也不能反向认定每条都见过。
来源：[论文数据配方](https://arxiv.org/html/2609.00111v1#S2.SS3.SSS2)。

当前结论收窄为：本项目配置下的Judge成绩未超过学生，标准成绩尚未复现，具体差距来源
未查清。应先做冻结样本的提示词／接口对照，再按可核验的官方设置复核官方500题；
这些步骤尚未执行，不能承诺改提示就达到79.4，也不因此修改已有分数、标签或训练集。
两例图像证据路径／指纹、原始行、600题数据核对和逐题Judge长度见
[协议审核快照](artifacts/qwen_drive_20261020/protocol_audit_20261020T133222Z.json)。

<a id="qwen-drive-scenery-adjudication"></a>

## 2026-10-20 14:11:17 UTC：Scenery 全部分歧题五帧视觉仲裁

完成内部600题中Scenery全部46道教师／保留CE的LingoJudge判分分歧题：
原Judge判教师退化33题、修复13题。每题实际查看按时间排序的完整5帧，
共230次帧呈现、195张不同路径图片。不是随机抽样，也不是官方500题。
本轮由Codex及并行审阅代理看图，**属于模型辅助视觉复核，不是独立人工金标**。
先记录匿名A/B观察，再读取参考与原Judge；回答长度仍能暴露模型风格，
第2、17、21、35、40题曾在先前上下文曝光，已明确标记，不能声称完全双盲。

保守仲裁如下；偏好表示回答整体的相对质量，包含部分正确的比较，不能当作二元正确率：

| 视觉比较结果 | 题数 |
|---|---:|
| 教师更好 | 12 |
| 保留CE更好 | 9 |
| 双方相当、均有图像支持 | 12 |
| 无法可靠分胜负 | 13 |

第25、27、29、42题另作五帧复核。深蓝／黑色受反光影响的第27题取消原先教师胜出，
降为低置信不确定；第25题双方均有问题，不强选教师；第29题远处目标以中等置信度保留。
保留初次记录、二次意见与合并理由，没有用参考答案覆盖初次观察。

明确发现及边界：

- 第13、17、18、20、21、22、26、35、41题：双方回答与参考的主要结论得到画面支持，
  无明显时间范围歧义且审阅置信度高，原Judge却一对一错。教师被拒6题、CE被拒3题。
  其中第17题“无行人”、第20题“无阻挡”、第35题“无STOP标志”是直接的同义判分不一致。
  9题是本次复核类别计数，不能外推Judge在600题的错误率，也不能断言都由答案长度造成。
- 第40题参考“没有交通标志”，五帧有清楚的靠左通行标志；第43题参考“没有摩托车骑手”，
  最后两帧左侧路口有清楚的踏板摩托车及骑乘者。两题均为教师识别正确却被拒、
  CE跟随否定参考获得正确判分，是高置信参考／图像冲突。
- 第25题右侧停靠车辆、第29题远处同向前车另有中等置信参考冲突；
  第25题教师又把显眼的对向白厢车当作停车，不能简单视为教师完全正确。
  四题参考冲突分别保留置信度，不写成“四个已获人工认证的标注错误”。
- 教师也有真实错误：第5题添加20限速而画面出现30、第6题虚构自行车等待区、
  第16题将移动车当停放、第19题重复计数行人与拖车、第33题混淆行人与行走方向、
  第36题把双向道路说成同向车道。部分正确答案中的附加错误单列为partial，
  不能把这些损失都解释为格式问题。
- 第2、10题早帧有人、后帧离开；另有安全车速、动作动机、模糊指代与计数范围问题。
  五帧不足时保留uncertain／partial，不强行恢复成一个“正确答案”。

因此原33退化／13修复不能直接理解为视觉能力净退化20题；本轮发现评测噪声、
参考缺陷、时间／对象范围歧义与真实模型错误并存。也不能把12胜9负换算为教师
总体优于学生：双方同判对和同判错的其余题未审，Action未纳入，600题不重算“校正准确率”。
原预测、参考、Judge分数和训练数据全部保持原样；原始三份输入SHA256复核一致。
本轮没有新GPU生成、评分或训练。

关于蒸馏前SFT：应先对齐任务和输入／回答规范，但额外再训一次教师不是OPD的必需步骤。
GKD原论文实验从已SFT的学生开始，假设学生已能生成可供教师反馈的合理序列。
Qwen-Drive本身已做驾驶VQA训练，数据配方含LingoQA，且做过回答格式统一；
其一致性过滤是和来源标注比较，不能解读为逐帧校验并纠正了来源标注。
本项目尚未做教师二次格式适配。合理的对照是先冻结时间口径和简洁回答提示，
必要时只用训练集的干净样本做小规模SFT，再与未适配教师比较；
不得把这46题或官方500题作为训练目标来修饰成绩。
参见[GKD初始化](https://arxiv.org/html/2306.13649v3)、
[Qwen-Drive数据配方](https://arxiv.org/html/2609.00111v1#S2.SS3.SSS2)。

Judge问题有公开先例：LingoQA作者说明其分类器不接收视频，针对官方问题与人工回答风格优化，
对新问题／回答风格泛化存在限制，附录还展示过长答案被高估的情况。
这说明误差方向可以是高估也可以是低估；不能以“Judge有局限”替所有教师错误开脱。
参见[LingoQA局限与风格研究](https://arxiv.org/html/2312.14115v4#S6)。

产物：

- [46题图文仲裁页面](../outputs/qwen_drive_scenery_adjudication_20261020/review.html)
  （每题可展开五帧，图片使用现有`/data`文件的相对路径，需在能访问原数据的机器打开）。
- [逐题合并结果](../outputs/qwen_drive_scenery_adjudication_20261020/adjudications.jsonl)、
  [统计摘要](../outputs/qwen_drive_scenery_adjudication_20261020/summary.json)。
- [带时间与输入／图片指纹的完整归档](artifacts/qwen_drive_20261020/scenery_adjudication_20261020T141117Z.json)。

<a id="qwen-drive-official-results"></a>

## 2026-10-21 03:43:03 UTC：官方500题完成、结果核验与下一步

官方任务完成时间为2026-10-21 02:33:21 UTC（workflow时间戳1792550001.642316）。
此前10-20的首次official任务在GPU1加载教师时OOM，0/500；用户随后用GPU1/2/4/5
补跑完成，每个分片125题。实验权重、提示、解码、像素预算与600题筛选一致，仅换物理卡号。
服务为`mpp-qwen-drive-official-retry-20261021.service`。
用户日志中的39min47.533s是累计CPU时间，不是实际墙钟耗时；分片记录的生成阶段耗时
约12分16秒至13分37秒，含各自模型加载，不代表完整工作流耗时。

### 结果与配对分析

| 模型 | Judge正确数 | Judge正确率 | Token F1 | normalized exact |
|---|---:|---:|---:|---:|
| 原全量SFT parent | 271/500 | 54.20% | 0.3196 | 0.40% |
| 保留CE | 277/500 | 55.40% | 0.3169 | 0.20% |
| Qwen-Drive教师 | 349/500 | **69.80%** | 0.3084 | 0.60% |

按原Judge标签，教师相对保留CE的配对关系为：

| 保留CE／教师 | 教师对 | 教师错 |
|---|---:|---:|
| 保留CE对 | 221 | 56 |
| 保留CE错 | 128 | 95 |

教师修复128题、退化56题，净增72题，即+14.40个百分点。原报告按100个segment重采样
1000次，95%区间为[+9.60,+19.00]个百分点；对parent修复130／退化52，净增78题，
即+15.60个百分点，区间[+10.80,+20.40]。本次重算配对计数并核验来源，区间沿用原报告，
未重新bootstrap。区间描述固定模型在这组视频上的重采样不确定性，不涵盖训练随机种子、
模型选择、Judge错误或外部教师训练重合。

这组结果支持将Qwen-Drive作为小规模知识迁移试验候选。它同时说明不能无差别替换全部
监督：56道原本CE正确的题，教师按Judge为错。两者共同错的95题也不能靠直接复制当前
教师回答解决。以上“对／错”均为原Judge标签，不是本轮人工逐题认证。

教师500条回答均正常EOS结束，无未闭合thinking，最大completion为330 token，低于512
生成上限。官方回答平均33.59个空格分词，CE为14.08；教师F1略低而Judge更高，说明两个
指标没有给出相同排序。没有做长度或措辞的受控干预，不能把差异单独归因于答案长短。
本次未重新测量官方评分输入的128-token截断数量，不能把此前dev的82/600直接套到这里。

### 为什么官方500题与内部600题不同

| 评测集合 | parent | 保留CE | 教师 | 教师对CE修复／退化 | 净变化 | 视频bootstrap 95%区间 |
|---|---:|---:|---:|---:|---:|---:|
| 内部dev600 | 58.33% | 59.00% | 56.83% | 97／110 | −2.17 pp | [−6.92,+2.50] pp |
| 官方500 | 54.20% | 55.40% | 69.80% | 128／56 | +14.40 pp | [+9.60,+19.00] pp |

已核验两套教师推理身份中的权重、processor来源、五帧模式、提示词、greedy、bf16、SDPA、
生成上限和像素预算相同；这不是“修改提示或SFT后56.83涨到69.80”的前后实验。
两个百分比来自不同样本集合，不可直接相减作为模型提升。

内部dev来自公开训练集的Action／Scenery划分；官方来自evaluation集合。本次检查准备数据：
内部600题全部只有1条参考，官方455题有2条参考、45题保留1条参考。评分均采用
`official_lower_strip_maxref_logit_gt0_maxlength128_v1`。max-reference会接受任一参考支持
的回答；参考覆盖度是可能影响评分的因素，但没有在相同问题上做单／多参考受控评分，
因此不能确定其贡献，更不能解释全部分差。

此前Scenery全46道分歧题的视觉仲裁已经发现参考冲突、同义判分不一致、时间范围歧义和
教师真实错误并存。那是模型辅助复核，不是独立人工金标，也不是本次官方500的审核。
官方报告新生成的32例复核页目前仍未逐例看图，不能写成“官方修复已人工确认”。

### 当前结论及限制

- **有证据的结论**：当前冻结教师在这套官方500协议上明显超过两组学生，可开展小规模
  迁移试验；此前内部dev未显示总体优势的结论仍保留。
- **尚未证明**：学生蒸馏后能超过匹配CE、能保留原有能力，或教师对未见驾驶分布有普遍优势。
  教师使用不同的视觉处理分辨率，能看到的细节不一定可传给当前学生的EVA表示。
- 官方500已在项目中反复使用，不是全新盲测；外部教师训练视频级重合尚未核实。
  这次69.80%也不等于完成作者整套公开成绩的严格复现。
- `observed_advantage_over_retained_ce=true`是本轮观测优势；
  `teacher_qualified_for_distillation=false`是当前筛选脚本固定的保守标记。
  二者不冲突，false不是任务失败，也不表示本轮教师没有胜过CE。本次不修改这个标记。
- 教师已经具备驾驶VQA训练基础，二次格式SFT不再作为本轮迁移试验的必需前置。
  若后续训练侧验证发现稳定的格式／任务适配问题，再单独做适配对照。

<a id="qwen-drive-transfer-pilot-plan"></a>

### 下一步：4k候选、500步的离线答案蒸馏对照（方案，尚未执行）

本阶段要回答“这个教师的监督能否帮助当前学生”。明确这是**离线答案蒸馏**，不是已有
同词表token OPD的直接延长；先建立可迁移性基线，再判断在线学生轨迹是否额外有益。

1. 冻结本轮Qwen-Drive权重、五帧输入、提示和解码，不用官方500选择提示或训练目标。
   学生初始化采用已保留CE的完整三路权重：
   `outputs/mpp_feedback_sft_control_40k_2epoch_v2/final/{adapter,projector.pt,qformer_adapter.pt}`。
   真实配置为完整Q-Former；文件名`qformer_adapter.pt`不代表此模型只训练Q-Former LoRA。
   两臂重新建立优化器并采用相同参数更新范围。
2. 仅从训练划分按Action／Scenery来源固定抽取约4000条候选，记录种子、视频及问题ID，
   核验与dev／official在问题、segment、帧路径上的隔离。教师和学生生成均不看参考答案。
   Judge只用于候选分层和辅助筛选；抽查看图覆盖修复、退化、双方正确／错误，避免把
   Judge判对直接视为已验证的教师监督。不得把官方128道修复题或已仲裁的dev题移入训练。
3. 审核可信教师纠错目标并保留原始答案回放，记录每条`target_source`，不覆盖原始`answers`。
   两臂共用最终样本ID、顺序和回放位置：CE臂全部用原始答案；蒸馏臂在可信位置使用教师
   答案，其余使用原始答案。若原始参考本身有疑点，该条从两臂同步排除、单独归档，不强行
   用错误参考作对照。实际教师目标数、回放数和有效N均需披露，4000候选不等于4000教师目标。
4. 先设有效batch16、每臂500次优化更新，固定学习率、训练模块、种子和采样计划。
   每臂共8000次样本呈现；有效N若少于4000，要记录循环／重复采样策略，不能仍称为两epoch。
   两臂均使用同一采样计划，并记录实际答案token数与耗时；相同更新预算不等于相同token计算量。
5. 预先固定比较两臂final，在完整4320题dev上与起点对比，报告Judge、Action／Scenery、
   修复／退化、视频配对区间，以及起点正确题的保留情况。主要判断是蒸馏是否超过匹配CE，
   不能只看是否超过起点。依据dev决定保留模型，官方只作选后复核，不用它挑参数或checkpoint。
   单次小试验若只差几题、区间跨零，结论应为未建立收益，不能直接扩大训练并宣称成功。

现有`train_sft.py --target-mode distillation`及`training_answer`字段可复用，但旧
`teacher_distillation.py`绑定旧教师、旧检查点目录与身份门槛；`--parent-run`也要求旧数据身份
和布局。新的4k实验需要明确三路权重加载、配置和来源核验，不能直接套用旧runner命令。

教师与学生词表不同，离线方案可将教师答案转为文本、再用学生tokenizer编码；不能把教师
logits或token ID直接交给旧KL/JSD，解码再编码也不等于概率分布对齐。若迁移小试验有效，
下一阶段再设计当前学生轨迹上的在线反馈训练；若要采用严格token分布OPD，还需要明确
跨词表分布对齐或兼容教师方案。在线批改答案训练与token分布OPD应分别命名和对照。

03:43 UTC结果归档时只完成核验与方案。后续已实现离线候选及双臂入口，并准备4000条候选清单；
尚未开始正式生成／训练／评测，在线接口仍不在本次范围。GPU短测未完成，不声称已通过；
正式启动命令包含自动GPU短测前置。长任务及监视命令见[实施文档](qwen_drive_distill_20261021.md)。

### 核验及归档

重算dev600与official500三方逐题总体指标及配对计数，检查问题／视频／参考身份、原预测与
评分预测一致性、完成状态和logit阈值。两份报告各33项来源指纹验证通过，共记录75份唯一
来源文件SHA256，另检查9个当前代码SHA与实验记录一致。本次不重算Judge、不重新bootstrap、
不重新加载权重或对完整权重做内容哈希，也不修改原预测、分数、标签及资格标记。

- [官方结果原报告](../outputs/qwen_drive_teacher_4gpu_20261020/official_report.md)
  ／[完整JSON](../outputs/qwen_drive_teacher_4gpu_20261020/official_report.json)。
- [带时间的结果核验快照](artifacts/qwen_drive_20261021/official_results_snapshot_20261021T034303Z.json)
  保存两套指标、四格配对、参考数量、相同推理配置、来源指纹和核验边界。
- [官方32例待视觉复核页](../outputs/qwen_drive_teacher_4gpu_20261020/official/review/review.html)
  仅为待核验产物，不表示本次完成了官方视觉仲裁。
