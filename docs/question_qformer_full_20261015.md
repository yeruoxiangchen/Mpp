# 2026-10-15：问题条件 Q-Former 全量原始问答 SFT

## 当前进度：2026-10-17同步

**两组正式训练均已完成，开发评测进行中。** 下方10-15验证记录中的“尚未启动”是
实现时快照；当前状态由两组原始训练summary与最新评测产物确认。

| 分支 | summary落盘时间（UTC） | QA访问数 | 更新数 | 耗时 | rank0峰值allocated |
|---|---|---:|---:|---:|---:|
| visual-only全量对照 | 10-16 00:47:19 | 406212 | 25389 | 11.99小时 | 16.945 GiB |
| question全量模型 | 10-16 14:42:38 | 406212 | 25389 | 13.75小时 | 18.143 GiB |

两组train SHA与父模型初始化身份一致，均为四卡／累积4／有效batch16／1epoch。
`summary.completed=true`，五个计划checkpoint与final训练状态文件齐全；第1000步
语言LoRA代表参数delta均为0，最终各自可训练分支的代表参数均有更新。普通CE训练的
rollout、教师调用和JSD均为0。`progress.json`仍写最后一步的`running`不影响完成判定；
完成状态以summary和final产物为准，状态脚本已有相同优先级。

用户正在GPU4运行：

```bash
CUDA_VISIBLE_DEVICES=4 bash scripts/run_question_qformer_full.sh eval_dev &&
CUDA_VISIBLE_DEVICES=4 bash scripts/run_question_qformer_full.sh eval_selected
```

**2026-10-17 06:26:53 UTC快照**：visual checkpoint-005000的Action答案生成达到
677/3120条，question_id无重复，尚未评分。本轮新增dev评分完成0/20项；尚无完整
`report.json`、dev选模结果或`official_report.json`。该计数来自当时落盘的完整JSON行，
是固定快照，不代表之后的实时数量。

`eval_dev`先复用parent／retained的历史评分，再依次评估visual五点与question五点。
每点包含Action3120＋Scenery1200，共20个新增生成／评分任务、43200次回答。两组各按
combined-dev Judge选优，并列选更早步；dev成功结束后`&&`才启动`eval_selected`。
后者复用parent／retained的official评分，新增两组入选checkpoint各500题，共1000次
回答，保持dev选择。`status`展示的未入选checkpoint的official空项属于正常情况。

在完整报告落盘前，保留旧40k CE的60.46%／55.4%作为参考，不据训练完成推断新结构
收益。两组完整训练使匹配全量对照具备评测条件；结构结论仍需其开发结果和配对比较。
官方500已经反复使用，沿用原有结果解释边界。

原始小型JSON与进度快照见
[10-17证据清单](artifacts/question_qformer_20261017/manifest.json)，详细事件见
[时间线第70节](project_timeline.md)。本次检查文件存在性与checkpoint metadata，未重载
大权重、修改训练／评测代码或启动额外作业。

## 上一轮真实结果与本轮目的

40k／2epoch问题条件实验已经完整运行5000更新，并于2026-10-15完成自由生成评测。
两分支都由combined-dev选择final；原始报告在`outputs/question_qformer_eval_v1/`。

| 模型 | Action3120 | Scenery1200 | dev4320 | 复用official500 |
|---|---:|---:|---:|---:|
| 全量SFT parent | 58.37% | 64.25% | 60.00% | 54.20% |
| 旧40k CE final | 58.75% | 64.92% | 60.46% | 55.40% |
| 问题条件40k final | 58.30% | 64.92% | 60.14% | 53.20% |

相对旧CE，新结构dev修复73／退化87／净−14题，segment-bootstrap95%区间
[−0.92,+0.30]pp；official修复5／退化16／净−11题，区间[−3.80,−0.60]pp。
这不是新的盲测；部分Judge翻转仅对应措辞变化，不当成人工确认的读图纠正。
dev约82.64%答案完全一致。训练loss、梯度与更新正常，新文本模块和推理身份确实启用。

该配方只用了完整406212 QA的40000条。虽覆盖全部27599个训练segment，但22582段
只配一题；新增约8052万文本参数与旧视觉参数统一5e-6。结果否定当前配方的稳定收益，
不能宣称新结构已充分训练或所有问题条件特征提取均无效。本轮扩大同视频不同问题的
原始监督，并区分新旧模块优化；不保证任务正确率提升。

## 数据、模型与优化器

直接读取`datasets/processed/lingoqa_qformer_all_v1`：406212条原始QA，Action262005／
Scenery144207，27599个训练segment。原问题、答案、五帧顺序及dev／official不改写。
`full_sft_data.py`核对该train与父模型全量train的同一SHA、原始manifest、数量、
question／video／frame隔离及dev子集一致性。未要求原始数据具有OPD生成的来源标记。
原OPD／40k分支仍保留旧来源校验。复用同28099段EVA缓存，不缓存问题条件Q-Former输出。

从`outputs/mpp_lingoqa_qformer_all_1epoch_v1`重新初始化：精确继承完整视觉Q-Former、
Projector、QKV LoRA；文本embedding／FFN从本地BERT恢复。EVA、Qwen语言基座冻结。
仅原答案及EOS的CE，无Teacher/Judge训练调用、裁剪、参考答案提示或新区域标签。

| 参数组 | 峰值LR | 第1–1000更新 | 第1001更新起 |
|---|---:|---|---|
| 语言QKV LoRA | 2.5e-6 | 冻结 | 训练，独立200步warmup |
| Projector | 5e-6 | 训练 | 训练 |
| 原Q-Former／query tokens | 1e-5 | 训练 | 训练 |
| 恢复的文本embedding／FFN | 2e-5 | 训练 | 训练 |

四卡、每卡1题、累积4，完整1epoch共25389更新；最后一次只有4条真实QA，按实际
尾批归一化。桥接组3%warmup后cosine；语言组从解冻时间起warmup／cosine。fp32可训
参数、bf16基座／计算、权重衰减.01、seed44，沿用梯度检查点。两阶段共享同一epoch
数据游标，前1000更新已经包含在25389内，不额外重复或漏掉训练数据。

`sft_schedule.py`建立稳定的四组AdamW（visual-only无文本组），无遗漏／重复参数。
冻结阶段LoRA requires_grad=False且grad=None；只归约、裁剪活跃参数，避免原归约器
补零后触发冻结参数的AdamW权重衰减。解冻时不重建优化器，保留桥接组状态。保存阶段、
四组顺序、优化器、scheduler、数据游标与各rank RNG；恢复按已完成全局step决定下一
次更新阶段，并核对调度位置／LR，不重新执行适配阶段。

## 代码与报告

- `mpp_video_qa/full_sft_data.py`：完整原始QA来源与划分校验。
- `mpp_video_qa/sft_schedule.py`：分组、冻结／解冻、分组cosine、恢复与梯度记录。
- `scripts/train_feedback_opd.py`、`feedback_opd_common.py`：显式original_full普通SFT
  分支；旧默认训练配方及checkpoint身份保持兼容。每步输出阶段、各组LR／梯度范数，
  第1000步记录冻结LoRA代表参数不变。
- `scripts/run_question_qformer_full.sh`：全量question、匹配visual-only、依次训练两组、
  单卡短验证、状态、开发选模及官方评测。
- `scripts/report_question_qformer_full.py`：动态预算／checkpoint，不复用写死40k的旧
  报告；原parent与保留40k CE的评分只读复用，校验自由生成的权重／问题协议与QA。
- `scripts/smoke_question_qformer_full.py`：真实模型4次更新／累积4，阶段边界暂设2步，
  检查冻结不变、解冻更新、四组梯度与自由生成；不保存完整大checkpoint。

新输出`outputs/mpp_question_qformer_full_1epoch_v2`；匹配对照
`outputs/mpp_visual_qformer_full_1epoch_v2`；报告`outputs/question_qformer_full_eval_v2`。
保存5000／10000／15000／20000／final，每点分别生成Action3120与Scenery1200，greedy96、
batch1。两分支各按combined-dev Judge选优、平分选较早步；官方500沿用dev选择。
报告同时给出相对parent／保留CE／匹配全量CE的修复／退化、按视频bootstrap区间和
原保留模型正确／错误子集。报告允许问题分支先完成；匹配全量对照未完成时明确标记，
不能据此归因结构收益。正式预算未完成的中间报告不可用于官方选模。

## 运行命令

正式问题条件训练（运行时确认所选四卡空闲）：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=29735 \
  bash scripts/run_question_qformer_full.sh train_question
```

后台启动与监视：

```bash
systemd-run --user --unit=mpp-question-qformer-full-v2 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=4,5,6,7 --setenv=MASTER_PORT=29735 \
  /usr/bin/bash scripts/run_question_qformer_full.sh train_question

watch -n 10 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_question_qformer_full.sh status'
journalctl --user -fu mpp-question-qformer-full-v2.service -o cat
```

同预算对照单独使用`train_control`；从零依次跑两组使用`train_pair`，每组全程四卡。
已完整训练的组会由train_pair跳过。训练过程不自动启动长时自由评测。

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=29735 \
  bash scripts/run_question_qformer_full.sh train_control

CUDA_VISIBLE_DEVICES=4 bash scripts/run_question_qformer_full.sh eval_dev
CUDA_VISIBLE_DEVICES=4 bash scripts/run_question_qformer_full.sh eval_selected
```

同布局恢复question：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=29735 \
QUESTION_FULL_RESUME_FROM=outputs/mpp_question_qformer_full_1epoch_v2/checkpoint-005000 \
  bash scripts/run_question_qformer_full.sh train_question
```

对照恢复使用`VISUAL_FULL_RESUME_FROM`和`train_control`。不要将旧40k已衰减到零的
优化器状态当作新全量配方恢复。环境路径可通过MPP_*及QUESTION_FULL_*变量设置。

## 验证记录

2026-10-15新增13项CPU测试通过，覆盖真实尾批预算、原QA完整性／frame泄漏、
冻结期间无Adam状态和权重衰减、跨阶段之前／边界／之后恢复下一步逐张量一致、
完整Bash四卡两组参数捕获、开发选模、问题协议错配拒绝及官方选择锁定。
既有问题桥／报告／feedback-v2共21项回归通过。编译及Bash语法检查通过。

沙箱内真实GPU短验证首次因CUDA不可见退出，未执行更新；已按宿主GPU环境重新运行。
该环境限制不属于训练代码报错。宿主GPU4短验证完成4次更新／累积4／16次原始QA
访问，前两步冻结LoRA梯度与代表参数delta均0；后两步解冻，梯度非零，最后代表参数
delta2.5034e-6。Projector／原Q-Former／新文本代表delta分别1.2532e-5／2.5015e-5／
4.9962e-5，自由生成成功，peak allocated18.143GiB、reserved18.434GiB。短验证临时
warmup使刚解冻的第一次更新LR为0，随后的更新实际改变LoRA，这是预期调度。
这是单GPU真实模型验证，不冒充四卡正式训练；跨阶段恢复由CPU完整优化器测试覆盖。

完整输入检查通过411032条train／dev／official QA及28099段缓存，最长问题42个
BERT token，最长监督序列334、最长generation prompt＋96为347，均在128／1536预算内。
全量数据仅29个训练segment只有一题，与旧40k的22582段单题覆盖明显不同。
预检时`new_training_started=false`。正式全量训练由用户启动和监视，没有自动后台作业。
六份小型原始配置、结果与验证快照随源码保存在`docs/artifacts/question_qformer_20261015/`，
没有新增大checkpoint副本或上传数据。旧21项与新13项共34项CPU测试通过。
