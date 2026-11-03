# 2026-10-21：Qwen-Drive → retained CE 离线答案蒸馏小试验

本轮实现用户确认的迁移试验。正式候选生成、500步训练和完整评测由用户启动。
固定教师为此前官方500题69.80%的Qwen-Drive，起点为55.40%的retained CE。
这两个数不是本轮蒸馏结果；本轮是离线文本目标蒸馏，不是token分布OPD。
2026-10-21 05:18 UTC完成CPU清单准备：4000题、4000个视频，Action 2580／Scenery 1420。
05:20:40 UTC归档初次实现与验证证据。最新状态：12:25:10 UTC用户启动的`train-eval`正常结束，
CE／distill各500步及完整4320题dev评测完成。起点60.46%、CE 59.42%、distill 59.19%；
dev选定仍为`start`，未建立蒸馏收益。`official_evaluated=false`为预设行为，尚无本轮官方报告。
本轮3979条有效样本，502条教师目标、3477条参考回放；两臂共同排除21条。
详见文末结果分析及[开发集报告归档](artifacts/qwen_drive_20261021/transfer_result_20261021T134500Z/dev_report.md)。

## 已实现的实验约束

- 训练集按Action／Scenery比例、固定种子44、视频轮转抽4000候选，每视频最多2题；
  校验问题ID、视频ID、帧路径与dev／official隔离。原始问题、参考及五帧顺序保持不变。
- 教师与起点学生均不读取参考答案提示；教师保持原有greedy、512新token、
  每帧921600像素预算与驾驶视角提示，学生保持五帧、legacy提示、96新token。
- 原版Lingo-Judge口径不改，另保存截断前组合输入token数。教师目标须为教师Judge判对、
  学生判错，教师正常EOS、无未闭合thinking、答案非空且不超过96个空格分词；两方所有
  参考组合均不超过128个Judge token。该长度限制是本次预设筛选规则，不是对旧评测改分。
- 按来源×四种Judge结果抽查最多32题，逐题查看五帧并记录参考、教师、学生是否被图像支持。
  参考存疑／矛盾的抽查题从两臂同步排除；参考可信而教师不支持／不确定时回放原参考。
  未被抽查的教师目标仍只是规则筛选目标，不能写成“全部视觉审核正确”。
- 保留其余候选作原答案回放，包含教师与学生均答错的题；不预设回放比例，不强凑教师样本。
  实际教师目标、回放数、有效N、来源比例和排除原因写入`data/selection.json`。
- 两臂分别为`ce`和`distill`。两者共用样本、顺序、初始化、优化器及学习率；
  CE全部用`answers[0]`，distill只在合格位置用`teacher_answer`，不覆盖`answers`。
- 三路初始化：`outputs/mpp_feedback_sft_control_40k_2epoch_v2/final/`下的
  `adapter/`、`projector.pt`、`qformer_adapter.pt`。这里是完整Q-Former。
  全部dropout关闭，BF16计算、可训练参数FP32；语言LoRA／投影层学习率2.5e-6，Q-Former 5e-6，
  AdamW、weight decay 0.01、warmup 0.03、答案位置CE。两臂各建新优化器。
- 每臂500次更新，有效batch16，共8000次有效样本呈现。`sample_plan.json`按有效N反复
  打乱循环至8000条，无尾批填充；DDP按全局顺序分发。过滤后不称为固定“两epoch”。
  按学生实际tokenizer统计被监督token（含结束标记），不能把相同更新数说成相同token成本。
- 每50步原子保存检查点；重跑同一命令自动续训。采样计划、目标类型、学习率、种子、
  GPU数量等身份变化会拒绝恢复。GPU物理编号可以换，worker数量须保持不变。
  首个检查点前失败会归档整次尝试后从0重启；残留`.incomplete`目录归档保留。
- 两臂顺序训练，每臂用全部指定GPU做DDP；生成和Judge按题分片，每卡独立模型副本。
  不是将两臂各占两卡，也不是教师与学生共同驻留显存。

## 第一段：候选生成、评分和待审材料

下面示例使用GPU 1、2、3、5。启动前用`nvidia-smi`确认这些卡可用；可以换成其他四个编号。
若只用两张卡，须从`bank`开始使用单独输出目录，例如
`--gpus 1,2 --output-dir outputs/qwen_drive_distill_4k_2gpu_20261021`，以后各命令带相同参数。
不能在已有四分片实验中途改成两分片；两卡训练仍保持有效batch16。

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
nvidia-smi
systemd-run --user --unit=mpp-qwen-distill-bank-4k-20261021 \
  --property=WorkingDirectory="$PWD" \
  /usr/bin/bash scripts/run_qwen_drive_distill.sh bank --gpus 1,2,3,5
```

监视文件进度和真实进程日志：

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_qwen_drive_distill.sh status'
journalctl --user -fu mpp-qwen-distill-bank-4k-20261021.service -o cat
```

日志在`outputs/qwen_drive_distill_4k_20261021/logs/`，分卡日志名称含`teacher`／`start`／`judge`。
加载模型期间题数暂时为0可能正常，需结合日志和进程检查；进度时间戳不作为存活证明。
命令中断后可用新服务名（例如追加`-retry1`）重跑相同`bank`命令，已生成／已评分题会校验复用。
不必删输出目录，不必重新download，旧`run_qwen_drive_teacher.sh official`也不是这轮的入口。

第一段完成时，状态为 **`awaiting_visual_audit`**，这是候选已生成后的预期阶段。
首次`bank`会先在独立`smoke/`目录自动完成8候选、两臂各2步、CE断点恢复及各8题生成评分。
只有该GPU短测成功才继续正式4000候选。短测数据与权重不会用于正式训练；
若短测失败，服务报错并保留日志，修复后重跑相同命令，不能把它误当成4000题已跑完。
短测阶段总状态为`preflight_smoke`，正式候选仍为0/4000；细节可查看
`tail -f outputs/qwen_drive_distill_4k_20261021/logs/preflight_smoke.log`，分模型日志在`smoke/logs/`。
产物：

- `bank/review/review.html`：五帧、问题、原参考、师生答案、筛选原因。
- `bank/review/cases.jsonl`：固定抽查清单及身份。
- `bank/review/decisions.jsonl`：待填写的逐题证据；不会自动把pending改为reviewed。

这时把完成状态交给Codex做短的看图审核即可，不需要用户手工写JSON。
审核字段包括五帧查看记录、逐方图像支持与否、理由、审核者、UTC时间；允许不确定。
这是已约定的监督质量检查，不是启动训练的额外授权。候选尚未产生时不能提前完成这项审核。

## 第二段：匹配双臂训练和完整dev

看图审核完成后运行：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
systemd-run --user --unit=mpp-qwen-distill-train-4k-20261021 \
  --property=WorkingDirectory="$PWD" \
  /usr/bin/bash scripts/run_qwen_drive_distill.sh train-eval --gpus 1,2,3,5
```

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_qwen_drive_distill.sh status'
journalctl --user -fu mpp-qwen-distill-train-4k-20261021.service -o cat
```

该命令先验证审核并落盘共享训练数据和计划，训练CE 500步、distill 500步，再各生成完整4320题dev。
起点使用已验证的retained CE原评测缓存，并核对三路权重内容指纹及原问题／参考身份，避免重复推理。
任何不完整训练或缺题、重复题、不同模型评分都不能生成完成报告。

结果为`dev_report.md`／`dev_report.json`：总体及Action／Scenery，修复／退化，
按视频配对bootstrap 95%区间，起点正确题保留率；同时记录实际监督token、耗时及模型身份。
只比较预先固定的两个final和起点，不对中间检查点择优。
按dev准确率选point-best；平分保留起点，其次CE。另单独报告`distillation_gain_established`：
仅当distill相对CE的视频配对区间下界大于0，且超过起点时才为true。
point-best不等于建立可靠收益；跨零时不能直接宣称OPD成功或扩大训练。

## 选后官方复核（单独启动）

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
systemd-run --user --unit=mpp-qwen-distill-official-4k-20261021 \
  --property=WorkingDirectory="$PWD" \
  /usr/bin/bash scripts/run_qwen_drive_distill.sh official --gpus 1,2,3,5
journalctl --user -fu mpp-qwen-distill-official-4k-20261021.service -o cat
```

只评dev选定模型，与已存档起点比较；若dev选回起点，复用其已验证官方结果，不重新择优。
官方500已被历史实验重复使用；外部教师是否见过相关视频仍未知，不作全新盲测声明。
原教师筛选报告`teacher_qualified_for_distillation=false`保持不变；本轮是基于已记录证据、
用户确认的探索性迁移试验，不伪装成原自动门槛已通过。

## 初次实现时的验证与状态（05:20 UTC，历史记录）

CPU：69项测试通过，覆盖分割隔离、配对监督、参考保留、Judge截断过滤、审核身份及完整性、
精确采样和续训顺序、三路初始化来源、旧教师分片及评分回归；Python与shell语法检查通过。
GPU短测曾在1、2卡启动，仅完成教师加载阶段便因共享机器的显存占用变化主动停止：
GPU1随后出现另一项ChartQA进程约17GB显存占用。已终止本次控制器及其两个教师子进程，
没有终止其他任务。该次目录为`outputs/qwen_drive_distill_smoke_20261021_v1`，
**没有GPU短测通过记录，不声称训练或续训实测成功**。修订后的正式`bank`命令会在用户选定
可用GPU后自动先做完整短测；旧中断目录仅保留排查，不混入新实验。
正式4000候选、500步双臂和完整dev／official均未启动。

[05:20:40 UTC实现快照](artifacts/qwen_drive_20261021/transfer_pilot_implementation_20261021T052040Z.json)
记录代码、数据及证据SHA256、初始化、预算和GPU验证边界；
[CPU测试原始日志](artifacts/qwen_drive_20261021/transfer_pilot_cpu_checks_20261021T052040Z.txt)。

实现入口：[runner](../scripts/qwen_drive_distill.py)、[shell](../scripts/run_qwen_drive_distill.sh)、
[数据与采样契约](../mpp_video_qa/transfer_pilot.py)、[可续跑评分](../scripts/score_transfer_pilot.py)、
[固定计划训练](../scripts/train_sft.py)、[测试](../tests/test_transfer_pilot.py)。

## 2026-10-21 05:34 UTC：用户四卡短测在学生加载阶段收到SIGTERM

用户以GPU 2、3、4、5启动`mpp-qwen-distill-bank-4k-20261021.service`。
短测教师的8条答案及8条Judge评分均已完成，四个教师分片正常退出；随后进入学生权重加载。
05:34:43 UTC，首先检测到GPU5上的学生worker（PID2709313）以`-15`退出，即收到SIGTERM。
该分片日志停在MPP权重第11/16片，没有Python异常或CUDA OOM堆栈；控制器随后清理其余
worker，外层服务05:35:00 UTC以`exit-code`失败。四个学生分片均0条答案，训练尚未开始，
正式4000条候选生成也没有开始。

服务未配置运行时长或内存上限（`RuntimeMaxUSec/MemoryMax/MemoryHigh=infinity`）。
现有日志不能确定SIGTERM发送者；当前账户看不到系统／内核日志，不能据此把原因定为OOM、
人工终止或程序异常。四卡同时载入权重不代表显存溢出已被证实。

无需修改实验代码或删除已有结果；等所选四张卡可用后，使用新服务名重跑相同命令：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
systemd-run --user --unit=mpp-qwen-distill-bank-4k-20261021-retry1 \
  --property=WorkingDirectory="$PWD" \
  /usr/bin/bash scripts/run_qwen_drive_distill.sh bank --gpus 2,3,4,5
journalctl --user -fu mpp-qwen-distill-bank-4k-20261021-retry1.service -o cat
```

已完成的教师答案和评分会校验复用。保留四worker数量可改物理GPU编号，避免与其他在跑任务
争用资源；本次排查没有代用户重启长任务，没有修改预测、评分或代码身份。

06:17:26 UTC核验了formal／smoke两份代码身份、四个教师分片及合并评分；8条教师答案均正常EOS。
来源日志和结果指纹保存于[本次SIGTERM排查快照](artifacts/qwen_drive_20261021/transfer_pilot_sigterm_20261021T061726Z.json)。

## 2026-10-21：retry1再次中断后的CPU加载修复

retry1（GPU 2/4/5/7）仍在学生加载阶段失败，首先检测到GPU4进程2842957以−15退出，
日志停在MPP第12/16分片。主机125GiB RAM、Swap已满；系统earlyoom服务正在运行，
配置为内存／Swap阈值8%／20%，优先选择Python进程。学生语言权重每份约14.382GiB，
四份同时在CPU驻留约57.53GiB，尚未计入临时张量和视觉模块。以上支持主机内存压力的解释，
但本账户无法读取完整系统日志，未直接确认这两次SIGTERM的发送者。

已为本轮学生生成和训练增加共享文件锁：每次只允许一个进程加载CPU权重，立即搬到自己的GPU、
释放CPU暂存后才让下一个进程加载。生成和训练计算仍使用指定的四张卡。未更改权重、精度、
题目、训练预算、损失或Judge口径。训练入口也在当前rank加载完成后立即搬到GPU，避免仅串行
构建却把四份模型累积留在CPU。锁随进程退出由内核释放；其他未指定锁的历史入口保持原行为。

CPU回归50项及并发锁测试3项通过。正式与smoke实验身份已做有记录的迁移：核验尚无学生答案、
筛选和训练产物，保留110个文件内容不变，含8条教师答案及评分。旧配置、源码和两次失败日志
保存在[加载修复归档](artifacts/qwen_drive_20261021/loading_repair_20261021T072514Z/)。

两卡加载验证在GPU4/5生成4条回答，两个进程均正常退出，耗时289.78秒，证据证明两段CPU加载
不重叠；详见[验证结果](artifacts/qwen_drive_20261021/loading_repair_20261021T072514Z/gpu_loading_check.json)。
这不是完整四卡smoke通过，也未验证本次训练改动的GPU执行。验证产物单独保存于
`outputs/qwen_drive_loading_check_20261021`，不混入正式或smoke数据。
用户再次明确要求长任务自行启动；该次验证已结束，此后包括GPU验证在内的生成、训练、评测
均提供命令由用户启动。

确认所选四张GPU可用后，由用户执行（保留原输出目录）：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
systemd-run --user --unit=mpp-qwen-distill-bank-4k-20261021-retry2 \
  --property=WorkingDirectory="$PWD" \
  /usr/bin/bash scripts/run_qwen_drive_distill.sh bank --gpus 2,4,5,7
journalctl --user -fu mpp-qwen-distill-bank-4k-20261021-retry2.service -o cat
```

流程仍是补完8题smoke，通过后自动生成／评分4000候选，最后停在`awaiting_visual_audit`。
等待加载锁时分卡日志显示`model_load_wait`，随后为`model_load_acquired`、`model_load_released`，
并记录主机可用内存、Swap剩余和当前进程RSS；无需跳过smoke或删除旧产物。

## 2026-10-21 10:51 UTC：候选完成、视觉抽查与训练交接

用户运行的retry2成功结束，停在`awaiting_visual_audit`是预设审核关口，不是异常。08:59:16 UTC
四卡smoke通过，包含两臂各2步、CE从第1步恢复、各8题评测；10:40:40 UTC正式教师及学生的
4000条答案和Judge评分全部完成。systemd的`Consumed 6h 44min`为累计CPU用时，不是剩余时间。

已完成固定32题、160张原图的Codex视觉复核，保存逐题证据；原参考、预测、Judge分数未改。
参考支持11、矛盾4、不确定17；不确定主要包含无法从五帧确认的驾驶意图、速度变化及语义歧义，
不能将这17条说成标注错误。按既定规则两臂共同排除21条，保留3979条，其中教师目标502条
（12.616%，Action 347／Scenery 155），原参考回放3477条。502条中4条经过本次看图抽查，
其余498条只经自动规则筛选，不是全部视觉确认的正确答案。

本次8条抽样Judge repair中，明确的“参考和教师支持、学生不支持”只有1条；4条师生均有视觉
支持，2条教师不支持，1条学生语义不确定。故Judge repair包含措辞适配，不能直接称为视觉能力
修复。小规模分层抽样不能推断总体比例。训练候选的原始Judge为起点2529/4000（63.225%）、
教师2313/4000（57.825%），修复601／退化817；这是训练划分上的候选统计，不是新dev结果。

`select`在禁用可见GPU下完成；筛选来源SHA、逐行重建、排除列表和采样计划均验证通过。
每臂500步×有效batch16=8000次呈现，蒸馏臂其中1007次用教师目标、6993次回放参考。
当前没有正式训练产物，未代用户启动GPU任务；此次只完成视觉审核、CPU筛选与归档。
详见[逐题报告](artifacts/qwen_drive_20261021/bank_audit_20261021T105138Z/report.md)及
[机器可读证据](artifacts/qwen_drive_20261021/bank_audit_20261021T105138Z/summary.json)。

确认GPU 2、4、5、7仍可用后，用户启动训练及完整开发集评测：

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
systemd-run --user --unit=mpp-qwen-distill-train-4k-20261021 \
  --property=WorkingDirectory="$PWD" \
  /usr/bin/bash scripts/run_qwen_drive_distill.sh train-eval --gpus 2,4,5,7
```

先顺序完成CE 500步、distill 500步，再分别生成并评分完整4320题dev，起点使用经校验的缓存。
不会重跑4000条bank；official仍需依据dev报告单独启动。本实验是离线文本蒸馏对照，尚无收益结论。

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_qwen_drive_distill.sh status'
```

```bash
journalctl --user -fu mpp-qwen-distill-train-4k-20261021.service -o cat
```

## 2026-10-21 12:25 UTC：双臂训练与完整开发集完成，未获得迁移收益

服务`mpp-qwen-distill-train-4k-20261021.service`已核验为`Result=success`、`ExecMainStatus=0`、
`inactive/dead`。两臂各完成500步、8000次有效样本呈现；CE监督165682 token，distill监督
182215 token。开发集三组各4320个唯一题目，24项报告来源SHA256均匹配；已从逐题评分独立
重算正确题数与修复／退化计数。systemd的4h55min是累计CPU用时，不是墙钟耗时或剩余时间。

| 模型 | 总体正确率 | Action（3120题） | Scenery（1200题） | 相对起点 |
|---|---:|---:|---:|---:|
| 起点 retained CE | 60.46%（2612/4320） | 58.75% | 64.92% | — |
| 原参考继续训练 CE | 59.42%（2567/4320） | 57.85% | 63.50% | −1.04 pp |
| 教师答案蒸馏 distill | 59.19%（2557/4320） | 57.31% | 64.08% | −1.27 pp |

相对起点，蒸馏修复189题、退化244题，净少答对55题，原正确题保留率90.66%；按400个视频
配对bootstrap的95%区间为[−2.29, −0.28] pp。CE也修复151题、退化196题，净少45题，
区间[−1.88, −0.16] pp。因此本轮并不是教师答案带来提升但幅度不足，而是两个续训分支均下降。

蒸馏相对匹配CE为−0.23 pp，修复182题／退化192题，区间[−1.13, +0.61] pp跨零，不能声称
教师监督显著优于或劣于原参考监督。Scenery相对CE为+0.58 pp，但区间[−0.92, +2.00] pp
同样跨零；Action为−0.54 pp。以上是单随机种子下的视频抽样区间，不覆盖训练种子波动。

这是离线文本目标迁移，不是学生在线轨迹上的KL/JSD OPD。仅502/3979条（12.62%）改用
教师答案，实际8000次呈现中教师1007次；其余6993次仍是参考回放。不能由教师官方69.80%
推断这些自动筛选目标必然改善学生。CE同时下降提示需要先检查小子集续训的漂移、预算和
保持原能力的训练方案；这是一条诊断方向，当前两臂不能唯一证明过拟合、监督噪声或能力上限。

预设只比较固定final，dev选中`start`，`distillation_gain_established=false`。当前没有运行
官方500题；`official`入口只评dev选中模型，在此选择下会校验复用起点旧官方结果，不会评
两个新final。因此无需为补一个完成标志启动长官方评测，也不应改用官方集挑本轮模型。
建议保留起点，先做续训漂移与教师目标质量的短诊断，再决定新的训练对照；不据此扩大训练。
本次只完成CPU核验和归档，未启动任何GPU任务。

证据：[原始开发集报告](artifacts/qwen_drive_20261021/transfer_result_20261021T134500Z/dev_report.md)、[完整JSON](artifacts/qwen_drive_20261021/transfer_result_20261021T134500Z/dev_report.json)、
[复核记录](artifacts/qwen_drive_20261021/transfer_result_20261021T134500Z/validation.json)。

## 2026-10-22：已有预测与训练日志的CPU诊断，下一步复用早期检查点

已核对原报告24项来源SHA及三组逐题身份。两臂共同退化121题，CE独有75、蒸馏独有123；
答案平均词数分别为起点16.90、CE 16.87、蒸馏17.14，无明显膨胀。相同答案前后Judge判定无冲突，
新模型Judge输入超128 token仅CE 3题、蒸馏2题；500步日志无非有限loss或梯度。
不能据此排除Judge语义偏差，也不能把Judge退化直接称为视觉事实退化。

两臂训练loss后段均下降，但缺少早期dev曲线，尚不能确定过拟合。教师目标在502个位置上
平均38.00个学生token，原参考21.53；蒸馏臂教师呈现token占20.98%，但当前batch1的CE逐样本
平均，不能将此比例视为loss权重。教师目标事实质量仍只做过此前抽样审核，此次无新增看图仲裁。

下一步优先复用CE/distill第100、250步检查点，四套adapter/projector/Q-Former/metadata齐全。
建议先用此前固定600题（Action433／Scenery167）做探索性曲线，共新增2400条生成及评分；
起点和final预测直接复用。该子集已有起点354/600、CE 349/600、distill 345/600，
约1个百分点差异需完整4320题复核，不能在600题小波动上宣称收益。
这是事后诊断，不能改写原fixed-final结论。先区分更新预算导致的退化与教师目标增量，随后
一次只改一个因素；不直接扩大训练、不用official选模。此次未启动任何GPU任务。

[诊断报告](artifacts/qwen_drive_20261022/diagnosis.md)、[统计与来源SHA](artifacts/qwen_drive_20261022/diagnosis.json)、
[可复现CPU分析脚本](artifacts/qwen_drive_20261022/diagnose_existing.py)。
