# 2026-10-18：反馈 OPD 的成功行为保留实验

## 当前状态与实验问题

**2026-10-19 07:05:47 UTC，bank、四臂训练和完整dev流程已正常结束。**
原服务`mpp-success-opd-4k-v1.service`于2026-10-18 08:36:34 UTC启动，使用GPU5／6。
bank含4,000条记录、2,549条eligible；四臂各完成500步，dev完成38,880条生成和18组评分。
按dev选模后的parent／CE／CE_keep／OPD／OPD_keep为59.86%／59.21%／59.33%／59.17%／59.58%。
OPD_keep比OPD高0.42个百分点，探索性95%区间为[-0.23, +1.11]个百分点；仍低于parent，
不能据此声称这轮已获得可靠提升。

**2026-10-19 07:23:31 UTC，另行启动官方评测服务`mpp-success-opd-official-4k-v1.service`，配置GPU5。**
**官方评测已于2026-10-19 09:33:27 UTC正常结束**，服务退出状态0，Result=success，MainPID=0。
parent及四臂入选模型共2,500条回答和五组评分全部完成；CE、CE_keep、OPD选final，
OPD_keep选checkpoint-000250，选择始终来自dev。`official_report.json`／`.md`均已生成。
检查了dev／official报告引用的92个来源文件hash，并独立按question_id复算逐题结果，均一致；
官方报告绑定的dev报告SHA256也匹配当前文件。日志中的TypedStorage弃用警告未导致任务失败。
`pipeline`本身只到dev，官方评测由上述独立服务补跑，本次完整实验现已结束。

真实两步训练和独立恢复已通过，三路最终权重逐张量差异均为0，峰值allocated显存17.38GiB；
这些是实现验证，不是准确率提升证据。

本轮只处理第一项改进：反馈 OPD 纠正当前错误时，是否会破坏父模型已经答对的行为，
以及用固定成功轨迹的分布约束能否减少这种退化。保留现有五帧224输入、原始QA、
完整视觉Q-Former、Projector和语言QKV LoRA，教师仍是原全量SFT父模型
`outputs/mpp_lingoqa_qformer_all_1epoch_v1`。本轮不引入新的强教师、分辨率、裁剪
或额外时间帧，便于把变化归因于成功行为保留项。

历史40k实验的parent／CE／OPD dev为60.00%／60.46%／60.23%；OPD相对CE
修复80、退化90。这是设计动机，不是本轮4k对照结果，也不能直接与新的4k模型
相减来宣称净收益。历史依据见[反馈权重对照](feedback_opd_weight_20261011.md)。

## 2026-10-19最终结果与分析

结论：成功保留项呈现减少父模型正确行为退化的信号，但会牺牲一部分纠错；
本轮所有入选训练模型都低于重新评估的SFT父模型，尚未实现最终正确率提升。
以下“正确／修复／退化”均按Lingo-Judge标签计数，不等于人工核验视频事实。

| 模型 | dev正确／4,320 | dev正确率 | official正确／500 | official正确率 | 入选更新步 |
|---|---:|---:|---:|---:|---:|
| SFT parent | 2,586 | 59.86% | 271 | 54.20% | — |
| CE | 2,558 | 59.21% | 263 | 52.60% | 500 |
| CE_keep | 2,563 | 59.33% | 262 | 52.40% | 500 |
| OPD | 2,556 | 59.17% | 264 | 52.80% | 500 |
| OPD_keep | 2,574 | 59.58% | 269 | 53.80% | 250 |

OPD_keep相对本轮OPD：dev +0.42个百分点、official +1.00个百分点；
相对parent仍分别为−0.28、−0.40个百分点。对照CE_keep时，OPD_keep的增量为
dev +0.25、official +1.40个百分点；相应95%区间分别为[-0.35,+0.97]、[-0.20,+3.20]。
因此尚不能认定OPD具有稳定的独立收益。

| 入选模型相对parent | dev修复 | dev退化 | official修复 | official退化 |
|---|---:|---:|---:|---:|
| CE | 165 | 193 | 12 | 20 |
| CE_keep | 141 | 164 | 8 | 17 |
| OPD | 177 | 207 | 12 | 19 |
| OPD_keep | 156 | 168 | 10 | 12 |

OPD_keep比OPD在dev少退化39题、也少修复21题，净增18题；official少退化7题、
也少修复2题，净增5题。逐条比较时，official的14次改善、9次退化中，
父模型原正确子集为10次改善／3次退化，原错误子集为4次改善／6次退化。
后者6次退化的KEEP答案均与parent原答案逐字相同，符合保留约束抑制部分纠错的解释。
这一观察来自已选检查点，不能排除训练步数差异的影响。

| 同步数dev比较：OPD_keep−OPD | 父模型原正确子集净增 | 原错误子集净增 | 总净增 | 总体95%区间（百分点） |
|---|---:|---:|---:|---:|
| 均250步 | +48题 | −17题 | +31题／+0.72个百分点 | [+0.07,+1.41] |
| 均500步 | +34题 | −36题 | −2题／−0.05个百分点 | [−0.60,+0.54] |

OPD_keep自身从250步的59.58%降到500步的59.12%（少20题）。到500步，保留收益仍在，
但纠错损失将其抵消。独立选模后的OPD_keep−OPD区间为dev[-0.23,+1.11]、
official[-1.20,+3.40]个百分点；交互项区间也跨零。250步的一项正区间是方向性证据，
不能用来概括整个训练过程或宣称稳定协同。
区间按视频segment分组bootstrap计算，dev为400个segment、official为100个；
本轮只有一个训练seed，检查点选择后的dev区间具有探索性质。

训练实现未发现未生效的迹象：四臂各8,000次QA访问／500次更新；keep两组各重放
5,098次（bank eligible 2,549×2），两种OPD分别有3,585／3,542次有效门控，占8,000次
rollout的44.81%／44.28%；教师代表参数差异为0，学生各可训练路径均有变化。
未见非有限梯度或总梯度为零的步骤。全程CE均值约0.418／QA，JSD约0.064／有效门控题，
按全部QA摊薄约0.0285；keep KL约0.00290／anchor，按全部QA摊薄约0.00185。
没有分损失梯度范数和夹角，不能仅比较这些标量大小就认定某项控制了更新方向。

第二轮训练CE继续下降，训练rollout的Judge通过率提高，而OPD_keep的dev下降；
结合CE对照也低于parent，提示小规模继续训练的参数偏移／泛化不足值得优先排查，
但本轮单seed、单配方尚不能证明因果。教师参考条件纠正的Judge通过率约96%，
这也不能直接证明沿学生错误前缀的分布蒸馏能转化为无参考推理能力。

建议的下一步（尚未启动）：固定4k、bank、五帧、损失权重和原500步调度，仅将各模块
学习率减半，配对比较CE_keep与OPD_keep；在100、250步保存检查点并先做dev判断。
原250步检查点可作为原学习率参照。这样优先检验更新幅度，同时保留OPD独立收益对照；
不因KL标量较小而直接增大keep权重，也不直接扩大数据或延长轮次。只有相对parent获得
开发集净收益、且修复大于退化后，再决定后续投入。official500已反复使用，不作为新的盲测，
也不用于下一轮选超参数。

原始报告：[开发集](../outputs/success_opd_eval_4k_v1/report.md)、
[官方集](../outputs/success_opd_eval_4k_v1/official_report.md)。

## 2026-10-19：事实监督审核与下一轮机制对照

用户授权补完关键事实审核，并分别验证错误分支 forward KL 与覆盖正确/错误的在线轨迹；
长训练、长评测放到后台，提供统一监视命令。原4k结果及所有旧模型保留。

事实审核复用旧诊断300条中的32个定向训练案例：20条历史Judge错误、12条Judge阳性控制。
按文本中的事实对比选取，检查全部五帧并补看小目标/信号灯的放大图，先保存事实注释再进行
新概率计算。审核由Codex完成，不是独立人工金标，也不是总体教师错误率的随机抽样。
17个案例的所选事实有较明确图像支持，13个存在时刻/对象/驾驶意图歧义，2个参考存疑。
索引151没有明确事实对比，保留在审核集中但不强行构造错误词，因此合计31个候选对比。
参考存疑与不确定案例仍保留原标签/原Judge，不进入“视觉支持”汇总。

17个视觉支持对比中，参考条件相对普通提示均改善正确候选对错误候选的log-odds，16/17
最终更偏向正确候选；当前重新生成的教师通过门控后为15/16。历史错误子集为7/8偏向
正确候选，正确控制为9/9。约束在原学生token前缀上，候选多token时沿候选自身的后续
token计算联合概率，并注明长度影响，不用原错误后缀给正确候选打分。

主要反例为索引41：前文已错误地说向左保持车道，随后询问原句中cycle lane的侧别，
图像/参考为left，教师仍偏向right。正确/错误log-odds由−9.125改善到−4.500，尚未反转。
这说明反馈改善与真正翻转决策不同；本轮定向审核整体上削弱了“教师普遍教错”的解释，
不能证明所有学生前缀可靠，也不证明蒸馏必然改善开发集。

审核产物：`outputs/opd_fact_audit_20261019/{manifest.json,annotations.json,scores.jsonl,report.json,report.html,decision.json}`。
脚本：`scripts/audit_opd_facts.py`、`scripts/render_opd_fact_audit.py`。
新计算未使用开发/官方参考，未更新模型权重；审核案例与heldout的question、segment、frame
交集均为0，保存原轨迹、模型身份、逐帧及注释的SHA256。

下一轮采用独立目录 `outputs/opd_mechanism_4k_v1` / `outputs/opd_mechanism_eval_4k_v1`：

| 组 | 原参考CE | 错误轨迹蒸馏 | 正确的当前在线轨迹 | 固定成功保留 |
|---|---:|---|---|---:|
| ce | 1 | 无 | 无 | 0 |
| jsd_errors | 1 | JSD，系数1 | 无 | 0 |
| kl_errors | 1 | forward KL，系数1 | 无 | 0 |
| jsd_all | 1 | JSD，系数1 | JSD，系数1 | 0 |

所有组重新从同一父模型开始，使用原4k、五帧、seed44、原学习率、batch16、单epoch/
250步完整cosine调度，在100步和final各评4320题，并重新评parent，共38880条回答。
这是新的250步完整调度，不冒充旧500步调度的前250步；所有对照组重新运行。
all分支仍要求学生生成完整、教师独立回答完整且通过Judge；保留原学生token和CE，
不把教师回答当成训练轨迹。所有损失按同一真实QA批次分母归一化。
forward KL不额外加入pointwise clipping，保持这次仅改变散度；它不是相关论文的完整复现。
相同系数不等于相同梯度大小；all范围扩大也增加有效蒸馏次数，不宣称监督量或计算量相等。
按各自dev选模且同分选早，另报告固定100/250步的配对比较、修复/退化和segment bootstrap。
单seed、选模后dev区间均属探索性；本轮不会自动扩量或用official500选择超参数。

入口：`bash scripts/run_opd_mechanism.sh pipeline`；要求明确两张GPU，代码/输入指纹、事实审核
结论和真实两步训练/断点恢复验证齐全，输出目录带排他锁。长任务的实时启动状态以
`outputs/opd_mechanism_4k_v1/workflow.json`和systemd为准，不从文档推断进程存活。

机制验证已通过40项相关测试（含宿主环境上的双进程梯度归约测试）。真实GPU小样本使用
8条原QA、forward KL/all、2次更新，7条通过教师门控，其中正确轨迹4条、错误轨迹3条；
教师三个模块的代表参数均无变化。独立从第1步恢复后，最终LoRA、Projector、Q-Former
逐张量最大绝对差异均为0，峰值allocated显存17.38GiB。验证文件为
`outputs/opd_mechanism_smoke8_20261019/smoke_validation.json`；这些只验证实现与恢复，
不是开发集提升证据。

2026-10-19 10:36 UTC 已启动后台服务 `mpp-opd-mechanism-4k-v1.service`，GPU 5/6，
控制进程PID 3008547；启动时已核实systemd为active/running并进入CE组输入校验/加载阶段。
四组依次训练，之后自动执行完整dev生成、Judge评分和配对报告；启动时尚无本轮dev结果。
阶段日志每30秒打印子进程与进度，训练详单见 `outputs/opd_mechanism_4k_v1/logs/`。

**2026-10-19 11:28:56 UTC进度快照**：CE已完成250/250，summary于10:56:38 UTC落盘；
JSD-errors为54/250，累计蒸馏正确0／错误390条；KL-errors和JSD-all尚未启动。
四组训练后才开始dev，快照时无新预测或报告，尚无本轮准确率结论。此次独立核对systemd
为active/running，MainPID3008547；文件计数本身不证明进程存活，也不等于纠错题数。
已同步[项目时间线第72–74节](project_timeline.md)；原始小型JSON及来源SHA256见
[带时间戳的证据快照](artifacts/opd_20261019/snapshot_20261019T112856Z.json)。
这些进度只代表上述时刻，后续以监视入口为准。

```bash
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa
watch -n 15 'bash scripts/run_opd_mechanism.sh status'
journalctl --user -fu mpp-opd-mechanism-4k-v1.service -o cat
```

此前“减半学习率”的建议保留为后续泛化诊断候选；当前按用户确认优先验证损失和在线轨迹
覆盖，并未同时改变学习率或学生SFT起点。

## 2026-10-20：机制对照最终结果（06:30:55 UTC核验）

本轮于04:49:33 UTC完成全部训练及38880条dev生成／评分，18/18评分汇总和最终报告齐全。
本次核验72份报告来源hash、8份实验输入、12份代码及四组checkpoint身份；重算逐题指标、
配对修复/退化和视频bootstrap均与报告一致。原报告和模型保持不变，没有新增official评测。

| 方案 | 100步Judge | 250步Judge | 250步正确／4320 | 相对同250步CE |
|---|---:|---:|---:|---:|
| 未续训parent | — | 59.86%（基线） | 2586 | — |
| CE | 59.33% | 60.05% | 2594 | — |
| 错误轨迹JSD | 59.38% | 60.16% | 2599 | +5题／+0.116个百分点 |
| 错误轨迹forward KL | 57.99% | 57.92% | 2502 | −92题／−2.130个百分点 |
| 正确＋错误轨迹JSD | 59.10% | 59.40% | 2566 | −28题／−0.648个百分点 |

**错误JSD的优势很小，尚不能确认为稳定提升。** 对CE修复69题、退化64题，95% segment
bootstrap区间[−0.415,+0.671]个百分点，包含零。对parent净增加13题，但CE续训本身已增加8题。
单seed／4k试验不能把这5题差异写成确定的OPD泛化收益。

KL对CE修复208／退化300，区间[−3.120,−1.051]个百分点；all修复74／退化102，区间
[−1.286,−0.069]。all相对错误JSD为37修复／70退化，净−33题，区间[−1.252,−0.257]。
区间以400个视频segment为单位重采样1000次、seed42，保留题目在视频内的相关性；它不覆盖
训练随机种子差异，也没有校正反复开发和多重比较。KL按dev选100步，其他选250步，选模后
结论仍相同；机制判断以上表同250步为主。

| 250步模型 | Action正确／3120 | Action Judge | Scenery正确／1200 | Scenery Judge |
|---|---:|---:|---:|---:|
| parent，未续训 | 1814 | 58.14% | 772 | 64.33% |
| CE | 1827 | 58.56% | 767 | 63.92% |
| 错误JSD | 1826 | 58.53% | 773 | 64.42% |
| 错误KL | 1737 | 55.67% | 765 | 63.75% |
| 全轨迹JSD | 1799 | 57.66% | 767 | 63.92% |

KL的92题损失中90题来自Action，all的28题损失全部来自Action；错误JSD净增5题是Action−1、
Scenery+6。因此本轮没有证据证明新的蒸馏方式改善了核心行为解释能力。

训练统计帮助限定归因，而不是证明因果：

| 分支 | 蒸馏正确／错误轨迹 | 蒸馏token | CE均值／全部QA | 蒸馏均值／全部QA |
|---|---:|---:|---:|---:|
| CE | 0／0 | 0 | 0.42918 | 0 |
| 错误JSD | 0／1824 | 38667 | 0.42979 | 0.02922 |
| 错误KL | 0／1976 | 42689 | 0.44700 | 0.14569 |
| 全轨迹JSD | 2101／1790 | 77412 | 0.43008 | 0.05269 |

教师始终冻结，学生轨迹截断、上下文跳过均为0。forward KL方向是teacher→student，回答token
平均后与CE使用同一QA分母。系数虽然都为1，但KL累计蒸馏标量约为错误JSD的4.99倍，all为
1.80倍；all轨迹数2.13倍、token数2.00倍。标量大小不是梯度大小，且在线轨迹随模型变化，
因此本轮只否定这些具体配方，不能否定KL或正确轨迹蒸馏的一般价值。

日志记录的合成梯度裁剪前范数均值为CE4.6660、错误JSD4.6562、错误KL5.2992、all4.7136；
每组250步均触发norm1全局梯度裁剪。这不是单独蒸馏梯度的测量。本轮缺少KL逐位置/逐词表项
pointwise clipping，但并非没有全局梯度裁剪。应先测量固定前缀下独立损失梯度，再决定系数，
不能简单按loss的4.99倍倒数给出已验证的最佳权重。

KL相对CE改动2320条回答，错误JSD697条、all856条；平均空格分词数为CE16.90、错误JSD16.97、
KL17.29、all16.99，所有输出非空。KL的F1从CE0.6500降至0.6166，Exact从10.625%降至7.731%。
它表现为更广泛的答法/内容漂移，而不是简单输出变长。含potential的回答CE为109条、KL350条；
这是风格线索，不证明退化原因。例如question_id `470c0ef5a233f64374e4c32750004425` 的参考为
减速以安全右转，CE回答减速，KL改为停车并添加交通法规理由；本次只对照文本/参考，未重新看视频。

32例教师事实审核说明选定事实的参考条件反馈多数改善候选概率，不等于每个token都可靠；
仍有错误前缀下改善但未翻转事实判断的案例。因此不能把审核通过等同于学生泛化必然提高。
本轮parent是406212题／25389更新的完整SFT，入口固定为`mpp_lingoqa_qformer_all_1epoch_v1`，
不是历史最佳40k继续CE模型。本轮不能回答从后者继续OPD能否提升，也不证明SFT能力到顶。

当前建议保留错误JSD作为机制基线，暂不将这两种新配置直接扩量长训。若继续，优先比较
匹配蒸馏强度的KL、正确轨迹独立权重／固定总蒸馏预算，并对事实相关token与非关键信息的
监督做小规模诊断；这些都是后续假设，本次没有启动新任务。

结果：[报告](../outputs/opd_mechanism_eval_4k_v1/report.json)、
[时间线第75节](project_timeline.md)、
[带时间戳的原始证据](artifacts/opd_20261020/snapshot_20261020T063055Z.json)。

## 目标函数与两种教师输入

对每条训练QA，保留原答案及EOS的CE。`opd`分支继续在线采样当前学生，沿用
原 `reference_only` 反馈：学生回答完整且Judge判错、教师纠正完整且Judge判对，
才在**这次学生生成的原始token前缀**上计算token平均JSD。

新增成功项使用训练前冻结的父模型成功轨迹。令 `x_i` 为原问题和五帧，`a_i` 为
原答案，`y_i` 为当前学生在线采样，`z_i` 为bank保存的父模型greedy token序列，
`g_i` 为错误纠正门控，`b_i` 为该题存在有效bank成功轨迹：

\[
L_i = L_{CE}(a_i\mid x_i)
 + \lambda g_i\frac{1}{|y_i|}\sum_t
   JSD\big(p_\theta(\cdot\mid x_i,y_{i,<t}),
            p_0(\cdot\mid x_i,reference_i,y_{i,<t})\big)
 + \beta b_i\frac{1}{|z_i|}\sum_t
   KL\big(p_0(\cdot\mid x_i,z_{i,<t})\;\|\;
          p_\theta(\cdot\mid x_i,z_{i,<t})\big).
\]

`p_0`固定为原SFT父模型；错误JSD的教师读取训练参考答案，成功KL的师生都只读
普通问题和五帧。成功KL方向明确为**父模型→学生的forward KL**，两侧沿同一份
原成功token IDs预测，包含EOS。bank不保存logits；训练时冻结父模型重新计算
目标分布，不把解码文本重新分词，也不将成功回答替换成新的CE标签。

这是在原错误OPD上加入固定轨迹约束：成功项本身是离线保留项，不应称为新的
在线成功采样。原parent成功的题即使当前学生已答错，仍使用同一份bank轨迹；
不会根据本轮开发结果更新bank。每项先取本题token均值，再以同一真实QA批次
分母归一化；没有按eligible题数重新放大JSD或成功KL。

父模型与学生刚初始化时，普通prompt下成功KL应接近零。训练后它用于限制
偏移；初始损失接近零不能单独解释为没有接入或没有作用。教师adapter切换和
学生反向分段顺序保持一致，避免梯度检查点在错误的adapter下重算。

## 数据与bank契约

从既有 `datasets/processed/lingoqa_feedback_opd_40k_v2/train.jsonl` 确定性抽取
4,000条原QA，seed44、按Action／Scenery比例分配，再按segment轮转。实际结果：

| 项目 | 数量／身份 |
|---|---|
| 训练QA／segment | 4,000／4,000，每segment一题 |
| Action／Scenery | 2,580／1,420 |
| 新训练集 | `datasets/processed/lingoqa_success_opd_4k_v1/train.jsonl` |
| train SHA256 | `a78c6f1189e614f3ec440698abe6136e276cfb34c637a595e2d34b1ce9aff768` |
| 原40k train SHA256 | `d885dc2f2c59e2a8ad4c01a8240e471ed3d04e4602e887dd7ef680febb770712` |
| 开发集 | 原Action3120＋Scenery1200，合计4320 |
| 官方集 | 原500题，已反复使用 |

准备阶段验证question／segment／frame与dev/eval隔离，完整保留问题、答案、五帧
路径和次序；heldout文件逐字节复制，全部4k原EVA缓存存在。manifest记录原40k
路径和SHA，bank加载还会验证每条记录完整等于原40k同qid的QA。额外200条probe
只是这4k内的子集，不增加训练数据，也不使用dev题目构造bank。

bank仅对这4k训练QA，用原SFT父模型在**无参考答案提示**下greedy生成一次。
记录每题原QA、完整生成token IDs、EOS完成状态、解码回答和Judge原始logit。
所有题都保留记录；仅完整、非空且官方Judge通过的回答为eligible。Judge阳性是
文本代理判断，不等于独立人工确认的视觉事实，不能把bank规模称为真实正确题数。

生成支持单GPU和独立多GPU分片。每条完整JSONL记录落盘后才记为完成，恢复时
跳过已验证记录；仅未完成文件的撕裂末行可截去重做，重复qid或已完成文件变化
直接拒绝。全部分片完成后merge为：

```text
outputs/success_opd_bank_4k_v1/
  shard-00000-of-00002/{run_config.json,records.jsonl,manifest.json,progress.json}
  shard-00001-of-00002/...
  records.jsonl
  bank_manifest.json
```

完成manifest绑定模型三路权重、冻结基座、数据和heldout SHA、tokenizer实现／
词表／special IDs、生成和Judge协议、逐条记录hash及整体文件hash。Qwen普通词表
不包含special token，token合法范围按实际LM输出词表151936校验，EOS为151645。
只有完成且全量覆盖4k原QA的bank可训练；不会接受部分成功样本文件冒充完整bank。

原错误teacher probe只作为已验证机械条件复用。训练端严格核对同parent、输入、
精度、采样、教师模式和评测协议，并证明4k是原40k的原样子集，才允许train数据
SHA变化；不以这个复用报告声称新成功项已经验证有效，也不重新生成历史probe。

## 四臂与固定预算

| 分支目录 | 原答案CE | 当前错误门控JSD | 固定成功forward KL |
|---|---:|---:|---:|
| `ce` | 1 | 0 | 0 |
| `ce_keep` | 1 | 0 | β＝1 |
| `opd` | 1 | λ＝1 | 0 |
| `opd_keep` | 1 | λ＝1 | β＝1 |

四组都从相同原parent重新初始化，使用同一4k池和bank身份、seed44、2epoch、
有效batch16，**每组8,000次QA访问、500次更新**，保存第250步和final。语言LoRA／
Projector LR为2.5e-6，Q-Former LR为5e-6；AdamW、weight decay .01、warmup3%、
cosine、FP32可训练权重／BF16冻结基座、梯度检查点及answer-only logits保持一致。
`ce_keep`有冻结教师但不进行错误在线rollout；两个OPD分支各应有8,000次rollout。
两个keep分支各应重放 `2×bank eligible` 题，报告核对这些实际计数。

支持2／3／4卡，但四臂和断点恢复保持同卡数。**本次使用GPU5／6两卡，每卡
累积8，8＋8条真实QA组成有效batch16。** 三卡时每更新六个同步槽，实际QA为
6＋5＋5，空槽不前向、不采样、不重复数据，有效batch仍为16。各组匹配QA和更新
预算；keep与在线rollout增加计算，**不宣称四组训练时间或GPU成本相同**。

## 评测与判定

主问题是 `OPD_keep − CE_keep`，即都有成功约束时OPD是否有独立价值；同时比较
`OPD_keep − OPD` 和 `CE_keep − CE`。交互项为：

\[
I=(OPD_{keep}-CE_{keep})-(OPD-CE).
\]

在250步和500步分别报告同step的四臂配对结果；另按完整dev Judge为各臂选模型，
平分选较早step。给出修复／退化、按segment分组bootstrap区间，并用重新评估的
parent划分原正确／原错误题，检查收益是否来自减少原正确题的退化、是否损害纠错。
开发选模后的置信区间属于探索性结果，需要结合固定step比较；训练KL变小本身
不能证明任务提升。独立收益和交互项都应结合区间与真实回答判断。

`eval_dev`默认重新评估parent，严格校验同一权重和输入／生成身份；不会默认读
旧预测冒充本轮parent。parent加四臂两个checkpoint，共**9个模型×4320＝38,880次
自由回答，18个Action／Scenery生成评分任务**。因此训练完成后，开发评测仍可能
持续很久。两种任务均greedy96、batch1、max_txt_len1536、原官方Judge。新runner
显式记录冻结基座身份并核验原完整五帧缓存；缺少这些身份字段的旧评测产物不能
直接作为本轮严格匹配的parent评测。

`pipeline`执行prepare→bank→四臂训练→完整dev；official另行 `eval_selected`。
官方评估沿用已冻结的dev选择，parent＋四个入选模型共2,500次回答；报告不会用
官方500重新选择checkpoint。该500题已反复使用，不能称新盲测。

## 后台运行与监视

先查看实际占用，确认计划使用的GPU5／6有足够空闲显存；卡号只是计划，
不表示执行时空闲。编写本记录时GPU0–3用于其他训练、GPU4用于Q-Former评测、
GPU7也有其他进程，正式pilot计划使用5／6。runner的bank／train／pipeline／评测
动作均要求显式 `CUDA_VISIBLE_DEVICES`；status、prepare和报告不需要GPU参数。

```bash
nvidia-smi
cd /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa

systemd-run --user --unit=mpp-success-opd-4k-v1 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=5,6 \
  --setenv=NPROC_PER_NODE=2 \
  --setenv=MASTER_PORT=29743 \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  --setenv=TOKENIZERS_PARALLELISM=false \
  /usr/bin/bash scripts/run_success_opd.sh pipeline
```

该命令已于2026-10-18 08:36:34 UTC执行，并于次日07:05:47 UTC完成；不必重复启动。
四臂顺序运行，同一输出目录的runner有锁，避免并行启动两条训练
或评测流程。

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa/scripts/run_success_opd.sh status'
journalctl --user -fu mpp-success-opd-4k-v1.service -o cat
systemctl --user status mpp-success-opd-4k-v1.service --no-pager
```

`status`显示bank分片、每臂更新／总预算、保留KL、开发生成／评分进度及产物校验。
磁盘中存在`running`进度不证明进程还活着，结合systemd与journal判断。bank日志在
`outputs/success_opd_bank_4k_v1/shard-0.log`等文件；训练状态在各臂的`progress.json`，
异常细节在`failure_rank_*.json`和journal。

默认目录：

| 产物 | 目录 |
|---|---|
| 数据 | `datasets/processed/lingoqa_success_opd_4k_v1` |
| bank | `outputs/success_opd_bank_4k_v1` |
| 四臂训练 | `outputs/success_opd_4k_v1/{ce,ce_keep,opd,opd_keep}` |
| 评测及报告 | `outputs/success_opd_eval_4k_v1` |

需要隔离新的实验时，可显式设置 `SUCCESS_OPD_DATA_DIR`、`SUCCESS_OPD_BANK_DIR`、
`SUCCESS_OPD_TRAIN_DIR`、`SUCCESS_OPD_EVAL_DIR`。监视和恢复命令必须传入同一组
覆盖变量，不要让状态入口误读另一个默认目录。

## 中断恢复与后续评测

bank中断后，在确认原任务已退出后，以**相同分片数**重跑bank即可；已完成记录
会核验后跳过，未完成题继续生成，最后merge。下例保持两卡分片：

```bash
CUDA_VISIBLE_DEVICES=5,6 bash scripts/run_success_opd.sh bank
```

训练中断后，`resume ARM`读取已验证的第250步完整训练状态，恢复优化器、调度、
数据游标和RNG。ARM为 `ce`、`ce_keep`、`opd`、`opd_keep` 之一。以下示例只恢复
`opd_keep`，实际使用时替换为失败分支，并保持原GPU数量：

```bash
systemd-run --user --unit=mpp-success-opd-resume-opd-keep-v1 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=5,6 --setenv=NPROC_PER_NODE=2 \
  --setenv=MASTER_PORT=29743 --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  /usr/bin/bash scripts/run_success_opd.sh resume opd_keep
```

恢复完成后，可再运行pipeline：已完成分支会经过完整预算／身份核验后跳过，
后续分支和dev继续执行。若第250步尚未保存，不能伪装为断点恢复；保留失败产物，
使用新的训练输出目录重新开展对应实验。不要删除失败目录来绕过身份检查。

如果只是dev生成／评分中断，直接重跑 `eval_dev`，保留已完成的同身份输出：

```bash
CUDA_VISIBLE_DEVICES=5 bash scripts/run_success_opd.sh eval_dev
```

完整dev报告落盘后，official单独后台运行；本步不包含在首次pipeline：

```bash
systemd-run --user --unit=mpp-success-opd-official-4k-v1 --collect \
  --property=WorkingDirectory=/home/zjr/Tracker/internship_project_lab/mpp_driving_video_qa \
  --setenv=CUDA_VISIBLE_DEVICES=5 --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  /usr/bin/bash scripts/run_success_opd.sh eval_selected
```

该官方评测服务已于2026-10-19 07:23:31 UTC启动，并于09:33:27 UTC正常结束；
无需再次执行启动命令。查看完整日志：

```bash
journalctl --user -u mpp-success-opd-official-4k-v1.service -o cat
```

最终查看 `outputs/success_opd_eval_4k_v1/report.json`／`report.md` 和
`official_report.json`／`official_report.md`。在四臂、两个dev检查点和身份校验
完成之前，不能只挑某条训练曲线或修复例子报告本轮有效。

## 实现与验证记录

- `mpp_video_qa/success_bank.py`：bank数据／模型／tokenizer／EOS／完整性校验及原token映射。
- `scripts/prepare_success_opd.py`、`build_success_bank.py`：原4k数据、分片生成、恢复和merge。
- `scripts/train_feedback_opd.py`：普通prompt父模型forward KL、三项统一QA分母、统计和恢复身份。
- `scripts/run_success_opd.sh`、`report_success_opd.py`：四臂预算、后台入口、开发选模、配对和交互报告。

成功bank模块12项CPU测试通过，覆盖heldout泄漏、错模型／数据／tokenizer、篡改
token、未完成bank、EOS门控、分片合并、恢复去重，以及生成token不经文本回编码。
真实4k准备和本地Qwen tokenizer身份预检通过，未据此推断GPU性能或准确率。
新增测试共32项通过：bank 12项、训练核心7项、启动器／报告13项。
相关历史回归47项也全部通过：GPU layout 12项、OPD speed 8项、OPD v2 9项、
staged full SFT 9项、question Q-Former 9项。两／三进程Gloo检查在沙箱外运行，
因为沙箱阻断本机通信。

真实短测入口为`scripts/smoke_success_opd.sh`：8条原训练QA、两次更新，第一步
保存checkpoint，再独立恢复第二步，检查成功保留、错误OPD、冻结teacher及最终权重。
短测bank生成8条完整Judge-positive原token轨迹；真实训练累计8次rollout、3条
eligible错误轨迹／57个蒸馏token，8条成功轨迹／142个保留token。三路学生代表参数
都有更新，教师代表参数变化为0；第二步成功KL均值0.0007103（初始第一步为0）。
峰值allocated 17.380GiB、reserved 17.742GiB。独立恢复第一步检查点后，最终语言
LoRA、projector和完整Q-Former的所有张量与不中断运行**完全一致（max abs diff=0）**。
验证产物为`outputs/success_opd_smoke8_train_20261018/smoke_validation.json`，小型
原始证据、测试计数和最终源码指纹归档于[artifacts/success_opd_20261018](artifacts/success_opd_20261018/)。

短测首轮trainer成功保存checkpoint和summary；当时给仍在执行的包装shell补充
单卡参数检查，旧shell退出时重新读取文件产生EOF。最终脚本已通过`bash -n`，
固定脚本的独立resume与逐张量检查均正常退出；正式运行脚本启动后保持不变。

```bash
/home/zjr/anaconda3/envs/mpp_llava/bin/python -m unittest discover \
  -s tests -p 'test_success_bank.py' -v
/home/zjr/anaconda3/envs/mpp_llava/bin/python -m unittest discover \
  -s tests -p 'test_success_opd_report.py' -v
bash -n scripts/run_success_opd.sh
```
