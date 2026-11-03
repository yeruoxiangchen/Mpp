# 环境与运行

本仓库包含驾驶视频问答的应用层源码。MPP-LLaVA 源码通过固定版本安装，模型权重与 LingoQA 数据按各自上游说明准备。

## 存储目录

在仓库根目录执行：

```bash
export MPP_DATA_ROOT=/data/$USER/mpp
source scripts/env.sh
bash scripts/setup_storage.sh
```

`datasets/`、`checkpoints/`、`outputs/`、`logs/` 和 `.venvs/` 会建立到数据目录的兼容链接。临时目录、模型缓存和编译缓存也由环境变量指向数据目录。已有目录或链接出现冲突时，脚本会报出具体路径供手动配置。

## MPP 运行环境

原训练环境使用 Python 3.8、PyTorch 2.0.1、Transformers 4.39.2 和 PEFT 0.13.2。可按下列方式建立独立环境：

```bash
conda create -p "$MPP_DATA_ROOT/env/mpp" python=3.8 pip=24.0 setuptools=69 wheel -y
conda activate "$MPP_DATA_ROOT/env/mpp"
export PYTHON_BIN="$CONDA_PREFIX/bin/python"
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r configs/requirements-runtime.txt
python -m pip install -e '.[data,test]'
bash scripts/setup_upstream.sh
```

上游固定在 `baf617b0f5217442ba20721be9aed710cdf999e2`，安装时应用 `patches/mpp_llava_baf617b_local.patch`，保留检查点加载和生成路径的适配。`MPP_UPSTREAM_ROOT` 可指定其他源码安装位置，安装脚本要求其版本与补丁一致。

可选的 Qwen-Drive 教师和区域检测器使用独立环境，相关入口为 `scripts/setup_qwen_drive_teacher.sh` 和 `scripts/install_region_detector.sh`。Qwen-Drive 环境安装时通过 `SETUP_PYTHON` 指向 Python 3.11；外部 VLM 教师通过 `TEACHER_PYTHON` 指定相应解释器。

## 模型与数据准备

按 [MPP-LLaVA 权重说明](https://github.com/Coobiw/MPP-LLaVA/blob/master/WEIGHT.md) 准备 EVA、BLIP-2、Qwen 和 MPP 模型资产。应用层默认读取以下结构：

```text
checkpoints/
├── ckpt/
│   ├── Qwen7B-chat/
│   ├── blip2/blip2_pretrained_flant5xxl.pth
│   └── bert-base-uncased/                 # 训练视觉桥接时使用
└── mpp-qwen-next-sft/
    └── unfreeze_llm_model.pth
```

MPP 检查点也支持 `mpp-qwen-next-sft/sharded/model.safetensors.index.json` 及其分片。需要恢复特定视觉桥接或 LoRA 时，在对应命令中传入产物路径；训练生成的适配器独立保存在输出目录。

下载 [LingoQA](https://github.com/wayveai/LingoQA) 数据并解压图像后，使用实际路径生成 JSONL：

```bash
export LINGO_RAW_ROOT="$MPP_DATA_ROOT/datasets/lingoqa"
export LINGO_PROCESSED_DIR="$MPP_DATA_ROOT/datasets/processed/lingoqa_v1"
python scripts/prepare_lingoqa.py \
  --train-input "action=$LINGO_RAW_ROOT/action/train.parquet" \
  --eval-input "$LINGO_RAW_ROOT/evaluation/val.parquet" \
  --frames-root-map "action=$LINGO_RAW_ROOT/action/extracted" \
  --frames-root-map "evaluation=$LINGO_RAW_ROOT/evaluation/extracted" \
  --output-dir "$LINGO_PROCESSED_DIR" --check-files
```

每条处理后记录包含 `question_id`、`segment_id`、`source_subset`、`question`、`answers` 和按时间排列的 `frames` 路径。切分与去重按视频片段执行，数据组成以生成的 `manifest.json` 为准。

## 基础流程

基础入口 `run_pipeline.sh` 默认以四个进程运行特征缓存和 LoRA SFT；应先根据可用设备设置 `CUDA_VISIBLE_DEVICES`。需要改变进程数时，可直接使用底层 Python 入口和 `torchrun`。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LINGO_CACHE_DIR="$MPP_DATA_ROOT/datasets/cache/lingoqa_5f_mpp"
export SFT_OUTPUT_DIR="$MPP_DATA_ROOT/outputs/lingoqa_lora"
bash scripts/run_pipeline.sh cache
bash scripts/run_pipeline.sh train
CUDA_VISIBLE_DEVICES=0 bash scripts/run_pipeline.sh eval_base
CUDA_VISIBLE_DEVICES=0 bash scripts/run_pipeline.sh eval_sft
CUDA_VISIBLE_DEVICES=0 bash scripts/run_pipeline.sh score_base
CUDA_VISIBLE_DEVICES=0 bash scripts/run_pipeline.sh score_sft
```

`score_*` 使用 Lingo-Judge。它依据问题、参考答案和生成答案进行文本判断，视觉事实仍需要回到原视频核查。

其他训练入口包括：

- `run_full_qformer_sft.sh`：视觉桥接与语言适配器联合训练。
- `run_question_qformer_full.sh`：问题条件化 Q-Former 训练。
- `run_feedback_opd.sh` / `train_feedback_opd.py`：反馈门控与在线蒸馏。
- `run_success_opd.sh`：教师门控、监督保留及对应对照。
- `run_preference.sh`：离线偏好训练。

这些研究入口依赖各自的数据清单、初始化适配器、视觉桥接权重和筛选记录。通过脚本参数或相应环境变量配置输入，具体前置步骤见同目录的过程文档。

## 单视频问答

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/ask_video.py \
  --video /absolute/path/to/driving_clip.mp4 \
  --question 'What is the vehicle doing?' \
  --upstream-root "$MPP_UPSTREAM_ROOT" \
  --checkpoint-root "$MPP_CHECKPOINT_ROOT"
```

加入 `--adapter /path/to/final_adapter` 可加载语言 LoRA；视觉投影与 Q-Former 产物按 `--help` 中对应参数传入。

## 本地测试

```bash
python -m pytest tests/test_core.py tests/test_judge_protocol.py tests/test_model_loading.py
```

完整测试还包含真实上游模块接口检查和独立环境测试；使用相应环境并安装上游后，再选择对应测试文件运行。
