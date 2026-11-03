# 上游与依赖说明

## MPP-LLaVA / MPP-Qwen-Next

- 来源：<https://github.com/Coobiw/MPP-LLaVA>
- 固定版本：`baf617b0f5217442ba20721be9aed710cdf999e2`
- 获取入口：`scripts/setup_upstream.sh`
- 本地适配差异：`patches/mpp_llava_baf617b_local.patch`

上游 README 的 License 部分注明其多处代码基于 LAVIS，并引用 BSD 3-Clause 许可，同时单独引用 Qwen 模型许可。各文件自带的版权和许可声明继续适用。上游源码由安装脚本获取，模型权重单独准备。

## 数据、评估与可选教师

- LingoQA 数据与 Lingo-Judge：<https://github.com/wayveai/LingoQA>
- Qwen 语言模型：<https://github.com/QwenLM/Qwen>
- Qwen-Drive 可选教师：<https://github.com/QwenLM/Qwen-Drive-1.0>
- LAVIS / BLIP-2：<https://github.com/salesforce/LAVIS>
- DeepSpeed：<https://github.com/microsoft/DeepSpeed>
- Transformers、PEFT、PyTorch 等依赖使用各自发行包所附许可。

数据集和模型的使用、再分发及用途要求以各自上游许可为准。
