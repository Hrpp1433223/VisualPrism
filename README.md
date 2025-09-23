# VisualPrism

## Introduction	🌈
**VisualPrism** is an enhancement over [LLaVA](https://github.com/haotian-liu/LLaVA), introducing a **guide-then-compress** strategy to significantly reduce visual tokens while preserving rich semantics. Inspired by the **fovea–periphery coordination** in human vision, VisualPrism injects explicit visual priors to improve efficiency and accuracy in vision-language understanding.

- 🎯 **Prior-Guided Compression**: Integrates frequency-domain priors (via Discrete Wavelet Transform) before token merging.
- 🔀 **Triple-Path Query Projection (TPQP)**: Decomposes visual queries into global, local, and identity-based components.
- 🧩 **Hierarchical Key-Value Encoding (HKVE)**: Structures visual information across scales for better alignment and retrieval.
- 🔌 **Modular & Compatible**: Drop-in replacement for LLaVA's projector, compatible with most MLLM pipelines.
![VisualPrism Architecture](assets/VisualPrism_paper.drawio.png)

## Install 🛠️
1. Clone this repository and navigate to VisualPrism folder
```
git clone https://github.com/Hrpp1433223/VisualPrism.git
cd VisualPrism
```
2. Install packages
```
conda create -n visualprism python=3.11 -y
conda activate visualprism
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```
3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Train 🧠

#### Training 
- Stage1: Image-Text Alignment Pre-training
```shell
bash scripts/v1_5/pretrain.sh
```
- Stage2: Visual Instruction Tuning
```shell
bash scripts/v1_5/finetune.sh
```
Note: 
- Using `--scale_factor` to control compression ratio, support [2,3,4]
- You can use regularization term by`--use_sinkhorn_lsd` and turn off by `--no_use_sinkhorn_lsd` for more details please check the [original paper](https://github.com/2018cx/SinKD)
## Evaluation 📊

#### Evaluation process 
VisualPrism follows LLaVA evaluation steps.
See [Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

#### Evaluation results 
| Model           | #Tokens | GQA  | VizWiz | VQAv2 | MM-Vet | Avg. Score |
|----------------|---------|------|--------|--------|--------|-------------|
|  VisualPrism -7b  | **144** | 59.9 | 53.3   | 76.3   | 34.5   | 62.4        |
|  VisualPrism-7b   | **64**  | 58.3 | 52.0   | 74.1   | 31.7   | 60.6        |
| VisualPrism-Qwen-7b  | **144**  | - | -   | -   | -   | -       |
| VisualPrism-Qwen-13b  | **144**  | - | -   | -   | -   | -        |
#### Ablation results
##### Ablation For Modules Proposed
| TPQP | HKVE | VQAv2  | GQA |
|---------|---------|-------|-------|
|❌|❌|61.9|56.7|
|✔️|❌|62.6|57.5|
|❌️|✔️|62.4|57.5|
|✔️|✔️|62.7|57.6|
##### Ablation For 3 Paths in TPQP
| Wavelet | Multi-Sacal Conv | Identity | VQAv2  | GQA |
|---------|---------|-------|-------|-------|
|❌|❌|✔️|61.9|56.7|
|✔️|❌|✔️|62.5|57.2|
|✔️|✔️|❌|62.6|57.4|
|✔️|✔️|✔️|62.6|57.5|

## Model Zoo

| Model              |  Max Res.   |  Compre. Ratio  |  Token Num.  |  Download |
|--------------------|:-----------:|:---------------:|:------------:|:----------------:|:--------------------------------------------------------------------------------------------------:|---------------------------------------------------------------------------------------|
| VisualPrism-7b     |   336x336   |       1/4       |     144         | [checkpoints](https://huggingface.co/)  |
| VisualPrism-13b     |   336x336   |       1/4       |     144      | [checkpoints](-) |
| VisualPrism-7b  |  336x336  |       1/9       |     64     | [checkpoints](-) |
| VisualPrism-7b |  336x336  |       1/16       |     36     | [checkpoints](-) |
| VisualPrism-7b-Qwen |  336x336  |       1/4       |    144 | [checkpoints](-) |
| VisualPrism-13b-Qwen |  336x336  |       1/4       |     144   | [checkpoints](-) |

Note: ONLY VisualPrism-7b is available now. Others will be available soon.
## Visualization ✨
VisualPrism provide visualization code for getting attention map of LLMs in LLaVA-series models.
1. Install packages
```
conda activate visualprism
pip install opencv-python
```
**Note**: VisualPrism uses `numpy` version == 1.26.4, remember to switch back to correct version after  `opencv-python` installation

2. Visualize the attention map
 ```shell
bash scripts/visualization.sh # Visualize attention distribution
bash scripts/heatmap.sh # Visualize attention map of LLMs
```
Hope the code will help you in your research🌷.

## Acknowledgement 💌
- [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA)
- [TokenPacker](https://github.com/CircleRadon/TokenPacker)
  
## More ## 
For more recent related works, please refer to this repo of  [Awesome-Token-Compress](https://github.com/daixiangzi/Awesome-Token-Compress).
