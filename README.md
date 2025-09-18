# VisualPrism

## Table of Contents
- [Introduction](#Introduction)
- [Install](#install)
- [Train](#train)
- [Evaluation](#evaluation)
- [Model Zoo](#Modelzoo)
- [Visualization](#Visualization)

## Introduction	🌈
**VisualPrism** is an enhancement over [LLaVA](https://github.com/haotian-liu/LLaVA), introducing a **guide-then-compress** strategy to significantly reduce visual tokens while preserving rich semantics. Inspired by the **fovea–periphery coordination** in human vision, VisualPrism injects explicit visual priors to improve efficiency and accuracy in vision-language understanding.

- 🎯 **Prior-Guided Compression**: Integrates frequency-domain priors (via Discrete Wavelet Transform) before token merging.
- 🔀 **Triple-Path Query Projection (TPQP)**: Decomposes visual queries into global, local, and identity-based components.
- 🧩 **Hierarchical Key-Value Encoding (HKVE)**: Structures visual information across scales for better alignment and retrieval.
- 🔌 **Modular & Compatible**: Drop-in replacement for LLaVA's projector, compatible with most MLLM pipelines.

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

#### Dataset
To make a fair comparison, we use the same training data as in [LLaVA-1.5](https://github.com/haotian-liu/LLaVA), i.e., [LLaVA-Pretrain-558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main) for stage 1, and  [Mix665k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main) for stage 2.

#### Training 
- Stage1: Image-Text Alignment Pre-training
```shell
bash scripts/v1_5/pretrain.sh
```
- Stage2: Visual Instruction Tuning
```shell
bash scripts/v1_5/finetune.sh
```
Note: Using `--scale_factor` to control compression ratio, support [2,3,4]

## Evaluation 📊

#### Evaluation process 
VisualPrism follows LLaVA evaluation steps.
See [Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

#### Evaluation results 
| Model           | #Tokens | GQA  | VizWiz | VQAv2 | MM-Vet | Avg. Score |
|----------------|---------|------|--------|--------|--------|-------------|
| LLaVA-1.5      | 576     | 62.0 | 50.0   | 78.5   | 31.1   | 62.0        |
| TokenPacker    | 144     | 60.6 | 52.0   | 76.5   | 33.0   | 61.8        |
| **VisualPrism**| **144** | 59.9 | **53.3** | 76.3 | **34.5** | **62.4** |
| VisualPrism    | **64**  | 58.3 | 52.0   | 74.1   | 31.7   | 60.6        |

## Model Zoo

| Model              |  Max Res.   |  Compre. Ratio  |  Token Num.  |  Max Patch Num.  |                                           Training Data                                            | Download                                                                              |
|--------------------|:-----------:|:---------------:|:------------:|:----------------:|:--------------------------------------------------------------------------------------------------:|---------------------------------------------------------------------------------------|
| VisualPrism-7b     |   336x336   |       1/4       |     144      |        -         |                                             558K+665K                                              | [checkpoints](https://huggingface.co/)  |
| VisualPrism-13b     |   336x336   |       1/4       |     144      |        -         |                                             558K+665K                                              | [checkpoints](https://huggingface.co/) |
| VisualPrism-7b  |  336x336  |       1/9       |     64     |         -          |                                             558K+665K                                              | [checkpoints](https://huggingface.co/) |
| VisualPrism-7b |  336x336  |       1/16       |     36     |         -          |                                             558K+665K                                              | [checkpoints](https://huggingface.co/) |
| VisualPrism-7b-Qwen |  336x336  |       1/4       |    144     |         -         |                                             1.2M+1.5M                                              | [checkpoints](https://huggingface.co/) |
| VisualPrism-13b-Qwen |  336x336  |       1/4       |     144     |         -         |                                             1.2M+1.5M                                              | [checkpoints](https://huggingface.co/) |

## Visualization ✨
