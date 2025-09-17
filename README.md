# VisualPrism

## Table of Contents
- [Introduction](Introduction)
- [Install](#install)
- [Model Zoo](https://github.com)
- [Train](#train)
- [Evaluation](#evaluation)
- [Visualization](#Visualization)

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

## Train

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
