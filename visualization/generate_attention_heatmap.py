#!/usr/bin/env python3
"""
LLaVA Attention Heatmap Visualization Script
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
import os
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# LLaVA imports
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model import *
from transformers import AutoTokenizer

class AttentionHeatmapExtractor:
    def __init__(self):
        self.attention_maps = {}
        self.original_forwards = {}
        
    def clear(self):
        for module, forward_fn in self.original_forwards.items():
            module.forward = forward_fn
        self.attention_maps = {}
        self.original_forwards = {}
        
    def register_hooks(self, model, layer_indices):
        self.clear()
        llm = model.model
        
        for idx, layer in enumerate(llm.layers):
            if idx in layer_indices:
                self.original_forwards[layer.self_attn] = layer.self_attn.forward
                
                def make_hook(layer_idx, original_forward):
                    def hooked_forward(*args, **kwargs):
                        kwargs['output_attentions'] = True
                        output = original_forward(*args, **kwargs)
                        
                        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                            attn_weights = output[1].detach().float()
                            self.attention_maps[f'layer_{layer_idx}'] = attn_weights
                        
                        return output
                    return hooked_forward
                
                layer.self_attn.forward = make_hook(idx, self.original_forwards[layer.self_attn])

def create_attention_heatmap(attention_weights, image, image_token_len, patch_size=14, image_size=336):
    """
    将attention权重转换为图像热力图
    
    Args:
        attention_weights: attention权重 (seq_len,) 或 (num_heads, seq_len)
        image: 原始图像 (PIL Image)
        image_token_len: 图像token数量
        patch_size: vision transformer的patch大小
        image_size: 处理后的图像大小
    """
    # 确保是numpy数组
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.cpu().numpy()
    
    # 如果是多头attention，取平均
    if len(attention_weights.shape) > 1:
        attention_weights = attention_weights.mean(axis=0)
    
    # 提取图像部分的attention
    image_attention = attention_weights[:image_token_len]
    
    # 计算patch网格大小
    num_patches_per_side = image_size // patch_size
    
    # 重塑为2D网格
    # 注意：可能需要去掉[CLS] token
    if len(image_attention) == num_patches_per_side * num_patches_per_side + 1:
        # 去掉CLS token
        image_attention = image_attention[1:]
    
    # 确保尺寸匹配
    expected_patches = num_patches_per_side * num_patches_per_side
    if len(image_attention) != expected_patches:
        print(f"Warning: Expected {expected_patches} patches, got {len(image_attention)}")
        # 尝试调整
        if len(image_attention) > expected_patches:
            image_attention = image_attention[:expected_patches]
        else:
            # 填充
            padding = expected_patches - len(image_attention)
            image_attention = np.pad(image_attention, (0, padding), mode='constant')
    
    # 重塑为2D
    attention_map = image_attention.reshape(num_patches_per_side, num_patches_per_side)
    
    # 上采样到原始图像大小
    attention_map_resized = cv2.resize(attention_map, (image_size, image_size), 
                                      interpolation=cv2.INTER_CUBIC)
    
    # 归一化
    attention_map_resized = (attention_map_resized - attention_map_resized.min()) / \
                           (attention_map_resized.max() - attention_map_resized.min() + 1e-8)
    
    # 调整图像大小以匹配attention map
    image_resized = image.resize((image_size, image_size), Image.LANCZOS)
    
    return attention_map_resized, image_resized

def visualize_attention_heatmaps(attention_maps, image, image_token_len, output_dir, 
                                text_positions=None, generated_text=""):
    """生成attention热力图可视化"""
    
    # 创建颜色映射
    colors = ['blue', 'cyan', 'yellow', 'red']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('attention', colors, N=n_bins)
    
    for layer_name, attn in attention_maps.items():
        print(f"\nProcessing {layer_name}")
        
        # attn shape: (batch, num_heads, query_len, key_len)
        attn_np = attn[0].cpu().numpy()  # (num_heads, query_len, key_len)
        
        # 获取最后几个token的attention（生成的文本）
        if attn_np.shape[1] == 1:
            # Cached attention - 最后一个token
            attention_to_image = attn_np[:, 0, :image_token_len]  # (num_heads, image_tokens)
            query_positions = ["Last Token"]
        else:
            # 完整attention矩阵 - 取最后几个位置
            num_text_tokens = min(5, attn_np.shape[1] - image_token_len)
            if num_text_tokens > 0:
                start_idx = max(image_token_len, attn_np.shape[1] - num_text_tokens)
                attention_to_image = attn_np[:, start_idx:, :image_token_len]  # (num_heads, num_queries, image_tokens)
                query_positions = [f"Position {i}" for i in range(start_idx, attn_np.shape[1])]
            else:
                continue
        
        # 为每个查询位置创建热力图
        if len(attention_to_image.shape) == 2:
            attention_to_image = attention_to_image[np.newaxis, :]  # 添加query维度
            
        num_queries = attention_to_image.shape[1]
        fig = plt.figure(figsize=(6 * min(num_queries, 4), 8))
        
        for q_idx in range(min(num_queries, 4)):  # 最多显示4个位置
            # 平均所有attention heads
            attn_weights = attention_to_image[:, q_idx, :].mean(axis=0)
            
            # 创建热力图
            heatmap, image_resized = create_attention_heatmap(
                attn_weights, image, image_token_len
            )
            
            # 显示原图和热力图
            ax1 = plt.subplot(2, min(num_queries, 4), q_idx + 1)
            ax1.imshow(image_resized)
            ax1.set_title(f'{query_positions[q_idx]}', fontsize=12)
            ax1.axis('off')
            
            ax2 = plt.subplot(2, min(num_queries, 4), min(num_queries, 4) + q_idx + 1)
            ax2.imshow(image_resized)
            im = ax2.imshow(heatmap, cmap=cmap, alpha=0.6)
            ax2.set_title(f'Attention Heatmap', fontsize=12)
            ax2.axis('off')
            
            # 添加colorbar
            if q_idx == min(num_queries, 4) - 1:
                cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
                cbar.set_label('Attention Weight', rotation=270, labelpad=15)
        
        plt.suptitle(f'{layer_name} - Image Attention Heatmaps', fontsize=14)
        plt.tight_layout()
        save_path = output_dir / f'heatmap_{layer_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
        
        # 生成聚合热力图（所有heads和positions的平均）
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. 平均attention热力图
        all_attn = attention_to_image.mean(axis=(0, 1))  # 平均所有heads和positions
        heatmap_avg, _ = create_attention_heatmap(all_attn, image, image_token_len)
        
        axes[0].imshow(image_resized)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(image_resized)
        im = axes[1].imshow(heatmap_avg, cmap=cmap, alpha=0.6)
        axes[1].set_title('Average Attention Heatmap', fontsize=14)
        axes[1].axis('off')
        
        # 2. 最高attention区域
        threshold = np.percentile(heatmap_avg, 80)  # top 20%
        mask = heatmap_avg > threshold
        
        axes[2].imshow(image_resized)
        masked_heatmap = np.zeros_like(heatmap_avg)
        masked_heatmap[mask] = heatmap_avg[mask]
        axes[2].imshow(masked_heatmap, cmap='Reds', alpha=0.7)
        axes[2].set_title('High Attention Regions (Top 20%)', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = output_dir / f'heatmap_summary_{layer_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--text', type=str, default="Describe this image in detail.")
    parser.add_argument('--output_dir', type=str, default='./attention_heatmaps')
    parser.add_argument('--layers', type=int, nargs='+', default=[16, 23])  # 后面的层通常更有意义
    parser.add_argument('--conv_mode', type=str, default="llava_v1")
    parser.add_argument('--max_new_tokens', type=int, default=50)
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from: {args.model_path}")
    print(f"Processing image: {args.image_path}")
    
    # 禁用初始化
    disable_torch_init()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=2048,
        padding_side="right",
        use_fast=True
    )
    
    # 加载模型
    print("Loading model...")
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).cuda()
    
    # 设置tokenizer
    for m in model.modules():
        m.tokenizer = tokenizer
    
    # 加载vision tower
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device='cuda', dtype=torch.float16)
    image_processor = vision_tower.image_processor
    
    model.eval()
    
    # 准备输入
    from run_attention_visualization import prepare_inputs, safe_decode_output
    image, image_tensor, input_ids, prompt = prepare_inputs(
        model, args.image_path, args.text, tokenizer, image_processor, args.conv_mode
    )
    
    # 移动到设备
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device='cuda', dtype=torch.bfloat16)
    input_ids = input_ids.unsqueeze(0).to(device='cuda')
    
    # 获取图像token长度
    print("\nEncoding image features...")
    with torch.no_grad():
        image_features = model.encode_images(image_tensor)
        image_token_len = image_features.shape[1]
    print(f"Image encoded into {image_token_len} tokens")
    
    # 创建attention提取器
    extractor = AttentionHeatmapExtractor()
    
    # 检查层数
    num_layers = len(model.model.layers)
    valid_layers = [l for l in args.layers if l < num_layers]
    print(f"Visualizing layers: {valid_layers}")
    
    # 注册hooks
    extractor.register_hooks(model, valid_layers)
    
    # 生成文本
    print(f"\nGenerating text...")
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        
        # 解码文本
        generated_text = safe_decode_output(tokenizer, output_ids)
        print(f"\nGenerated text:\n{generated_text}\n")
        
        # 生成热力图
        print("Generating attention heatmaps...")
        visualize_attention_heatmaps(
            extractor.attention_maps,
            image,
            image_token_len,
            output_dir,
            generated_text=generated_text
        )
        
        # 生成跨层对比
        if len(extractor.attention_maps) > 1:
            fig, axes = plt.subplots(1, len(valid_layers), figsize=(5*len(valid_layers), 5))
            if len(valid_layers) == 1:
                axes = [axes]
                
            for idx, layer_idx in enumerate(valid_layers):
                layer_name = f'layer_{layer_idx}'
                if layer_name in extractor.attention_maps:
                    attn = extractor.attention_maps[layer_name]
                    # 获取最后一个token对图像的attention
                    if attn.shape[2] == 1:  # cached
                        attn_to_img = attn[0, :, 0, :image_token_len].mean(0).cpu().numpy()
                    else:
                        attn_to_img = attn[0, :, -1, :image_token_len].mean(0).cpu().numpy()
                    
                    heatmap, image_resized = create_attention_heatmap(
                        attn_to_img, image, image_token_len
                    )
                    
                    axes[idx].imshow(image_resized)
                    axes[idx].imshow(heatmap, cmap='hot', alpha=0.6)
                    axes[idx].set_title(f'Layer {layer_idx}')
                    axes[idx].axis('off')
            
            plt.suptitle('Attention Heatmaps Across Layers', fontsize=16)
            plt.tight_layout()
            save_path = output_dir / 'layer_comparison_heatmap.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_path}")
        
    finally:
        extractor.clear()
    
    print("\nDone!")

if __name__ == "__main__":
    main()