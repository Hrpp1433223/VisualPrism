#!/usr/bin/env python3
"""
LLaVA LLM Attention Visualization Script - Fixed BFloat16 Issue
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
from functools import partial

# LLaVA imports
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model import *
from transformers import AutoTokenizer

class LLMAttentionExtractor:
    def __init__(self):
        self.attention_maps = {}
        self.all_attention_maps = {}  # 存储所有步骤的attention
        self.hooks = []
        self.layer_count = 0
        self.original_forwards = {}
        self.step_count = 0
        
    def clear(self):
        # 恢复原始forward方法
        for module, forward_fn in self.original_forwards.items():
            module.forward = forward_fn
        
        self.attention_maps = {}
        self.all_attention_maps = {}
        self.hooks = []
        self.layer_count = 0
        self.original_forwards = {}
        self.step_count = 0
        
    def register_llm_hooks(self, model, layer_indices=None):
        """注册hooks到LLM的attention层"""
        self.clear()
        
        # 获取language model
        llm = model.model  # LlamaModel
        
        # 注册到每个decoder layer的self attention
        for idx, layer in enumerate(llm.layers):
            if layer_indices is None or idx in layer_indices:
                # 保存原始forward方法
                self.original_forwards[layer.self_attn] = layer.self_attn.forward
                
                # 创建hook函数
                def make_attention_hook(layer_idx, original_forward):
                    def hooked_forward(*args, **kwargs):
                        # 强制输出attention weights
                        kwargs['output_attentions'] = True
                        output = original_forward(*args, **kwargs)
                        
                        # 保存attention weights
                        if isinstance(output, tuple) and len(output) >= 2:
                            if output[1] is not None:
                                # 转换为float32以避免BFloat16问题
                                attn_weights = output[1].detach().float()
                                layer_name = f'layer_{layer_idx}'
                                
                                # 保存最新的attention
                                self.attention_maps[layer_name] = attn_weights
                                
                                # 累积所有步骤的attention
                                if layer_name not in self.all_attention_maps:
                                    self.all_attention_maps[layer_name] = []
                                self.all_attention_maps[layer_name].append(attn_weights)
                        
                        return output
                    return hooked_forward
                
                # 设置新的forward方法
                layer.self_attn.forward = make_attention_hook(idx, self.original_forwards[layer.self_attn])
                print(f"Registered hook for LLM layer {idx}")
                self.layer_count += 1

def safe_decode_output(tokenizer, output_ids):
    """安全地解码输出，处理特殊token"""
    try:
        # 将tensor转换为列表
        if torch.is_tensor(output_ids):
            if len(output_ids.shape) == 2:
                output_ids = output_ids[0]  # 取第一个batch
            token_list = output_ids.cpu().tolist()
        else:
            token_list = output_ids
        
        # 过滤掉可能导致问题的token
        filtered_tokens = []
        for token_id in token_list:
            # 跳过IMAGE_TOKEN_INDEX和其他特殊值
            if token_id == IMAGE_TOKEN_INDEX:
                continue
            # 确保token在有效范围内
            if 0 <= token_id < tokenizer.vocab_size:
                filtered_tokens.append(token_id)
        
        # 尝试解码
        if filtered_tokens:
            text = tokenizer.decode(filtered_tokens, skip_special_tokens=True)
            return text
        else:
            return "[No valid tokens to decode]"
            
    except Exception as e:
        print(f"Warning: Failed to decode tokens: {e}")
        return "[Decoding failed]"

def prepare_inputs(model, image_path, text, tokenizer, image_processor, conv_mode='llava_v1'):
    """准备模型输入"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 处理图像
    image_tensor = process_images([image], image_processor, model.config)
    
    if isinstance(image_tensor, list):
        image_tensor = image_tensor[0]
        
    # 准备对话
    if hasattr(model.config, 'mm_use_im_start_end') and model.config.mm_use_im_start_end:
        text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text
    else:
        text = DEFAULT_IMAGE_TOKEN + '\n' + text
        
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Tokenize
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    
    return image, image_tensor, input_ids, prompt

def combine_cached_attention(attention_list):
    """组合缓存的attention steps成完整的attention矩阵"""
    if not attention_list:
        return None
        
    # 获取第一个attention的形状信息
    first_attn = attention_list[0]
    batch_size = first_attn.shape[0]
    num_heads = first_attn.shape[1]
    
    # 收集所有的attention weights
    all_attns = []
    for attn in attention_list:
        # attn shape: (batch, heads, query_len, key_len)
        # 对于cached attention，query_len通常是1
        all_attns.append(attn)
    
    # 组合成完整的矩阵
    # 这里我们只使用最后一个attention map，它包含了完整的key序列
    if len(attention_list) > 0:
        # 使用最后几个attention maps的平均或最后一个
        return attention_list[-1]
    
    return None

def visualize_attention_simple(attention_maps, all_attention_maps, image_token_len, layer_indices, output_dir):
    """简化的attention可视化"""
    
    for layer_idx in layer_indices:
        layer_name = f'layer_{layer_idx}'
        
        # 尝试使用累积的attention或最后的attention
        if layer_name in all_attention_maps and all_attention_maps[layer_name]:
            # 使用最后一个完整的attention state
            attn = all_attention_maps[layer_name][-1]
            print(f"\nProcessing {layer_name}, using last attention state, shape: {attn.shape}")
        elif layer_name in attention_maps:
            attn = attention_maps[layer_name]
            print(f"\nProcessing {layer_name}, attention shape: {attn.shape}")
        else:
            print(f"Warning: No attention data for {layer_name}")
            continue
        
        # 确保是float类型
        if attn.dtype == torch.bfloat16:
            attn = attn.float()
            
        # attn shape: (batch, num_heads, seq_len, seq_len) 或 (batch, num_heads, 1, seq_len) for cached
        attn_cpu = attn[0].cpu()  # (num_heads, query_len, key_len)
        
        # 如果是cached attention (query_len=1)，我们展示最后一个token的attention pattern
        if attn_cpu.shape[1] == 1:
            # 这是最新生成token的attention
            attn_avg = attn_cpu.mean(dim=0).squeeze(0).numpy()  # (key_len,)
            
            # 创建可视化
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # 1. Attention分布
            ax = axes[0]
            seq_len = len(attn_avg)
            ax.plot(range(seq_len), attn_avg, 'b-', linewidth=2)
            ax.fill_between(range(seq_len), attn_avg, alpha=0.3)
            ax.set_xlabel('Position')
            ax.set_ylabel('Attention Weight')
            ax.set_title(f'Last Token Attention Distribution - Layer {layer_idx}')
            ax.grid(True, alpha=0.3)
            
            # 标记图像区域
            if image_token_len > 0 and image_token_len < seq_len:
                ax.axvspan(0, image_token_len, alpha=0.2, color='red', label='Image region')
                ax.legend()
            
            # 2. 热力图
            ax = axes[1]
            # 展示前几个attention heads的pattern
            num_heads_to_show = min(8, attn_cpu.shape[0])
            heads_attn = attn_cpu[:num_heads_to_show, 0, :].numpy()  # (num_heads, key_len)
            
            im = ax.imshow(heads_attn, cmap='hot', aspect='auto')
            ax.set_xlabel('Position')
            ax.set_ylabel('Attention Head')
            ax.set_title(f'Per-Head Attention Patterns (first {num_heads_to_show} heads)')
            plt.colorbar(im, ax=ax)
            
            # 标记图像边界
            if image_token_len > 0:
                ax.axvline(x=image_token_len, color='blue', linestyle='--', alpha=0.7)
                
        else:
            # 完整的attention矩阵
            attn_avg = attn_cpu.mean(dim=0).numpy()  # (seq_len, seq_len)
            seq_len = attn_avg.shape[0]
            
            # 创建可视化
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 1. 完整attention矩阵
            ax = axes[0]
            max_size = 256
            if seq_len > max_size:
                factor = seq_len // max_size
                attn_small = attn_avg[::factor, ::factor]
            else:
                attn_small = attn_avg
                
            im = ax.imshow(attn_small, cmap='Blues', aspect='auto')
            ax.set_title(f'Attention Matrix - Layer {layer_idx}\n(size: {seq_len}x{seq_len})')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            plt.colorbar(im, ax=ax)
            
            # 2. 注意力熵
            ax = axes[1]
            attn_norm = attn_avg / (attn_avg.sum(axis=1, keepdims=True) + 1e-8)
            entropy = -np.sum(attn_norm * np.log(attn_norm + 1e-8), axis=1)
            
            display_len = min(seq_len, 500)
            ax.plot(range(display_len), entropy[:display_len], 'g-', linewidth=2)
            ax.set_xlabel('Position')
            ax.set_ylabel('Attention Entropy')
            ax.set_title('Attention Focus')
            ax.grid(True, alpha=0.3)
            
            # 3. Cross-modal attention
            ax = axes[2]
            if image_token_len > 0 and image_token_len < seq_len:
                text_to_image = attn_avg[image_token_len:, :image_token_len]
                if text_to_image.size > 0:
                    text_to_image_avg = text_to_image.mean(axis=1)
                    display_len = min(len(text_to_image_avg), 200)
                    ax.plot(range(display_len), text_to_image_avg[:display_len], 'b-', linewidth=2)
                    ax.fill_between(range(display_len), text_to_image_avg[:display_len], alpha=0.3)
                    ax.set_xlabel('Text Position')
                    ax.set_ylabel('Avg Attention to Image')
                    ax.set_title('Text→Image Attention')
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = output_dir / f'attention_layer_{layer_idx}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--text', type=str, default="Describe this image in detail.")
    parser.add_argument('--output_dir', type=str, default='./attention_results')
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 8, 16, 23])
    parser.add_argument('--conv_mode', type=str, default="llava_v1")
    parser.add_argument('--max_new_tokens', type=int, default=30)
    parser.add_argument('--use_cache', type=bool, default=True)
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from: {args.model_path}")
    print(f"Processing image: {args.image_path}")
    print(f"Text prompt: {args.text}")
    print(f"Visualizing LLM layers: {args.layers}")
    print(f"Use cache: {args.use_cache}")
    
    # 禁用初始化
    disable_torch_init()
    
    # 加载模型
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
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
    image, image_tensor, input_ids, prompt = prepare_inputs(
        model, args.image_path, args.text, tokenizer, image_processor, args.conv_mode
    )
    
    # 移动到正确的设备
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device='cuda', dtype=torch.bfloat16)
    input_ids = input_ids.unsqueeze(0).to(device='cuda')
    
    # 创建attention提取器
    extractor = LLMAttentionExtractor()
    
    # 检查可用层数
    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} LLM layers")
    valid_layers = [l for l in args.layers if l < num_layers]
    
    # 首先获取图像features以确定token长度
    print("\nEncoding image features...")
    with torch.no_grad():
        image_features = model.encode_images(image_tensor)
        image_token_len = image_features.shape[1]
    print(f"Image encoded into {image_token_len} tokens")
    
    # 注册hooks
    extractor.register_llm_hooks(model, valid_layers)
    
    # 生成文本并捕获attention
    print(f"\nGenerating {args.max_new_tokens} tokens...")
    generated_text = "[Generation not completed]"
    
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0,
                max_new_tokens=args.max_new_tokens,
                use_cache=args.use_cache,
            )
        
        print(f"Generated sequence length: {output_ids.shape[1]}")
        
        # 安全地解码生成的文本
        generated_text = safe_decode_output(tokenizer, output_ids)
        print(f"\nGenerated text:\n{generated_text}\n")
        
        # 检查捕获的attention
        print(f"Captured attention maps for {len(extractor.attention_maps)} layers")
        print(f"Total attention steps captured: {len(extractor.all_attention_maps.get('layer_0', []))}")
        
        # 可视化attention
        print("Visualizing attention patterns...")
        visualize_attention_simple(
            extractor.attention_maps,
            extractor.all_attention_maps,
            image_token_len,
            valid_layers,
            output_dir
        )
        
    except Exception as e:
        print(f"Error during generation or visualization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理hooks
        extractor.clear()
    
    # 保存配置
    config_path = output_dir / 'config.txt'
    with open(config_path, 'w') as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Image: {args.image_path}\n") 
        f.write(f"Text prompt: {args.text}\n")
        f.write(f"Valid layers: {valid_layers}\n")
        f.write(f"Image token length: {image_token_len}\n")
        f.write(f"Use cache: {args.use_cache}\n")
        if 'output_ids' in locals():
            f.write(f"Generated sequence length: {output_ids.shape[1]}\n")
        f.write(f"\nGenerated text:\n{generated_text}\n")
        
    print("\nDone!")

if __name__ == "__main__":
    main()