import torch
import torch.nn as nn
import re
from functools import partial
import numpy as np
from torch.nn.init import trunc_normal_
from torch.nn import functional as F
import math

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}

class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class OptimizedHaarWavelet2D(nn.Module):
    """ Haar Wavelet Transform Module """
    def __init__(self, channels):
        super().__init__()
        self._init_filters()

    def _init_filters(self):
        # Precompute Haar filter and register it as buffer
        ll_filter = torch.tensor([[1., 1.], [1., 1.]]) / 2.0
        lh_filter = torch.tensor([[1., -1.], [1., -1.]]) / 2.0
        hl_filter = torch.tensor([[1., 1.], [-1., -1.]]) / 2.0
        hh_filter = torch.tensor([[1., -1.], [-1., 1.]]) / 2.0

        # Register as buffers
        dec_filter = torch.stack([
            ll_filter, lh_filter, hl_filter, hh_filter
        ], dim=0).unsqueeze(1) # [4, 1, 2, 2]
        self.register_buffer('dec_filter', dec_filter, persistent=True)

        # Inverse Transform filter
        rec_filter = torch.stack([
            ll_filter, lh_filter, hl_filter, hh_filter
        ], dim=0).unsqueeze(1) / 2.0
        self.register_buffer('rec_filter', rec_filter, persistent=True)

    def decompose(self, x):
        """
        x: [B, C, H, W]
        return: [B, C, 4, H/2, W/2] (LL, LH, HL, HH located at dim=2)
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # Apply filters on each channel
        x_unfold = F.unfold(x, kernel_size=2, stride=2) # [B, C*4, H*W/4]
        x_unfold = x_unfold.view(B, C, 4, -1) # [B, C, 4, H*W/4]

        # Apply Haar Matrix transform
        haar_matrix = self.dec_filter.view(4, 4).t().to(device=device, dtype=dtype)
        x_transformed = torch.matmul(haar_matrix.unsqueeze(0).unsqueeze(0), x_unfold) # [B, C, 4, H*W/4]

        return x_transformed.view(B, C, 4, H//2, W//2)

    def reconstruct(self, x_wt):
        """
        x_wt: [B, C, 4, H/2, W/2]
        return: [B, C, H, W]
        """
        B, C, _, H_half, W_half = x_wt.shape
        device = x_wt.device
        dtype = x_wt.dtype

        # Inverse Haar transform
        inverse_haar = self.rec_filter.view(4, 4).to(device=device, dtype=dtype)
        x_flat = x_wt.view(B, C, 4, -1) # [B, C, 4, H_half*W_half]
        x_reconstructed = torch.matmul(inverse_haar.unsqueeze(0).unsqueeze(0), x_flat) # [B, C, 4, H_half*W_half]

        # Reconstruct back to image
        x_reconstructed = x_reconstructed.view(B, C, 2, 2, H_half, W_half)
        x_reconstructed = x_reconstructed.permute(0, 1, 4, 2, 5, 3).contiguous()
        x_reconstructed = x_reconstructed.view(B, C, H_half*2, W_half*2)

        return x_reconstructed

class EfficientWaveletConv(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.wavelet = OptimizedHaarWavelet2D(channels)

        # Apply single class convolution at each of the 4 channels
        self.subband_conv = nn.Conv2d(
            channels, channels,
            kernel_size = kernel_size,
            padding = kernel_size//2,
            groups = channels,
            bias = False
        )
        self.subband_bn = nn.BatchNorm2d(channels)
        self.subband_act = nn.GELU()

        # Scale factor
        self.register_buffer('subband_scale', torch.tensor([1.0, 1.2, 1.2, 1.5], dtype=torch.float32))

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        dtype = x.dtype
        
        # Wavelet Decomposition
        x_wt = self.wavelet.decompose(x)  # [B, C, 4, H/2, W/2]
        
        # Perform in each subband
        enhanced_subbands = []
        for i in range(4):
            # extract subband
            subband = x_wt[:, :, i, :, :]  # [B, C, H/2, W/2]
            
            # apply conv enhancement
            subband_enhanced = self.subband_conv(subband)
            subband_enhanced = self.subband_bn(subband_enhanced)
            subband_enhanced = self.subband_act(subband_enhanced)
            
            # apply scale factor
            scale = self.subband_scale[i].to(dtype)
            subband_enhanced = subband_enhanced * scale
            
            enhanced_subbands.append(subband_enhanced)
        
        # Enhance and Reconstruct
        x_enhanced = torch.stack(enhanced_subbands, dim=2)  # [B, C, 4, H/2, W/2]
        x_reconstructed = self.wavelet.reconstruct(x_enhanced)
        
        return x_reconstructed
    
class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList()

        for k in kernels:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, k, padding=k//2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.GELU()
            ))

        # Combine different kernels
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * len(kernels), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        multi_scale_features = []

        for conv in self.convs:
            multi_scale_features.append(conv(x))

        x = torch.cat(multi_scale_features, dim=1)
        x = self.fusion(x)
        return x
    
class GlobalContextModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Pooling + Channel attention
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.gap(x) # [B, C, 1, 1]
        y = self.excite(y) # [B, C ,1, 1]
        return x * y

class VisualPrism(nn.Module):
    def __init__(
        self,
        raw_grid=24,
        embed_dim=1024,
        num_heads=8,
        kv_dim=1024,
        hidden_size=4096,
        scale_factor=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        if raw_grid % scale_factor != 0:
            raise ValueError('scale_factor must be divisible by grid size')
        
        self.raw_grid = raw_grid
        self.grid_size = raw_grid // scale_factor
        self.num_queries = self.grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale_factor = scale_factor
        self.kv_dim = kv_dim
        self.hidden_size = hidden_size

        # Query - 3 paths
        # path1: Linear projection
        self.q_identity_proj = nn.Linear(kv_dim, embed_dim)

        # path2: MultiScaleConv - for local feature
        self.q_local = MultiScaleConvBlock(kv_dim, embed_dim, kernels=[3, 5])

        # path3: Wavelet
        self.q_freq = nn.Sequential(
            EfficientWaveletConv(kv_dim, kernel_size=3),
            nn.Conv2d(kv_dim, embed_dim, 1)
        )

        # Channel Fusion
        self.q_fusion_weights = nn.Parameter(torch.ones(3) / 3)

        # Final Query Projection
        self.q_proj_1 = nn.Linear(embed_dim, embed_dim, bias=False)

        # K & V -- Local & Global information
        self.kv_down_proj = nn.Linear(4096, embed_dim)

        # Key
        self.k_local = MultiScaleConvBlock(embed_dim, embed_dim, kernels=[3, 5, 7])
        self.k_global = GlobalContextModule(embed_dim)

        # Value
        self.v_local = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        self.v_global = GlobalContextModule(embed_dim)

        # LayerNorm
        self.ln_q_1 = norm_layer(embed_dim)
        self.ln_k_1 = norm_layer(embed_dim)
        self.ln_v_1 = norm_layer(embed_dim)

        # MultiheadAttn
        self.clip_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def process_query(self, x):
        B, H, W, C = x.shape

        # Path1: Linear Projection
        x_flat = x.reshape(B, H * W, C)
        q1 = self.q_identity_proj(x_flat).reshape(B, H, W, self.embed_dim)

        # Path2&3
        x_cf = x.permute(0, 3, 1, 2).contiguous() # [B, C, H, W]

        # Path2: Multi scale local
        q2 = self.q_local(x_cf).permute(0, 2, 3, 1) # [B, H, W, embed_dim]

        #Path3: Wavelet Frequency
        q3 = self.q_freq(x_cf).permute(0, 2, 3, 1) # [B, H, W, embed_dim]

        # Fusion
        weights = F.softmax(self.q_fusion_weights, dim=0)
        q_combined = weights[0] * q1 + weights[1] * q2 + weights[2] * q3

        # Down sampling
        if self.scale_factor > 1:
            q_combined = F.interpolate(
                q_combined.permute(0, 3, 1, 2).float(),
                size = (self.grid_size, self.grid_size),
                mode = 'bilinear',
                align_corners = False
            ).permute(0, 2, 3, 1).to(x.dtype)

        return q_combined

    def process_kv(self, x):
        B, N, D = x.shape
        H = W = int(math.sqrt(N))

        # Downsampling
        x = self.kv_down_proj(x) # [B, N, embed_dim]
        x_2d = x.reshape(B, H, W, self.embed_dim).permute(0, 3, 1, 2).contiguous()

        # Key
        k_local = self.k_local(x_2d)
        k_global = self.k_global(x_2d)
        k = k_local + 0.6 * k_global

        # Value
        v_local = self.v_local(x_2d)
        v_global = self.v_global(x_2d)
        v = v_local + 0.4 * v_global

        # Permute
        k = k.permute(0, 2, 3, 1).reshape(B, N, self.embed_dim)
        v = v.permute(0, 2, 3, 1).reshape(B, N, self.embed_dim)

        return k, v

    def divide_feature(self, x, kernel_size, token_num, N, c):
        h = w = int(token_num ** 0.5)

        # Reshape
        x = x.view(h//kernel_size, kernel_size, w//kernel_size, kernel_size, N, c)
        x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
        x = x.view(h//kernel_size * w//kernel_size, kernel_size*kernel_size, N, c)
        x = x.transpose(0, 1).contiguous()
        x = x.view(kernel_size*kernel_size, -1, c)

        return x

    def forward(self, x, attn_mask=None):
        x_multi = x[1] # [B, N, hidden_size]
        x = x[0] # [B, H*W, kv_dim]

        # K & V
        key, value = self.process_kv(x_multi)
        key = self.ln_k_1(key).permute(1, 0, 2)  # [N, B, embed_dim]
        value = self.ln_v_1(value).permute(1, 0, 2)
        
        token_num, N, c = key.shape

        # Query
        x_2d = x.reshape(x.shape[0], self.raw_grid, self.raw_grid, -1)
        q = self.process_query(x_2d)  # [B, grid_size, grid_size, embed_dim]
        # Norm & Proj
        q = q.reshape(q.shape[0], -1, q.shape[-1])
        query = self.ln_q_1(self.q_proj_1(q)).permute(1, 0, 2) 

        # Reconstruct features
        reshape_query = self.divide_feature(query, 1, self.num_queries, N, c)
        reshape_key = self.divide_feature(key, self.scale_factor, token_num, N, c)
        reshape_value = self.divide_feature(value, self.scale_factor, token_num, N, value.shape[-1])

        # MultiheadAttn
        out = self.clip_attn(
            reshape_query.transpose(0, 1),
            reshape_key.transpose(0, 1),
            reshape_value.transpose(0, 1),
            attn_mask = attn_mask
        )[0].transpose(0, 1)

        # Output
        x = out.reshape(self.num_queries, N, -1).permute(1, 0, 2)
        x = self.mlp(x)

        return x

def build_vision_projector(config):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    
    elif projector_type == 'identity':
        return IdentityMap()
    
    elif projector_type == 'visualprism':
        return VisualPrism(hidden_size=config.hidden_size, scale_factor=config.scale_factor)