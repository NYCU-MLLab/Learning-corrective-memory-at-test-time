#!/usr/bin/env python3
"""
PreCo 模型的 Dual Form 實現
參考 JAX main 的封閉解實現，避免複雜的迭代和參數更新
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.ttt import TTTConfig, TTTLinear
# Longhorn 類複製自 PreCo.py
from einops import rearrange
from mamba_ssm.ops.selective_scan_interface import selective_scan_online7_fn

class Longhorn(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, conv_bias=True, bias=False, use_fast_path=False, layer_idx=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.activation = "silu"
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state*2, bias=False, **factory_kwargs
        )
        self.dt_head = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.x_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.x_proj.weight.device,
                dtype=self.x_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, dim = hidden_states.shape
        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out, None
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        x, z = xz.chunk(2, dim=1)
        if conv_state is not None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            x = self.act(self.conv1d(x)[..., :seqlen])
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, k, q = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_head.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        k = rearrange(k, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        q = rearrange(q, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_online7_fn(x, q.to(x), k.to(x), dt.to(x), D=self.D.float(), t_bias=self.dt_head.bias.float(), z=z)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out, y

    def step(self, hidden_states, conv_state, ssm_state):
        # 簡化的 step 實現
        batch, seqlen, dim = hidden_states.shape
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x)[..., :seqlen])
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, k, q = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_head.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        k = rearrange(k, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        q = rearrange(q, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_online7_fn(x, q.to(x), k.to(x), dt.to(x), D=self.D.float(), t_bias=self.dt_head.bias.float(), z=z)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out, conv_state, ssm_state

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMSNorm 實現
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        x_norm = x_norm * (x.shape[-1] ** -0.5)
        return self.weight * (x / (x_norm + self.eps))

class PreCoDualConfig:
    def __init__(self,
                 vocab_size=50257,
                 d_model=512,
                 n_layer=12,
                 d_state=8,
                 d_conv=3,
                 expand=6,
                 ttt_num_heads=8,
                 dropout=0.1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.ttt_num_heads = ttt_num_heads
        self.dropout = dropout

class PreCoDualBlock(nn.Module):
    def __init__(self, config: PreCoDualConfig, layer_idx=None):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        
        # Longhorn 分支
        self.longhorn_block = Longhorn(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            layer_idx=layer_idx,
        )
        self.reduce_hidden = nn.Linear(config.d_model * config.expand, config.d_model, bias=False)
        
        # TTT 分支 - 簡化的 dual form
        ttt_cfg = TTTConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.d_model,
            intermediate_size=config.d_model * 4,
            num_attention_heads=config.ttt_num_heads,
            num_hidden_layers=1,
            pre_conv=False,
            mini_batch_size=8,
            mlp_ratio=4,
        )
        self.corr_linear = TTTLinear(ttt_cfg)
        
        # 投影網絡
        self.q_network = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # 簡化的 gain 機制
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        
        # 初始化
        with torch.no_grad():
            self.alpha.data.fill_(1.0)
            self.beta.data.fill_(0.0)

    def forward(self, x, residual=None, inference_params=None, ttt_lr_mult=1.0):
        residual = (x + residual) if residual is not None else x
        x_norm = self.norm(residual)
        
        # Longhorn 分支
        h_out, h_hidden = self.longhorn_block(x_norm, inference_params=inference_params)
        h_hidden_reduced = self.reduce_hidden(h_hidden)  # [B, L, d_model]
        
        # TTT 分支 - 使用簡化的 dual form
        c_hidden = self._ttt_dual_forward(h_hidden_reduced, ttt_lr_mult)
        
        # 投影
        z_longhorn = self.q_network(h_hidden_reduced)
        z_ttt = self.q_network(c_hidden)
        
        # 簡化的 gain 計算
        gain = self._compute_gain(h_hidden_reduced, z_longhorn, z_ttt)
        
        # 融合輸出
        out = z_longhorn + gain * (z_ttt - z_longhorn)
        return out, residual, gain
    
    def _ttt_dual_forward(self, hidden_states, ttt_lr_mult=1.0):
        """TTT 的 dual form 實現 - 封閉解，無需迭代"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        ttt_layer = self.corr_linear
        
        # 獲取 QKV 投影
        XQK = ttt_layer.qkv_learnable_ttt_lr_proj(hidden_states)
        XQK, XV, XGate, ttt_lr = torch.split(XQK, [
            ttt_layer.hidden_size, 
            ttt_layer.hidden_size, 
            ttt_layer.hidden_size, 
            ttt_layer.num_heads
        ], dim=-1)
        
        # 重塑為多頭格式
        XQ = XQK.reshape(batch_size, seq_len, ttt_layer.num_heads, ttt_layer.head_dim).permute(0, 2, 1, 3)
        XK = XQK.reshape(batch_size, seq_len, ttt_layer.num_heads, ttt_layer.head_dim).permute(0, 2, 1, 3)
        XV = XV.reshape(batch_size, seq_len, ttt_layer.num_heads, ttt_layer.head_dim).permute(0, 2, 1, 3)
        
        # 展平為 [B*nh, N, f]
        XQ = XQ.reshape(-1, seq_len, ttt_layer.head_dim).contiguous()
        XK = XK.reshape(-1, seq_len, ttt_layer.head_dim).contiguous()
        XV = XV.reshape(-1, seq_len, ttt_layer.head_dim).contiguous()
        
        # Dual Form: 直接計算最優解
        # 1. 計算自監督目標
        ssl_target = XV - XK  # [B*nh, N, f]
        
        # 2. 計算最優參數（封閉解）
        # W* = (XK^T XK + λI)^(-1) XK^T ssl_target
        lambda_reg = 1e-6  # 正則化參數
        
        # 計算 XK^T XK
        XK_t_XK = torch.matmul(XK.transpose(-1, -2), XK)  # [B*nh, f, f]
        XK_t_XK += lambda_reg * torch.eye(XK_t_XK.size(-1), device=XK_t_XK.device)
        
        # 計算 XK^T ssl_target
        XK_t_target = torch.matmul(XK.transpose(-1, -2), ssl_target)  # [B*nh, f, f]
        
        # 求解線性系統 (使用 Cholesky 分解提高穩定性)
        try:
            L = torch.linalg.cholesky(XK_t_XK)
            W_opt = torch.cholesky_solve(XK_t_target, L)  # [B*nh, f, f]
        except:
            # 如果 Cholesky 失敗，使用偽逆
            W_opt = torch.linalg.pinv(XK_t_XK) @ XK_t_target
        
        # 3. 計算最優輸出
        Z_opt = torch.matmul(XK, W_opt)  # [B*nh, N, f]
        
        # 4. 應用 LayerNorm
        mu = Z_opt.mean(dim=-1, keepdim=True)
        var = Z_opt.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + 1e-5)
        Z_normed = (Z_opt - mu) / std
        
        # 5. 重塑回原始格式
        output = Z_normed.reshape(batch_size, ttt_layer.num_heads, seq_len, ttt_layer.head_dim)
        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        # 6. 應用門控和最終投影
        gate = F.gelu(XGate)
        output = gate * ttt_layer.post_norm(output)
        output = ttt_layer.wo(output)
        
        return output
    
    def _compute_gain(self, h_hidden_reduced, z_longhorn, z_ttt):
        """簡化的 gain 計算"""
        # 基於方差的最優 gain
        hidden_var = torch.var(h_hidden_reduced, dim=-1, keepdim=True)
        gain = torch.sigmoid(self.alpha * hidden_var + self.beta)
        gain = torch.clamp(gain, min=0.1, max=0.9)
        
        if not self.training:
            gain = gain * 0.8
        
        return gain

class PreCoDualModel(nn.Module):
    def __init__(self, config: PreCoDualConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([PreCoDualBlock(config, layer_idx=i) for i in range(config.n_layer)])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.dropout = nn.Dropout(config.dropout)
        self.config = config
        
        # 權重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            if hasattr(module, 'weight') and module.weight is not None:
                torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, targets=None, inference_params=None, ttt_lr_mult=None, compute_branch_loss=False):
        x = self.embedding(input_ids)
        residual = None
        
        # 收集 gain 統計
        all_gains = []
        
        for i, block in enumerate(self.blocks):
            block_infer = None
            if inference_params is not None and hasattr(inference_params, 'key_value_memory_dict'):
                block_infer = inference_params
            x, residual, gain = block(x, residual=residual, inference_params=block_infer, ttt_lr_mult=ttt_lr_mult)
            all_gains.append(gain)
        
        # 最後一層 RMSNorm
        x = self.norm_f((x + residual) if residual is not None else x)
        x = self.dropout(x)
        logits = self.lm_head(x)
        
        # 保存 gain 統計
        if not self.training and all_gains:
            try:
                all_gains_tensor = torch.cat([g.view(-1, 1) for g in all_gains], dim=0)
                self.last_gains = all_gains_tensor
            except Exception as e:
                first_gain = all_gains[0].view(-1)
                self.last_gains = first_gain
        
        if targets is not None:
            # 主要 loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            
            # 簡化的統計
            if all_gains:
                try:
                    all_gains_tensor = torch.cat([g.view(-1, 1) for g in all_gains], dim=0)
                    kalman_mean = all_gains_tensor.mean()
                    kalman_std = all_gains_tensor.std()
                except Exception as e:
                    first_gain = all_gains[0].view(-1)
                    kalman_mean = first_gain.mean()
                    kalman_std = first_gain.std()
            else:
                kalman_mean = torch.tensor(0.5, device=logits.device)
                kalman_std = torch.tensor(0.1, device=logits.device)
            
            loss_dict = {
                'total_loss': loss,
                'ce_loss': loss,
                'longhorn_loss': torch.tensor(0.0, device=logits.device),
                'ttt_loss': torch.tensor(0.0, device=logits.device),
                'kalman_mean': kalman_mean,
                'kalman_std': kalman_std,
            }
            return logits, loss_dict
        else:
            return logits 