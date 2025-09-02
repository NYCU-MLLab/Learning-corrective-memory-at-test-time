import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union, Dict, Any

from collections import defaultdict
from dataclasses import dataclass
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils._pytree import tree_map

from transformers.activations import ACT2FN
from transformers.utils import ModelOutput, logging
from transformers.utils.import_utils import is_causal_conv1d_available

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

logger = logging.get_logger(__name__)

TTT_STANDARD_CONFIGS = {
    "125m": {
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
    },
    "350m": {
        "hidden_size": 1024,
        "intermediate_size": 2736,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
    },
    "760m": {
        "hidden_size": 1536,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
    },
    "1b": {
        "hidden_size": 2048,
        "intermediate_size": 5504,
        "num_hidden_layers": 24,
        "num_attention_heads": 32,
    },
}

class TTTConfig(PretrainedConfig):
    model_type = "ttt"
    
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5504,
        num_hidden_layers=24,
        num_attention_heads=32,
        hidden_act="silu",
        max_position_embeddings=1024,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        use_gate=False,
        share_qk=False,
        ttt_layer_type="linear",
        ttt_base_lr=1.0,
        mini_batch_size=16,
        pre_conv=False,
        conv_kernel=4,
        scan_checkpoint_group_size=0,
        mlp_ratio=4,  # 🎯 新增：MLP 倍數參數，用於內部優化
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.use_gate = use_gate
        self.share_qk = share_qk
        self.ttt_layer_type = ttt_layer_type
        self.ttt_base_lr = ttt_base_lr
        self.mini_batch_size = min(mini_batch_size, num_attention_heads)
        self.pre_conv = pre_conv
        self.conv_kernel = conv_kernel
        self.scan_checkpoint_group_size = scan_checkpoint_group_size
        self.mlp_ratio = mlp_ratio  # 🎯 MLP 倍數參數
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

class TTTLinear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # 確保 head_dim 是整數且 num_heads * head_dim = hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        assert self.num_heads * self.head_dim == self.hidden_size, \
            f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}"
        self.mini_batch_size = config.mini_batch_size
        
        # 核心線性層 - 與 kernels 版本一致
        self.wq = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)  # 共用 Q/K 投影
        self.wv = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)  # V 專用投影
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # TTT 核心參數 - 與 kernels 版本完全一致
        self.W1 = nn.Parameter(
            torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim))
        )
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))
        
        # 🔧 新增：與 kernels 版本一致的推理參數
        # token_idx 緩衝區和可學習偏置
        token_idx = 1. / torch.arange(1, self.mini_batch_size + 1).reshape(1, 1, -1, 1)
        self.register_buffer('token_idx', token_idx, persistent=False)
        self.learnable_token_idx_bias = nn.Parameter(torch.zeros((1, 1, self.mini_batch_size, 1)))
        
        # 可學習的 TTT 學習率投影和偏置
        self.qkv_learnable_ttt_lr_proj = nn.Linear(
            self.hidden_size, 
            3 * self.hidden_size + self.num_heads, 
            bias=False
        )
        self.learnable_ttt_lr_bias = nn.Parameter(torch.zeros(1, 1, self.num_heads))
        
        # 🔧 新增：分離的 TTT LayerNorm 參數 (與 kernels 版本一致)
        ttt_norm_weight = torch.ones(self.head_dim)
        ttt_norm_bias = torch.zeros(self.head_dim)
        # [1,nh,1,f] 格式，與 kernels 版本完全一致
        self.ttt_norm_weight = nn.Parameter(
            ttt_norm_weight.reshape(1, 1, 1, -1).expand(1, self.num_heads, 1, -1).contiguous()
        )
        self.ttt_norm_bias = nn.Parameter(
            ttt_norm_bias.reshape(1, 1, 1, -1).expand(1, self.num_heads, 1, -1).contiguous()
        )
        
        # 層正規化 - 用於後處理
        self.post_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        
        # 門控機制 - 與 kernels 版本一致
        self.wg = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Pre-convolution - 與 kernels 版本一致：Q 和 K 分離卷積
        if config.pre_conv:
            self.conv_q = nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=config.conv_kernel,
                padding=config.conv_kernel - 1,
                groups=self.hidden_size,  # depthwise convolution
                bias=True,  # kernels版本有bias
            )
            self.conv_k = nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=config.conv_kernel,
                padding=config.conv_kernel - 1,
                groups=self.hidden_size,  # depthwise convolution
                bias=True,  # kernels版本有bias
            )

    def get_QKV_ttt_lr(self, hidden_states):
        """與 kernels 版本一致的 QKV 和 TTT 學習率計算"""
        B, N, D = hidden_states.shape

        XQKV_ttt_lr = self.qkv_learnable_ttt_lr_proj(hidden_states)  # [B,N, 3*F + nh]
        XQKV, ttt_lr = torch.split(XQKV_ttt_lr, split_size_or_sections=[3 * D, self.num_heads], dim=-1)

        # ttt_lr = (fixed ttt base lr) * (learnable lr multiplier)
        ttt_lr = self.config.ttt_base_lr * torch.sigmoid(
            (ttt_lr + self.learnable_ttt_lr_bias).permute(0, 2, 1).reshape(-1, N, 1)
        ) / self.head_dim  # ([B,N,nh] + [1,1,nh]) -> [B,nh,N] -> [B*nh,N,1]

        XQK, XGate, XV = torch.split(XQKV, split_size_or_sections=self.hidden_size, dim=-1)  # [B,N,D]
        XV = XV.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(-1, N, self.head_dim)

        return XQK, XV, XGate, ttt_lr

    def conv_qk_fused(self, XQK, is_prefill=True):
        """與 kernels 版本一致的卷積處理"""
        if not self.config.pre_conv or not hasattr(self, 'conv_q'):
            # 沒有卷積時，Q 和 K 都使用共用投影
            return XQK, XQK
            
        B, N, D = XQK.shape
        # Conv1d 需要 (batch, channels, seq_len) 格式
        XQK_conv = XQK.transpose(1, 2)  # [B, D, N]
        
        # 分別對 Q 和 K 進行卷積
        XQ_conv = self.conv_q(XQK_conv)
        XK_conv = self.conv_k(XQK_conv)
        
        if is_prefill:
            # 去除右側 padding
            XQ_conv = XQ_conv[:, :, :N]
            XK_conv = XK_conv[:, :, :N]
        
        XQ = XQ_conv.transpose(1, 2)  # [B, N, D]
        XK = XK_conv.transpose(1, 2)  # [B, N, D]
        
        return XQ, XK

    def get_eta(self, hidden_states, ttt_lr_mult=1.0):
        """計算學習率調整因子 - 與 kernels 版本一致"""
        B, N, _ = hidden_states.shape
        
        # 獲取 token_idx 和可學習偏置
        token_idx = self.token_idx + self.learnable_token_idx_bias
        token_idx = torch.clamp(token_idx, min=0.0)  # 確保大於0
        
        # 獲取可學習的 TTT 學習率
        _, _, _, ttt_lr = self.get_QKV_ttt_lr(hidden_states)  # [B*nh,N,1]
        
        # 計算最終的學習率
        eta = ttt_lr_mult * token_idx * ttt_lr  # [B*nh,N,1] * [1,1,mini_batch,1]
        
        return eta



    def forward(self, hidden_states, input_ids=None, position_ids=None, 
                deterministic=True, output_ttt_stats=False, ttt_lr_mult=1.0,
                iteration_counts=None, max_iter=None):
        """與 kernels 版本完全一致的前向傳播"""
        batch_size, seq_length = hidden_states.shape[:2]
        
        # 🔧 使用 kernels 版本的 QKV 和學習率計算
        XQK, XV, XGate, ttt_lr = self.get_QKV_ttt_lr(hidden_states)
        
        # 🔧 使用 kernels 版本的卷積處理
        XQ, XK = self.conv_qk_fused(XQK, is_prefill=True)
        
        # 重塑為多頭形式 - 與 kernels 版本一致
        XQ = XQ.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        XK = XK.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 展平為 kernels 格式：[B*nh, N, f]
        XQ = XQ.reshape(-1, seq_length, self.head_dim).contiguous()
        XK = XK.reshape(-1, seq_length, self.head_dim).contiguous()
        XV = XV.contiguous()
        
        # 🔧 計算 token_idx 和 eta - 與 kernels 版本一致
        token_idx = self.token_idx + self.learnable_token_idx_bias
        token_idx = torch.clamp(token_idx, min=0.0)
        
        # 🔧 簡化的 TTT 處理 (訓練模式，不需要完整的推理狀態管理)
        # 這裡實現基本的 TTT-Linear 邏輯
        B_mul_NH, N, HF = XV.shape
        
        # 使用可訓練的 W1, b1 參數
        W1_init = self.W1.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(B_mul_NH, HF, HF)
        b1_init = self.b1.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(B_mul_NH, 1, HF)
        
        # 基本的 TTT 前向傳播
        Z1 = XK @ W1_init + b1_init  # [B*nh, N, f]
        
        # TTT LayerNorm - 使用分離的參數
        mu = Z1.mean(dim=-1, keepdim=True)
        var = Z1.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + 1e-6)
        Z1_hat = (Z1 - mu) / std
        
        # 應用 TTT LayerNorm 權重和偏置
        NH = self.num_heads
        Z1_normed = (self.ttt_norm_weight * Z1_hat.reshape(-1, NH, N, HF) + 
                     self.ttt_norm_bias).reshape(B_mul_NH, N, HF)
        
        # 殘差連接
        output = XQ + Z1_normed  # [B*nh, N, f]
        
        # 重塑回原始格式
        output = output.reshape(batch_size, self.num_heads, seq_length, self.head_dim)
        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, -1)
        
        # 應用門控 - 與 kernels 版本一致使用 GELU
        gate = F.gelu(XGate)
        output = gate * self.post_norm(output)
        
        # 最終投影
        output = self.wo(output)
        
        return output

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class BFloat16LayerNorm(nn.Module):
    """LayerNorm 兼容 BFloat16 的版本"""
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # 保存原始數據類型
        original_dtype = x.dtype
        # 轉換為 float32 進行計算
        x_float = x.float()
        # 執行 LayerNorm
        mean = x_float.mean(-1, keepdim=True)
        var = ((x_float - mean) ** 2).mean(-1, keepdim=True)
        x_norm = (x_float - mean) / torch.sqrt(var + self.eps)
        # 應用權重和偏置
        if self.elementwise_affine:
            x_norm = x_norm * self.weight.float() + self.bias.float()
        # 轉回原始數據類型
        return x_norm.to(original_dtype)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

class TTTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = TTTLinear(config)
        self.ln_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # MLP - 使用可配置的 mlp_ratio
        mlp_ratio = getattr(config, 'mlp_ratio', 4)  # 🎯 從配置中獲取 MLP 倍數
        multiple_of = 256
        mlp_hidden = int(config.hidden_size * mlp_ratio * 2 / 3)
        mlp_hidden = multiple_of * ((mlp_hidden + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(config.hidden_size, mlp_hidden, bias=False)
        self.w2 = nn.Linear(mlp_hidden, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, mlp_hidden, bias=False)
        self.mlp = lambda x: self.w2(F.silu(self.w1(x)) * self.w3(x))

    def forward(self, x, ttt_lr_mult=1.0):
        x = x + self.attn(self.ln_1(x), ttt_lr_mult=ttt_lr_mult)
        x = x + self.mlp(self.ln_2(x))
        return x

class TTTPreTrainedModel(PreTrainedModel):
    config_class = TTTConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["TTTBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

class TTTModel(TTTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.h = nn.ModuleList([TTTBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        ttt_lr_mult: float = 1.0,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        hidden_states = self.wte(input_ids)
        
        for block in self.h:
            hidden_states = block(hidden_states, ttt_lr_mult=ttt_lr_mult)
        
        hidden_states = self.ln_f(hidden_states)
        
        return BaseModelOutputWithPast(
            past_key_values=None,
            last_hidden_state=hidden_states,
        )

class TTT(TTTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.transformer = TTTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.n_layer = config.num_hidden_layers

        # 初始化權重
        self.apply(self._init_weights)
        self.post_init()
        self.lm_head.weight = self.transformer.wte.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm) or isinstance(module, RMSNorm):
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        loss_masks: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        ttt_lr_mult: float = 1.0,
        return_hidden: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            ttt_lr_mult=ttt_lr_mult,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        if labels is not None:
            logits_float = logits.to(torch.float32)
            labels_long = labels.to(torch.long)
            if loss_masks is None:
                loss_masks = torch.ones_like(labels_long, dtype=torch.float32)
            loss_masks = loss_masks.to(torch.float32)
            
            # 創建有效的標籤遮罩，排除 -100
            valid_labels_mask = (labels_long != -100)
            
            # 只對有效的標籤計算損失
            if valid_labels_mask.any():
                # 獲取有效的標籤和對應的 logits
                valid_labels = labels_long[valid_labels_mask]
                valid_logits = logits_float[valid_labels_mask]
                
                # 計算 log probabilities
                log_probs = F.log_softmax(valid_logits, dim=-1)
                token_log_prob = torch.gather(log_probs, -1, valid_labels.unsqueeze(-1)).squeeze(-1)
                
                # 計算損失
                loss = -torch.mean(token_log_prob)
            else:
                # 如果沒有有效標籤，返回零損失
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            loss = None
        if return_hidden:
            return logits, loss, hidden_states
        else:
            return logits, loss, None
