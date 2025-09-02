import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from .ttt import TTTLinear, TTTConfig
from mamba_ssm.ops.selective_scan_interface import selective_scan_online7_fn
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    RMSNorm = nn.LayerNorm


class PreCoNoGainConfig:
    def __init__(self,
                 vocab_size=50257,
                 d_model=512,
                 n_layer=12,
                 d_state=4,
                 d_conv=3,
                 expand=6,
                 ttt_num_heads=8,
                 ttt_num_layers=8,
                 dropout=0.1,
                 # 簡化版參數
                 longhorn_weight=0.7,  # Longhorn 分支的固定權重
                 ttt_weight=0.3):     # TTT 分支的固定權重
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.ttt_num_heads = ttt_num_heads
        self.ttt_num_layers = ttt_num_layers
        self.dropout = dropout
        self.longhorn_weight = longhorn_weight
        self.ttt_weight = ttt_weight


class Longhorn(nn.Module):
    """簡化版 Longhorn SSM 組件"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", 
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, 
                 dt_init_floor=1e-4, conv_bias=True, bias=False, use_fast_path=False, 
                 layer_idx=None, device=None, dtype=None):
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
        
        # 簡化卷積處理
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
        
        y = selective_scan_online7_fn(x, q.to(x), k.to(x), dt.to(x), 
                                      D=self.D.float(), t_bias=self.dt_head.bias.float(), z=z)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out, y

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)
        
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = x
        x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
        if self.conv1d.bias is not None:
            x = x + self.conv1d.bias
        x = self.act(x).to(dtype=dtype)
        
        x_db = self.x_proj(x)
        dt, k, q = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_head.weight)
        dt = torch.sigmoid(dt + self.dt_head.bias.to(dtype=dt.dtype))
        dt = dt / (1 + dt * k.square().sum(dim=-1, keepdim=True))
        
        dA = 1 - torch.einsum("bd,bn->bdn", dt, k.pow(2))
        dB = torch.einsum("bd,bn->bdn", dt, k)
        ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), q)
        y = y + self.D.to(dtype) * x
        y = y * self.act(z)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

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
            conv_state = torch.zeros(
                batch_size, self.d_model * self.expand, self.d_conv,
                device=self.conv1d.weight.device, dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size, self.d_model * self.expand, self.d_state,
                device=self.x_proj.weight.device, dtype=self.x_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class PreCoNoGainBlock(nn.Module):
    """簡化版 PreCo Block - 使用可學習權重而非複雜的 Kalman Gate"""
    def __init__(self, config: PreCoNoGainConfig, layer_idx=None):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        
        # Longhorn 預測分支
        self.longhorn_block = Longhorn(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            layer_idx=layer_idx,
        )
        
        # 降維層
        self.reduce_hidden = nn.Linear(config.d_model * config.expand, config.d_model, bias=False)
        
        # TTT 校正分支 - 修正：作為單層使用
        ttt_cfg = TTTConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.d_model,
            num_attention_heads=config.ttt_num_heads,
            num_hidden_layers=1,  # 🔧 修正：每個 TTTLinear 只是單層
            pre_conv=False,
        )
        self.corr_linear = TTTLinear(ttt_cfg)
        
        # 統一的輸出投影
        self.q_network = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # 🎯 簡化的權重機制 - 三種選擇
        self.weight_type = "adaptive"  # "fixed", "learnable", "adaptive"
        
        if self.weight_type == "fixed":
            # 選項1: 固定權重
            self.register_buffer('longhorn_weight', torch.tensor(config.longhorn_weight))
            self.register_buffer('ttt_weight', torch.tensor(config.ttt_weight))
        elif self.weight_type == "learnable":
            # 選項2: 可學習的全局權重
            self.longhorn_weight = nn.Parameter(torch.tensor(config.longhorn_weight))
            self.ttt_weight = nn.Parameter(torch.tensor(config.ttt_weight))
        elif self.weight_type == "adaptive":
            # 選項3: 自適應權重網絡（比原版簡單）
            hidden_dim = config.d_model // 4
            
            # 上下文降維層（處理 3*d_model → d_model）
            self.context_reducer = nn.Linear(3 * config.d_model, config.d_model)
            
            # 權重預測網絡
            self.weight_net = nn.Sequential(
                nn.Linear(config.d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),  # 防止過擬合
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2),  # 輸出 [longhorn_weight, ttt_weight]
                nn.Softmax(dim=-1)
            )
            
            # 初始化：讓初始權重接近 [0.7, 0.3]
            with torch.no_grad():
                # 調整最後一層的 bias，使初始輸出偏向 [0.7, 0.3]
                self.weight_net[-2].bias.data = torch.tensor([0.5, -0.5])  # logits before softmax
        
        # 初始化權重
        self._init_weights()

    def _init_weights(self):
        """權重初始化"""
        if self.weight_type == "learnable":
            # 確保權重和為1
            with torch.no_grad():
                total = self.longhorn_weight + self.ttt_weight
                self.longhorn_weight.data /= total
                self.ttt_weight.data /= total

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.longhorn_block.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, x, residual=None, inference_params=None):
        residual = (x + residual) if residual is not None else x
        x_norm = self.norm(residual)
        
        # 1. Longhorn 預測分支
        h_out, h_hidden = self.longhorn_block(x_norm, inference_params=inference_params)
        h_hidden_reduced = self.reduce_hidden(h_hidden)  # [B, L, d_model]
        
        # 2. TTT 校正分支
        c_hidden = self.corr_linear(h_hidden_reduced)
        
        # 3. 統一輸出投影
        z_longhorn = self.q_network(h_hidden_reduced)
        z_ttt = self.q_network(c_hidden)
        
        # 4. 簡化的權重融合
        if self.weight_type == "fixed" or self.weight_type == "learnable":
            # 全局權重融合
            out = self.longhorn_weight * z_longhorn + self.ttt_weight * z_ttt
            # 統計信息
            weight_stats = {
                'longhorn_weight': self.longhorn_weight.item(),
                'ttt_weight': self.ttt_weight.item(),
                'weight_mean': 0.5,  # 固定值
                'weight_std': 0.0,   # 固定值
            }
        elif self.weight_type == "adaptive":
            # 自適應權重融合 - 使用更豐富的上下文信息
            B, L, D = h_hidden_reduced.shape
            
            # 🎯 智能上下文聚合：結合平均值、最大值和最後一個 token
            context_mean = h_hidden_reduced.mean(dim=1)  # [B, D] 全局平均
            context_max = h_hidden_reduced.max(dim=1)[0]  # [B, D] 最大激活
            context_last = h_hidden_reduced[:, -1, :]     # [B, D] 最後一個 token
            
            # 組合上下文信息
            context_combined = torch.cat([
                context_mean,
                context_max,
                context_last
            ], dim=-1)  # [B, 3*D]
            
            # 降維到原始維度
            context_reduced = self.context_reducer(context_combined)
            
            # 計算自適應權重
            weights = self.weight_net(context_reduced)  # [B, 2]
            longhorn_w = weights[:, 0:1].unsqueeze(1)  # [B, 1, 1]
            ttt_w = weights[:, 1:2].unsqueeze(1)       # [B, 1, 1]
            
            out = longhorn_w * z_longhorn + ttt_w * z_ttt
            
            # 統計信息
            weight_stats = {
                'longhorn_weight': longhorn_w.mean().item(),
                'ttt_weight': ttt_w.mean().item(),
                'weight_mean': weights.mean().item(),
                'weight_std': weights.std().item(),
                'weight_entropy': -torch.sum(weights * torch.log(weights + 1e-8), dim=1).mean().item(),  # 權重分佈的熵
            }
        
        return out, residual, weight_stats


class PreCoNoGainModel(nn.Module):
    """簡化版 PreCo 模型 - 無複雜 Kalman Gate"""
    def __init__(self, config: PreCoNoGainConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            PreCoNoGainBlock(config, layer_idx=i) 
            for i in range(config.n_layer)
        ])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 權重共享
        self.lm_head.weight = self.embedding.weight
        
        self.dropout = nn.Dropout(config.dropout)
        self.config = config
        
        # 權重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """正確的權重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            if hasattr(module, 'weight') and module.weight is not None:
                torch.nn.init.ones_(module.weight)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: block.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, block in enumerate(self.blocks)
        }

    def forward(self, input_ids, targets=None, inference_params=None, ttt_lr_mult=None, compute_branch_loss=False):
        x = self.embedding(input_ids)
        residual = None
        
        # 收集權重統計
        all_weight_stats = []
        final_h_hidden = None
        
        for i, block in enumerate(self.blocks):
            block_infer = None
            if inference_params is not None and hasattr(inference_params, 'key_value_memory_dict'):
                block_infer = inference_params
            
            x, residual, weight_stats = block(x, residual=residual, inference_params=block_infer)
            all_weight_stats.append(weight_stats)
            
            # 保存最後一層的 hidden state（用於分支 loss 計算）
            if compute_branch_loss and i == len(self.blocks) - 1:
                final_h_hidden = block.longhorn_block(block.norm(residual if residual is not None else x))[1]
        
        # 最後一層處理
        x = self.norm_f((x + residual) if residual is not None else x)
        x = self.dropout(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            # 主要損失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # 計算權重統計
            longhorn_weights = [stats['longhorn_weight'] for stats in all_weight_stats]
            ttt_weights = [stats['ttt_weight'] for stats in all_weight_stats]
            
            # 可選的分支損失計算
            if compute_branch_loss and final_h_hidden is not None:
                final_block = self.blocks[-1]
                final_h_hidden_reduced = final_block.reduce_hidden(final_h_hidden)
                final_c_hidden = final_block.corr_linear(final_h_hidden_reduced)
                
                z_longhorn = final_block.q_network(final_h_hidden_reduced)
                z_ttt = final_block.q_network(final_c_hidden)
                
                longhorn_logits = self.lm_head(z_longhorn)
                longhorn_loss = F.cross_entropy(longhorn_logits.view(-1, longhorn_logits.size(-1)), targets.view(-1))
                del longhorn_logits
                
                ttt_logits = self.lm_head(z_ttt)
                ttt_loss = F.cross_entropy(ttt_logits.view(-1, ttt_logits.size(-1)), targets.view(-1))
                del ttt_logits
                
                del z_longhorn, z_ttt, final_h_hidden_reduced, final_c_hidden
            else:
                longhorn_loss = torch.tensor(0.0, device=logits.device)
                ttt_loss = torch.tensor(0.0, device=logits.device)
            
            # 返回損失字典
            loss_dict = {
                'total_loss': loss,
                'ce_loss': loss.item(),
                'longhorn_loss': longhorn_loss.item() if hasattr(longhorn_loss, 'item') else longhorn_loss,
                'ttt_loss': ttt_loss.item() if hasattr(ttt_loss, 'item') else ttt_loss,
                'longhorn_weight_mean': sum(longhorn_weights) / len(longhorn_weights),
                'ttt_weight_mean': sum(ttt_weights) / len(ttt_weights),
                'weight_mean': sum([stats['weight_mean'] for stats in all_weight_stats]) / len(all_weight_stats),
                'weight_std': sum([stats['weight_std'] for stats in all_weight_stats]) / len(all_weight_stats),
                'weight_entropy': sum([stats.get('weight_entropy', 0.0) for stats in all_weight_stats]) / len(all_weight_stats),
            }
            
            return logits, loss_dict
        
        return logits


# 工廠函數
def create_preco_nogain_model(
    vocab_size: int = 50257,
    d_model: int = 512,
    n_layer: int = 12,
    d_state: int = 4,
    expand: int = 6,
    ttt_num_heads: int = 8,
    ttt_num_layers: int = 8,
    longhorn_weight: float = 0.7,
    ttt_weight: float = 0.3,
    **kwargs
) -> PreCoNoGainModel:
    """創建簡化版 PreCo 模型"""
    config = PreCoNoGainConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layer=n_layer,
        d_state=d_state,
        expand=expand,
        ttt_num_heads=ttt_num_heads,
        ttt_num_layers=ttt_num_layers,
        longhorn_weight=longhorn_weight,
        ttt_weight=ttt_weight,
        **kwargs
    )
    return PreCoNoGainModel(config) 