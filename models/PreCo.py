import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
try:
    from .ttt import TTTLinear, TTTConfig
except ImportError:
    from ttt import TTTLinear, TTTConfig
from mamba_ssm.ops.selective_scan_interface import selective_scan_online7_fn
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    RMSNorm = nn.LayerNorm

class PreCoNewConfig:
    def __init__(self,
                 vocab_size=50257,
                 d_model=512,
                 n_layer=12,
                 d_state=8,
                 d_conv=3,
                 expand=6,
                 ttt_num_heads=8,
                 ttt_num_layers=1,  # 新增參數，預設 4 層
                 mini_batch_size=8,  # TTT mini batch size
                 dropout=0.1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.ttt_num_heads = ttt_num_heads
        self.ttt_num_layers = ttt_num_layers
        self.mini_batch_size = mini_batch_size
        self.dropout = dropout

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
                out, _, _, y_step = self.step(hidden_states, conv_state, ssm_state)
                return out, y_step
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
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
        conv_state[:, :, -1] = x
        x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
        if self.conv1d.bias is not None:
            x = x + self.conv1d.bias
        x = self.act(x).to(dtype=dtype)
        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, k, q = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_head.weight)  # (B d_inner)
        dt = torch.sigmoid(dt + self.dt_head.bias.to(dtype=dt.dtype))
        dt = dt / (1 + dt * k.square().sum(dim=-1, keepdim=True))
        dA = 1 - torch.einsum("bd,bn->bdn", dt, k.pow(2))
        dB = torch.einsum("bd,bn->bdn", dt, k)
        ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), q)
        y = y + self.D.to(dtype) * x
        y = y * self.act(z)  # (B D)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state, y.unsqueeze(1)

class PreCoBlock(nn.Module):
    def __init__(self, config: PreCoNewConfig, layer_idx=None):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model, eps=1e-6)
        self.longhorn_block = Longhorn(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            layer_idx=layer_idx,
        )
        self.reduce_hidden = nn.Linear(config.d_model * config.expand, config.d_model, bias=False)
        # TTT 校正分支 - 改為直接使用 ttt.py 的 TTTLinear forward
        ttt_cfg = TTTConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.d_model,  # 只用 d_model 維度
            intermediate_size=config.d_model * 4,  # 🔧 新增：intermediate_size
            num_attention_heads=config.ttt_num_heads,
            # num_hidden_layers=config.ttt_num_layers,  # 這裡用 config.ttt_num_layers
            num_hidden_layers=1,  # 🔧 修正：每個 TTTLinear 只是單層
            pre_conv=False,
            mini_batch_size=config.mini_batch_size,  # 🔧 新增：mini_batch_size
            mlp_ratio=4,  # 🔧 新增：mlp_ratio
        )
        self.corr_linear = TTTLinear(ttt_cfg, layer_idx=layer_idx)
        self.q_network = nn.Linear(config.d_model, config.d_model, bias=False)
        # TokenButler-style gain
        self.gain_q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.gain_k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        # Local attention 參數
        self.local_attn_window = 5
        self.gain_type = 'local_attn'
        # 暫存本層最新一次 forward 的隱狀態（供最後一層分支 loss 使用）
        self._last_h_hidden = None
        self._last_h_hidden_reduced = None
        self._last_c_hidden = None
        
        # 初始化 alpha 和 beta 為更穩定的值
        with torch.no_grad():
            self.alpha.data.fill_(1.0)
            self.beta.data.fill_(0.0)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.longhorn_block.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, x, residual=None, inference_params=None, ttt_lr_mult=1.0):
        residual = (x + residual) if residual is not None else x
        x_norm = self.norm(residual)
        h_out, h_hidden = self.longhorn_block(x_norm, inference_params=inference_params)
        h_hidden_reduced = self.reduce_hidden(h_hidden)  # [B, L, d_model]
        # 暫存當前層的隱狀態
        self._last_h_hidden = h_hidden
        self._last_h_hidden_reduced = h_hidden_reduced
        
        # 🔧 使用 ttt.py 的 TTTLinear forward 作為校正分支（獨立於 Longhorn，直接作用於 x_norm）
        # ✨ 新版：使用 TTT 原生的自適應學習率，不再外部調整 ttt_base_lr
        B, L, _ = x_norm.shape
        position_ids = torch.arange(0, L, device=x_norm.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        
        # 🚀 直接使用 TTT 原生的自適應學習率機制，移除外部 ttt_lr_mult 干預
        # TTT 內部會根據輸入內容和可學習參數動態計算最適合的學習率
        c_hidden = self.corr_linear(
            hidden_states=x_norm,
            position_ids=position_ids,
            cache_params=None,
            return_pre_o_proj=True,
        )
        
        z_longhorn = self.q_network(h_hidden_reduced)
        z_ttt = self.q_network(c_hidden)
        # 暫存 TTT 分支隱狀態
        self._last_c_hidden = c_hidden
        
        # 🔧 Local attention 的 gain（window size = 5）：
        # 每個 token 基於自己的 local attention score 計算 gate，
        # 不使用全局聚合，實現真正的 token-level 動態 gating
        gain_type = getattr(self, 'gain_type', 'local_attn')
        if gain_type == 'local_attn':
            Bq, Lq, Cq = x_norm.shape
            q = self.gain_q_proj(x_norm)  # [B, L, Dg]
            k = self.gain_k_proj(x_norm)  # [B, L, Dg]
            d_g = q.shape[-1]
            Bsz = q.size(0)
            w = int(self.local_attn_window)
            window_size = 2 * w + 1
            center = torch.arange(Lq, device=q.device).unsqueeze(1)  # [L,1]
            offsets = torch.arange(-w, w + 1, device=q.device).unsqueeze(0)  # [1,window]
            idx = (center + offsets).clamp(0, Lq - 1)  # [L,window]
            idx = idx.unsqueeze(0).expand(Bsz, -1, -1)  # [B, L, window]
            batch_ids = torch.arange(Bsz, device=q.device).view(Bsz, 1, 1).expand(Bsz, Lq, window_size)
            k_windows = k[batch_ids, idx, :]  # [B, L, window, Dg]
            scores = (q.unsqueeze(2) * k_windows).sum(dim=-1) / math.sqrt(d_g)  # [B, L, window]
            attn_weights = torch.softmax(scores, dim=-1)  # [B, L, window]
            
            # 🔧 修改：每個 token 只取自己的 local attention 統計，不做全局聚合
            # 方案：取每個 token 的注意力權重的最大值作為 gate 強度
            # 邏輯：如果某個 token 在其窗口內有很強的注意力焦點，說明需要更多校正
            local_max_attn = attn_weights.max(dim=-1)[0]  # [B, L] 每個token的最大注意力權重
            gain = torch.sigmoid(local_max_attn * 4.0 - 2.0).unsqueeze(-1)  # [B, L, 1] 調整範圍到合理區間
        else:
            gain = torch.ones_like(h_hidden_reduced[:, :, :1]) * 0.5
        
        out = z_longhorn + gain * (z_ttt - z_longhorn)
        return out, residual, gain
    
    # 舊版自寫 TTT 更新邏輯已移除，統一改為直接使用 TTTLinear.forward
    


class PreCoNewModel(nn.Module):
    def __init__(self, config: PreCoNewConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([PreCoBlock(config, layer_idx=i) for i in range(config.n_layer)])
        self.norm_f = nn.LayerNorm(config.d_model, eps=1e-6)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.dropout = nn.Dropout(config.dropout)
        self.config = config
        
        # 添加權重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """正確的權重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
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
        
        # 收集 Kalman Gain 統計
        all_gains = []
        final_h_hidden = None  # 保存最後一層的 h_hidden
        
        for i, block in enumerate(self.blocks):
            block_infer = None
            if inference_params is not None and hasattr(inference_params, 'key_value_memory_dict'):
                block_infer = inference_params
            x, residual, gain = block(x, residual=residual, inference_params=block_infer, ttt_lr_mult=ttt_lr_mult)
            
            # 收集統計值
            all_gains.append(gain)
            
            # 只在需要時保存最後一層的 h_hidden
            if compute_branch_loss and i == len(self.blocks) - 1:
                # 直接使用在 block.forward 暫存的隱狀態，避免重跑 Longhorn
                final_h_hidden = block._last_h_hidden
            
        # 最後一層 RMSNorm
        x = self.norm_f((x + residual) if residual is not None else x)
        x = self.dropout(x)
        logits = self.lm_head(x)
        
        # 保存 gain 統計供測試使用
        if not self.training and all_gains:
            try:
                all_gains_tensor = torch.cat([g.view(-1, 1) for g in all_gains], dim=0)
                self.last_gains = all_gains_tensor
            except Exception as e:
                first_gain = all_gains[0].view(-1)
                self.last_gains = first_gain
        
        if targets is not None:
            # 主要 loss
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            
            # 可選的分支 loss 計算（記憶體密集）
            if compute_branch_loss and final_h_hidden is not None:
                final_block = self.blocks[-1]
                final_h_hidden_reduced = final_block.reduce_hidden(final_h_hidden)
                # 🔧 使用 ttt.py 的 TTTLinear forward 作為分支校正輸出
                # ✨ 新版：使用 TTT 原生自適應學習率，不干預 ttt_base_lr
                B2, L2, _ = final_h_hidden_reduced.shape
                position_ids2 = torch.arange(0, L2, device=final_h_hidden_reduced.device, dtype=torch.long).unsqueeze(0).expand(B2, -1)
                
                # 🚀 直接使用 TTT 原生機制，讓其自適應調整學習率
                final_c_hidden = final_block.corr_linear(
                    hidden_states=final_h_hidden_reduced,
                    position_ids=position_ids2,
                    cache_params=None,
                    return_pre_o_proj=True,
                )
                
                # 分支輸出
                z_longhorn = final_block.q_network(final_h_hidden_reduced)
                z_ttt = final_block.q_network(final_c_hidden)
                
                # 節省內存：逐個計算分支 loss
                longhorn_logits = self.lm_head(z_longhorn)
                # longhorn_loss = F.cross_entropy(longhorn_logits.view(-1, longhorn_logits.size(-1)), targets.view(-1))
                longhorn_loss = F.cross_entropy(longhorn_logits.view(-1, longhorn_logits.size(-1)), targets.view(-1), ignore_index=-100)
                del longhorn_logits
                
                ttt_logits = self.lm_head(z_ttt)
                # ttt_loss = F.cross_entropy(ttt_logits.view(-1, ttt_logits.size(-1)), targets.view(-1))
                ttt_loss = F.cross_entropy(ttt_logits.view(-1, ttt_logits.size(-1)), targets.view(-1), ignore_index=-100)
                del ttt_logits
                
                # 清理中間變量
                del z_longhorn, z_ttt, final_h_hidden_reduced, final_c_hidden
            else:
                # 輕量級：不計算分支 loss，節省記憶體
                longhorn_loss = torch.tensor(0.0, device=logits.device)
                ttt_loss = torch.tensor(0.0, device=logits.device)
            
            # 輕量級的 Kalman Gain 統計（始終計算）
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
                'longhorn_loss': longhorn_loss,
                'ttt_loss': ttt_loss,
                'kalman_mean': kalman_mean,
                'kalman_std': kalman_std,
            }
            return logits, loss_dict
        else:
            return logits 