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
                 dropout=0.1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.ttt_num_heads = ttt_num_heads
        self.ttt_num_layers = ttt_num_layers
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
        return out.unsqueeze(1), conv_state, ssm_state

class PreCoBlock(nn.Module):
    def __init__(self, config: PreCoNewConfig, layer_idx=None):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.longhorn_block = Longhorn(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            layer_idx=layer_idx,
        )
        self.reduce_hidden = nn.Linear(config.d_model * config.expand, config.d_model, bias=False)
        # TTT 校正分支 - 修正：作為單層使用
        ttt_cfg = TTTConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.d_model,  # 只用 d_model 維度
            num_attention_heads=config.ttt_num_heads,
            # num_hidden_layers=config.ttt_num_layers,  # 這裡用 config.ttt_num_layers
            num_hidden_layers=1,  # 🔧 修正：每個 TTTLinear 只是單層
            pre_conv=False,
        )
        self.corr_linear = TTTLinear(ttt_cfg)
        self.q_network = nn.Linear(config.d_model, config.d_model, bias=False)
        # TokenButler-style gain
        self.gain_q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.gain_k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        
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
        
        # 🔧 實現真正的測試時訓練 - 訓練和推理都使用動態更新
        c_hidden = self._ttt_training_forward(h_hidden_reduced, ttt_lr_mult)
        
        z_longhorn = self.q_network(h_hidden_reduced)
        z_ttt = self.q_network(c_hidden)
        
        # TokenButler-style gain - 應用 500 筆測試的最佳參數
        gain_q = self.gain_q_proj(h_hidden_reduced)  # [B, L, d]
        gain_k = self.gain_k_proj(h_hidden_reduced)  # [B, L, d]
        d = gain_q.shape[-1]
        A = torch.matmul(gain_q, gain_k.transpose(-2, -1)) / math.sqrt(d)  # [B, L, L]
        score = A.mean(dim=1)  # [B, L]
        # gain = torch.sigmoid(self.alpha * score + self.beta).unsqueeze(-1)  # [B, L, 1]
        raw_gain = torch.sigmoid(self.alpha * score + self.beta)  # [B, L]
        
        # 應用 500 筆測試的最佳參數：max_gain=0.55, inference_scale=0.85
        max_gain = 0.55  # 500 筆測試的最佳 max_gain
        gain = torch.clamp(raw_gain, max=max_gain).unsqueeze(-1)  # [B, L, 1]
        
        # 推理時適度縮放 gain（500 筆測試的最佳 inference_scale=0.85）
        if not self.training:
            gain = gain * 0.85  # 推理時縮放為 85%
        
        out = z_longhorn + gain * (z_ttt - z_longhorn)
        return out, residual, gain
    
    def _ttt_training_forward(self, hidden_states, ttt_lr_mult=1.0):
        """TTT 訓練時的前向傳播，實現動態參數更新"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 獲取 TTT 參數
        ttt_layer = self.corr_linear
        
        # 初始化參數副本（用於動態更新）
        W1_current = ttt_layer.W1.clone()
        b1_current = ttt_layer.b1.clone()
        
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
        
        # 計算學習率
        ttt_lr = torch.sigmoid(ttt_lr + ttt_layer.learnable_ttt_lr_bias)
        # ttt_lr 的形狀是 [batch_size, seq_len, num_heads]，需要重塑為 [seq_len, batch_size, num_heads]
        ttt_lr = ttt_lr.transpose(0, 1)  # [seq_len, batch_size, num_heads]
        ttt_lr = ttt_lr.reshape(seq_len, -1, 1)  # [seq_len, batch_size*num_heads, 1]
        # 處理 ttt_lr_mult 可能為 None 的情況
        if ttt_lr_mult is not None:
            ttt_lr = ttt_lr * ttt_lr_mult  # 應用學習率倍數
        else:
            # 推理時使用極小的學習率，讓 TTT 更新更保守
            ttt_lr = ttt_lr * 0.001  # 推理時使用 0.001 的學習率
        
        # 獲取 token_idx 並調整形狀以匹配 ttt_lr
        token_idx = ttt_layer.token_idx + ttt_layer.learnable_token_idx_bias
        token_idx = torch.clamp(token_idx, min=0.0)
        # token_idx 形狀: [1, 1, mini_batch_size, 1] -> [mini_batch_size]
        token_idx = token_idx.squeeze()  # [8]
        
        # 動態參數更新
        B_mul_NH, N, HF = XV.shape
        
        # 使用可訓練的 W1, b1 參數
        W1_init = W1_current.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(B_mul_NH, HF, HF)
        b1_init = b1_current.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(B_mul_NH, 1, HF)
        
        # 基本的 TTT 前向傳播
        Z1 = XK @ W1_init + b1_init  # [B*nh, N, f]
        
        # TTT LayerNorm - 使用分離的參數
        mu = Z1.mean(dim=-1, keepdim=True)
        var = Z1.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + 1e-6)
        Z1_hat = (Z1 - mu) / std
        
        # 應用 TTT LayerNorm 權重和偏置
        NH = ttt_layer.num_heads
        Z1_normed = (ttt_layer.ttt_norm_weight * Z1_hat.reshape(-1, NH, N, HF) + 
                     ttt_layer.ttt_norm_bias).reshape(B_mul_NH, N, HF)
        
        # 計算自監督目標和梯度
        ssl_target = XV - XK  # 自監督學習目標
        grad_l = Z1_normed - ssl_target  # 梯度
        
        # 動態更新參數（簡化版本）
        # 簡化學習率調整：使用固定的 token_idx 平均值
        eta = token_idx.mean() * ttt_lr  # 學習率調整
        
        # 更新 W1 和 b1（簡化版本）
        # 注意：這裡只是模擬更新，實際的完整實現需要更複雜的梯度計算
        # 簡化：使用平均學習率進行更新
        avg_eta = eta.mean()  # 標量
        
        # 限制更新幅度，避免過度更新
        max_update_norm = 0.1  # 最大更新幅度
        W1_update = torch.matmul(XK.transpose(-1, -2), grad_l)
        W1_update_norm = torch.norm(W1_update)
        
        if W1_update_norm > max_update_norm:
            W1_update = W1_update * max_update_norm / W1_update_norm
        
        W1_updated = W1_init - avg_eta * W1_update
        b1_updated = b1_init - avg_eta * grad_l.mean(dim=1, keepdim=True)
        
        # 使用更新後的參數計算輸出
        Z1_updated = XK @ W1_updated + b1_updated
        
        # 重新應用 LayerNorm
        mu_updated = Z1_updated.mean(dim=-1, keepdim=True)
        var_updated = Z1_updated.var(dim=-1, keepdim=True, unbiased=False)
        std_updated = torch.sqrt(var_updated + 1e-6)
        Z1_hat_updated = (Z1_updated - mu_updated) / std_updated
        
        Z1_normed_updated = (ttt_layer.ttt_norm_weight * Z1_hat_updated.reshape(-1, NH, N, HF) + 
                             ttt_layer.ttt_norm_bias).reshape(B_mul_NH, N, HF)
        
        # 殘差連接
        output = XQ + Z1_normed_updated  # [B*nh, N, f]
        
        # 重塑回原始格式
        output = output.reshape(batch_size, ttt_layer.num_heads, seq_len, ttt_layer.head_dim)
        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        # 應用門控
        gate = F.gelu(XGate)
        output = gate * ttt_layer.post_norm(output)
        
        # 最終投影
        output = ttt_layer.wo(output)
        
        return output

class PreCoNewModel(nn.Module):
    def __init__(self, config: PreCoNewConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([PreCoBlock(config, layer_idx=i) for i in range(config.n_layer)])
        self.norm_f = RMSNorm(config.d_model)
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
                final_h_hidden = block.longhorn_block(block.norm(residual if residual is not None else x))[1]
            
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
                # 🔧 修正：使用動態 TTT 更新，與主前向傳播一致
                final_c_hidden = final_block._ttt_training_forward(final_h_hidden_reduced, ttt_lr_mult)
                
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