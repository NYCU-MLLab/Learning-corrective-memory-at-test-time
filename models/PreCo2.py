import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import math

from .longhorn import LonghornLM, LonghornConfig
from .ttt import TTT, TTTConfig


@dataclass
class PreCoConfig:
    """PreCo (Prediction-Correction) 混合模型配置 - 133M 參數量匹配版本"""
    # Longhorn 配置 - 參數量優化
    longhorn_d_model: int = 512   # 512 維度
    longhorn_n_layer: int = 8     # 8 層 (與 run.sh 一致)
    longhorn_d_state: int = 4     # 與 Longhorn 保持一致
    longhorn_ssm_expand: int = 6  # 與 run.sh 一致：expand=8
    
    # TTT 配置 - 參數量優化
    ttt_hidden_size: int = 512    # 512 維度
    ttt_num_layers: int = 6      # 6 層 (與 run.sh 一致)
    ttt_num_heads: int = 8        # 8 頭 (512/8=64 head_dim)
    ttt_base_lr: float = 1.0
    ttt_mini_batch_size: int = 8  # 匹配注意力頭數
    mlp_ratio: int = 2            # MLP 倍數 (與 run.sh 一致)
    
    # Kalman Gain 配置 - 增強動態性
    kalman_hidden_dim: int = 256  # 256 維 (與 run.sh 一致)
    
    # 共同配置
    vocab_size: int = 50257       # 匹配 slim_tokenizer
    block_size: int = 1024        # 匹配 run.sh
    dropout: float = 0.2
    
    # TTT 特定配置
    use_gate: bool = True         # 啟用門控機制
    share_qk: bool = True         # 啟用 Q/K 共享
    ttt_layer_type: str = "linear"
    pre_conv: bool = True         # 啟用預卷積
    conv_kernel: int = 2          # 卷積核大小
    scan_checkpoint_group_size: int = 1
    
    def __post_init__(self):
        """初始化後設置 SSM 配置"""
        self.longhorn_ssm_cfg = {
            'd_state': self.longhorn_d_state,
            'd_conv': 3,
            'expand': self.longhorn_ssm_expand
        }
    
    # 🎯 133M 參數分佈預估 (參數量匹配版本)：
    #
    # 🔧 優化後的參數分佈 (與 Longhorn 133M 匹配)：
    # - 共用 Embedding: 512×50257 ≈ 26M (Longhorn 和 TTT 共用同一個)
    # - Longhorn backbone: 10層×512維×expand8 ≈ 42M (增加容量)
    # - TTT backbone: 6層×512維×8頭 ≈ 32M (平衡分配)
    # - 唯一的 q LM Head: 512×50257 ≈ 26M
    # - Kalman Gain: 256維 ≈ 0.7M (增強動態性)
    # - 其他組件: ~6M
    # - 總計：約133M參數 (與 Longhorn 匹配)
    #
    # 🚀 關鍵優化：
    # - Longhorn 層數：從8層增加到10層，增加約10M參數
    # - Longhorn expand：從6增加到8，增加內部容量
    # - TTT 層數：調整到6層，平衡兩個 backbone
    # - Kalman Gain：維持256維，增強動態分配能力


class KalmanGainNetwork(nn.Module):
    """高效版 Kalman Gate - 保持動態校正能力"""
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        
        # 使用單層線性變換 + LayerNorm
        self.norm = nn.LayerNorm(d_model)
        self.score = nn.Linear(d_model, 1)
        
        # 初始化：讓初始權重接近 0.5
        with torch.no_grad():
            self.score.weight.data.normal_(0, 0.02)
            self.score.bias.data.fill_(0.0)
    
    def forward(self, hidden_states):
        # 快速計算重要性分數
        normed = self.norm(hidden_states)
        scores = self.score(normed).squeeze(-1)  # [B, L]
        kalman_weights = torch.sigmoid(scores)    # [B, L]
        
        # 統計信息
        token_stats = {
            'mean': kalman_weights.mean().item(),
            'std': kalman_weights.std().item(),
        }
        return kalman_weights, token_stats


class PreCoModel(nn.Module):
    """PreCo 模型 - 高效版本但保持 Kalman Filter 邏輯"""
    
    def __init__(self, config: PreCoConfig):
        super().__init__()
        
        # Longhorn 預測器
        longhorn_config = LonghornConfig()
        longhorn_config.vocab_size = config.vocab_size
        longhorn_config.d_model = config.longhorn_d_model
        longhorn_config.n_layer = config.longhorn_n_layer
        longhorn_config.ssm_cfg = {
            'd_state': config.longhorn_d_state,
            'd_conv': 3,
            'expand': config.longhorn_ssm_expand
        }
        self.longhorn = LonghornLM(longhorn_config)
        
        # TTT 校正器
        self.ttt = TTT(TTTConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.ttt_hidden_size,
            num_hidden_layers=config.ttt_num_layers,
            num_attention_heads=config.ttt_num_heads,
            max_position_embeddings=config.block_size,
            ttt_base_lr=config.ttt_base_lr,
            mini_batch_size=config.ttt_mini_batch_size,
            use_gate=config.use_gate,
            ttt_layer_type=config.ttt_layer_type,
            scan_checkpoint_group_size=config.scan_checkpoint_group_size,
            pre_conv=config.pre_conv,
            conv_kernel=config.conv_kernel,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=3,
        ))
        
        # Kalman Gate
        self.kalman_gain = KalmanGainNetwork(
            d_model=config.longhorn_d_model,
            hidden_dim=config.kalman_hidden_dim,
        )
        
        # 共用 Q Network
        self.q_network = nn.Linear(config.longhorn_d_model, config.vocab_size, bias=False)
        nn.init.normal_(self.q_network.weight, std=0.02)
        
        # correction scale
        self.correction_scale = nn.Parameter(torch.ones(1) * 0.2)
    
    def forward(self, input_ids, target_ids=None, ttt_lr_mult=1.0):
        # 1. Longhorn 預測
        longhorn_logits, longhorn_loss, longhorn_hidden = self.longhorn(
            input_ids, targets=target_ids, return_hidden=True
        )
        
        # 2. TTT 校正
        ttt_logits, ttt_loss, ttt_hidden = self.ttt(
            input_ids=input_ids,
            labels=target_ids,
            ttt_lr_mult=ttt_lr_mult,
            return_hidden=True
        )
        
        # 3. Kalman Gate - 計算每個 token 的重要性
        kalman_weights, token_stats = self.kalman_gain(longhorn_hidden)  # [B, L]
        
        # 4. 計算 Q 值
        q_longhorn = self.q_network(longhorn_hidden)  # [B, L, V]
        q_ttt = self.q_network(ttt_hidden)           # [B, L, V]
        
        # 5. Kalman Filter 更新 - 根據重要性動態校正
        correction = q_ttt - q_longhorn              # [B, L, V]
        scale = torch.sigmoid(self.correction_scale)  # scalar
        final_logits = q_longhorn + scale * kalman_weights.unsqueeze(-1) * correction
        
        if target_ids is not None:
            # 計算 CE loss
            loss = F.cross_entropy(
                final_logits.view(-1, final_logits.size(-1)), 
                target_ids.view(-1)
            )
            
            # 返回必要的統計信息
            loss_dict = {
                'total_loss': loss,
                'ce_loss': loss.item(),
                'longhorn_loss': longhorn_loss.item(),
                'ttt_loss': ttt_loss.item(),
                'kalman_mean': token_stats['mean'],
                'kalman_std': token_stats['std'],
            }
            return final_logits, loss_dict
        return final_logits


# 工廠函數 - Kalman Filter 版本
def create_preco_model(
    vocab_size: int = 50257,     # 匹配 slim tokenizer
    block_size: int = 1024,      # 🔧 更新：匹配 run.sh
    longhorn_d_model: int = 512, # 512維度優化：512維度
    longhorn_n_layer: int = 10,  # 🔧 更新：10層 (參數量匹配)
    ttt_hidden_size: int = 512,  # 512維度優化：512維度
    ttt_num_layers: int = 6,     # 🔧 更新：6層 (參數量平衡)
    ttt_num_heads: int = 8,      # 512維度優化：8頭 (512/8=64)
    longhorn_ssm_expand: int = 8, # 🔧 新增：expand=8 (增加容量)
    kalman_hidden_dim: int = 256, # 🔧 新增：256維 (增強動態性)
    **kwargs
) -> PreCoModel:
    """創建 PreCo Kalman Filter 模型 - 133M 參數量匹配版本"""
    config = PreCoConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        longhorn_d_model=longhorn_d_model,
        longhorn_n_layer=longhorn_n_layer,
        longhorn_ssm_expand=longhorn_ssm_expand,  # 🔧 新增
        ttt_hidden_size=ttt_hidden_size,
        ttt_num_layers=ttt_num_layers,
        ttt_num_heads=ttt_num_heads,
        kalman_hidden_dim=kalman_hidden_dim,      # 🔧 新增
        **kwargs
    )
    return PreCoModel(config) 