"""
line 59, 467
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""
import os
import time
import inspect
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.nn import Linear as nnLinear
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.models.config_mamba import MambaConfig
from models import *
from models.ttt import TTTLinear, TTTConfig, TTTForCausalLM
from models.PreCo import PreCoNewConfig, PreCoNewModel
from models.preco_nogain import PreCoNoGainConfig, PreCoNoGainModel
import wandb

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
# master_seed = 1337
# out_dir = 'results'
# eval_interval = 2000
# log_interval = 1
# eval_iters = 200
# eval_only = False # if True, script exits right after the first eval
# always_save_checkpoint = False # if True, always save a checkpoint after each eval
master_seed = 1337
out_dir = 'results/slim_results'
eval_interval = 500
log_interval = 10
eval_iters = 50
eval_only = False
always_save_checkpoint = False

# model
model_name = "preco" # llama, gla, rwkv, retnet, mamba, longhorn, ttt, preco, preco_nogain

# data
# dataset = 'openwebtext'
# dataset = 'ED'
dataset = 'Slim'
do_eval = True
# gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
gradient_accumulation_steps = 16
# batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
# block_size = 1024
batch_size = 4 #mamba需要較大的batch size #longhorn:4
block_size = 256 #預設值會被run.sh 的 block_size 覆蓋

# 分支損失（PreCo）
use_branch_losses = False  # 僅用 CE 端到端訓練兩分支（建議）
branch_longhorn_weight = 1.0
branch_ttt_weight = 1.0

# model - 130M TTT配置優化
n_head = 8    # 從6頭增加到8頭 (768/8=96 head_dim)
n_embd = 512  # 從1024維減少到768維 (平衡參數分配)
dropout = 0.2
n_layer = 12  # TTT 層數配置 (從15層減少到12層)
# 原配置 (568M): 32層×1024維×6頭
# 新配置 (128M): 10層×768維×8頭 - 接近130M目標
# dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 3e-4 # max learning rate #longhorn
max_iters = 20000
# max_iters = 600000 # total number of training iterations
weight_decay = 1e-1 #longhorn
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 500
# warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 20000
# lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
# min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
min_lr = 1e-5  # 對齊JAX main
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# dtype = 'float32' # 先用float32測試穩定性
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# 優化：確保使用最佳精度
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # 優化 cuDNN 性能
compile = False
# compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
# TTT 特定配置 - 130M 優化版本
ttt_base_lr = 1.0
mini_batch_size = 8   # 130M 配置：匹配注意力頭數 (8頭)
temperature = 1.0
use_gate = True
share_qk = True       # 啟用 Q/K 共享
ttt_layer_type = "linear"
scan_checkpoint_group_size = 1  # 最小化檢查點以提升速度
pre_conv = True  # 重要：官方所有TTT模型都使用
conv_kernel = 2  # 🔧 修正：與 run.sh 保持一致

# PreCo 特定配置 - 與 run.sh 保持一致
longhorn_d_model = 512    # Longhorn 模型維度
longhorn_n_layer = 12     # Longhorn 層數 (與 run.sh 一致)
longhorn_d_state = 8      # Longhorn SSM 狀態維度 (與 run.sh 一致)
longhorn_ssm_expand = 6   # Longhorn SSM 擴展倍數 (與 run.sh 一致)
ttt_hidden_size = 512     # TTT 隱藏層維度
ttt_num_layers = 1        # TTT 層數 (與 run.sh 一致，每層一個 TTTLinear)
ttt_num_heads = 8         # TTT 注意力頭數
kalman_hidden_dim = 256   # Kalman 網絡維度
mlp_ratio = 14            # MLP 倍數 (可通過命令行參數覆蓋)

# 更新 config_keys
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]

# 🔧 移除配置文件依賴：所有配置都通過命令行參數傳遞
# 不再載入任何配置文件，完全依賴命令行參數和預設值
print(f"✅ 使用命令行參數配置，模型類型: {model_name}")

# 🔧 修正：確保模型名稱正確
if model_name not in ["ttt", "longhorn", "preco", "preco_nogain"]:
    raise ValueError(f"不支持的模型類型: {model_name}")

exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

# Change to 2% of max iters
# if dataset == "openwebtext":
if dataset == "Slim":
    warmup_iters = min(int(0.15 * max_iters), 3000) # 增加warmup比例
    #warmup_iters = min(int(0.05 * max_iters), 2000) longhorn

# 單卡安全檢查：TTT 固定切分需要 block_size 可整除 mini_batch_size
if model_name == "ttt":
    assert block_size % mini_batch_size == 0, \
        f"block_size({block_size}) 必須可被 mini_batch_size({mini_batch_size}) 整除，否則固定切分會錯位"

# logging
total_tokens = block_size * batch_size * gradient_accumulation_steps * max_iters / 1e9
experiment_name = f"{dataset}_{total_tokens:.1f}_block{block_size}_{model_name}"
wandb_log_interval = 50  # 每 50 次迭代才記錄到 wandb

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(master_seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)

# 🔧 恢復：使用 PyTorch 原生的簡單數據載入
print("🚀 使用 PyTorch 原生數據載入方式")

# 載入長陣列
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

print(f"✅ 載入數據:")
print(f"  Train tokens: {len(train_data):,}")
print(f"  Val tokens: {len(val_data):,}")
print(f"  序列長度: {block_size}")

# 載入meta資訊
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta.get('vocab_size', 50257)
    print(f"✅ 載入meta資訊: vocab_size={meta_vocab_size}")
else:
    meta_vocab_size = 50257
    print(f"⚠️ 未找到meta.pkl，使用預設vocab_size={meta_vocab_size}")

def get_batch(split):
    """PyTorch 經典的數據載入方式"""
    data = train_data if split == 'train' else val_data
    # 隨機選擇起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 直接切片獲取序列
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
# 注意：meta 已經在上面讀取過了，這裡直接使用
# meta_vocab_size = meta.get('vocab_size', 50257)
# print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
# print(f"Using custom tokenizer with vocab_size = {meta_vocab_size}")
# if meta_vocab_size != 32000:  # Llama-2 預設大小
#     print(f"Note: Using non-standard vocab_size. Ensure model embedding layer matches.")

# model init
if model_name == "preco" or model_name == "preco_nogain":
    # PreCo 系列不使用通用的 model_args
    model_args = {}
else:
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

# # TTT 特定配置
# ttt_base_lr = 0.1
# mini_batch_size = 8
# temperature = 1.0
# use_gate = True
# ttt_layer_type = "linear"
# scan_checkpoint_group_size = 4

if model_name == "rwkv":
    model_config = RWKVConfig()
    model_config.vocab_size = meta_vocab_size
    model_config.n_layer = n_layer
    model_config.n_head = n_head
    model_config.n_embd = n_embd
    model_config.block_size = block_size
    model_config.bias = bias
    model_config.dropout = dropout
    model = RWKV(model_config)

# elif model_name == "llama":
    # model_config = LLaMAConfig()
    # model_config.vocab_size = meta_vocab_size
    # model_config.n_layer = n_layer
    # model_config.n_head = n_head
    # model_config.n_embd = n_embd
    # model_config.block_size = block_size
    # model_config.bias = bias
    # model_config.dropout = dropout
    # model = LLaMA(model_config)

# elif model_name == "retnet":
    # model_config = RetNetConfig()
    # model_config.vocab_size = meta_vocab_size
    # model_config.decoder_layers = n_layer
    # model_config.decoder_retention_heads = n_head
    # model_config.decoder_embed_dim = n_embd
    # model_config.decoder_value_embed_dim = int(1.8 * n_embd) // n_head * n_head
    # model_config.decoder_ffn_embed_dim = int(1.8 * n_embd) // n_head * n_head
    # model_config.max_target_positions = block_size
    # model = RetNet(model_config)

# elif model_name == "gla":
    # model_config = GLAConfig(
        # d_model=n_embd,
        # n_head=n_head,
        # n_layer=n_layer,
        # context_length=block_size,
        # vocab_size=meta_vocab_size,
    # )
    # model = GLA(model_config)

# elif model_name == "gla_online":
    # model_config = GLAOnlineConfig(
        # d_model=n_embd,
        # n_head=n_head,
        # n_layer=n_layer,
        # context_length=block_size,
        # vocab_size=meta_vocab_size,
        # delta_bias=0.0,
    # )
    # model = GLAOnline(model_config)

elif model_name == "mamba":
    model_config = MambaConfig()
    model_config.vocab_size = meta_vocab_size
    model_config.d_model = n_embd
    model_config.n_layer = 2 * n_layer
    model_config.d_state = 16
    model = Mamba(model_config)

elif model_name == "longhorn":
    model_config = LonghornConfig()
    model_config.vocab_size = meta_vocab_size
    model_config.d_model = n_embd
    model_config.n_layer = n_layer  # 
    model_config.ssm_cfg = {
        'd_state': 16,      # SSM 狀態維度 4
        'd_conv': 4,        # 卷積核大小 3
        'expand': 4         # 🎯 調整：內部維度倍數 (10×512=5120) 10
    }
    model = LonghornLM(model_config)
    
    # Longhorn 公平比較配置提示
    if master_process:
        print("=" * 60)
        print("Longhorn SSM 公平比較配置")
        print("=" * 60)
        print(f"🎯 公平比較設計:")
        print(f"  1. 層數: {n_layer} 層 (與其他模型一致)")
        print(f"  2. 維度: {n_embd} 維 (與其他模型一致)")
        print(f"  3. 內部倍數: expand={model_config.ssm_cfg['expand']} (d_inner={model_config.ssm_cfg['expand']*n_embd})")
        print(f"  4. 預期參數量: ~130M (expand=11 調整版本)")
        print(f"  5. SSM 狀態維度: {model_config.ssm_cfg['d_state']}")
        print("=" * 60)

elif model_name == "ttt":
    # 計算intermediate_size (通常是hidden_size的2.67倍)
    intermediate_size = int(n_embd * 2.67)
    
    model_config = TTTConfig(
        vocab_size=meta_vocab_size,
        hidden_size=n_embd,
        intermediate_size=intermediate_size,  # 添加MLP中間層維度
        num_hidden_layers=n_layer,  # 使用新的層數參數
        num_attention_heads=n_head,
        max_position_embeddings=block_size,
        ttt_base_lr=ttt_base_lr,
        mini_batch_size=mini_batch_size,  # 平行優化：更大的 mini-batch
        use_gate=use_gate,
        ttt_layer_type=ttt_layer_type,
        scan_checkpoint_group_size=scan_checkpoint_group_size,
        dropout=dropout,
        pre_conv=pre_conv,  # 平行優化：預卷積
        conv_kernel=conv_kernel,
        # 修正特殊token ID以匹配您的tokenizer
        pad_token_id=0,   # <pad>
        bos_token_id=2,   # <s>
        eos_token_id=3,   # </s>
    )
    model = TTTForCausalLM(model_config)
    
    # TTT 平行訓練優化提示
    if master_process:
        print("=" * 60)
        print("TTT 平行訓練優化配置")
        print("=" * 60)
        print(f"🚀 平行優化特點:")
        print(f"  1. 更大的 mini-batch: {mini_batch_size} (減少循環次數)")
        print(f"  2. 預卷積處理: {pre_conv}")
        print(f"  3. 最小化檢查點: {scan_checkpoint_group_size}")
        print(f"  4. 優化的 TTT 學習率: {ttt_base_lr}")
        print(f"  5. 預期速度提升: 3-5倍")
        print("=" * 60)

elif model_name == "preco":
    model_config = PreCoNewConfig(
        vocab_size=meta_vocab_size,
        d_model=longhorn_d_model,
        n_layer=longhorn_n_layer,
        d_state=longhorn_d_state,
        d_conv=3,  # 固定為3，與PreCo.py一致
        expand=longhorn_ssm_expand,
        ttt_num_heads=ttt_num_heads,
        ttt_num_layers=ttt_num_layers,  # 這裡是 1，每層一個 TTTLinear
        mini_batch_size=mini_batch_size,  # TTT mini batch size
        dropout=dropout,
    )
    model = PreCoNewModel(model_config)
    
    # PreCo 混合模型訓練提示
    if master_process:
        print("=" * 70)
        print("PreCo (Prediction-Correction) 混合模型配置 - 修正版架構")
        print("=" * 70)
        print(f"🔬 Kalman Filter 架構特點:")
        print(f"  1. Prediction (Longhorn): 快速封閉解預測")
        print(f"  2. Correction (TTT): 測試時訓練校正")
        print(f"  3. Kalman Gain: Token 重要性動態權重")
        print(f"  4. 聯合訓練: 端到端優化所有組件")
        print(f"")
        print(f"🎯 修正後架構設計:")
        print(f"  - 每個 PreCoBlock: 1個 Longhorn + 1個 TTTLinear + 1個 Kalman Gate")
        print(f"  - 總層數: {longhorn_n_layer} 層 PreCoBlock")
        print(f"  - 實際 TTT 層數: {longhorn_n_layer} 層 (每層一個 TTTLinear)")
        print(f"  - 共用 LM Head 設計節省參數")
        print(f"")
        print(f"📊 模型參數 (修正版):")
        print(f"  - Longhorn 層數: {longhorn_n_layer}, 維度: {longhorn_d_model}, expand: {longhorn_ssm_expand}")
        print(f"  - TTT 實際層數: {longhorn_n_layer} (每層 {ttt_num_heads} 頭), 維度: {ttt_hidden_size}")
        print(f"  - Kalman 網絡維度: {kalman_hidden_dim} (增強動態分配)")
        print(f"  - 共用 LM Head: [{longhorn_d_model}, {meta_vocab_size}] = {longhorn_d_model * meta_vocab_size / 1e6:.1f}M 參數")
        print(f"  - 參數節省: 約 {2 * longhorn_d_model * meta_vocab_size / 1e6:.1f}M (原本需要三個獨立的輸出層)")
        print(f"")
        print(f"🚀 優化重點:")
        print(f"  - 修正 TTT 層數：從 {longhorn_n_layer}×{ttt_num_layers} 改為 {longhorn_n_layer}×1")
        print(f"  - 參數量優化：從 180M+ 降至約 140M")
        print(f"  - 增強 Kalman Gate 動態分配")
        print("=" * 70)

elif model_name == "preco_nogain":
    model_config = PreCoNoGainConfig(
        vocab_size=meta_vocab_size,
        d_model=longhorn_d_model,
        n_layer=longhorn_n_layer,
        d_state=longhorn_d_state,
        d_conv=3,  # 固定為3
        expand=longhorn_ssm_expand,
        ttt_num_heads=ttt_num_heads,
        ttt_num_layers=ttt_num_layers,  # 這裡是 1，每層一個 TTTLinear
        dropout=dropout,
        longhorn_weight=0.7,  # 可調整
        ttt_weight=0.3,       # 可調整
    )
    model = PreCoNoGainModel(model_config)
    
    # PreCo NoGain 簡化模型訓練提示
    if master_process:
        print("=" * 70)
        print("PreCo NoGain (簡化版 PreCo) 混合模型配置 - 133M 參數量匹配版本")
        print("=" * 70)
        print(f"🔬 簡化架構特點:")
        print(f"  1. Prediction (Longhorn): 快速封閉解預測")
        print(f"  2. Correction (TTT): 測試時訓練校正")
        print(f"  3. 簡化權重: 可學習的全局權重融合 (無複雜 Kalman Gate)")
        print(f"  4. 聯合訓練: 端到端優化所有組件")
        print(f"")
        print(f"🎯 簡化設計優勢:")
        print(f"  - 移除複雜的 TokenButler-style Kalman Gate")
        print(f"  - 使用簡單的可學習權重: Longhorn {model_config.longhorn_weight:.1f} + TTT {model_config.ttt_weight:.1f}")
        print(f"  - 減少訓練不穩定性")
        print(f"  - 更快的收斂速度")
        print(f"")
        print(f"📊 模型參數 (133M 匹配版本):")
        print(f"  - Longhorn 層數: {longhorn_n_layer}, 維度: {longhorn_d_model}, expand: {longhorn_ssm_expand}")
        print(f"  - TTT 層數: {ttt_num_layers}, 頭數: {ttt_num_heads}, 維度: {ttt_hidden_size}")
        print(f"  - 權重類型: learnable (可在訓練中調整)")
        print(f"  - 共用 LM Head: [{longhorn_d_model}, {meta_vocab_size}] = {longhorn_d_model * meta_vocab_size / 1e6:.1f}M 參數")
        print(f"")
        print(f"🚀 優化重點:")
        print(f"  - 簡化的權重融合機制")
        print(f"  - 減少超參數調優複雜度")
        print(f"  - 更穩定的訓練過程")
        print("=" * 70)

else:
    raise Exception(f"Unknown model name {model_name}")

model.to(device)
# 只在 TTT 模型時轉換模型參數的數據類型
if model_name == "ttt":
    model = model.to(ptdtype)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer

# start with all of the candidate parameters
param_dict = {pn: p for pn, p in model.named_parameters()}
# filter out those that do not require grad
param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
# create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
# i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
num_decay_params = sum(p.numel() for p in decay_params)
num_nodecay_params = sum(p.numel() for p in nodecay_params)
total_params = num_decay_params + num_nodecay_params
print(f"[info] total model params: {total_params/1e6:.1f} M")

# Create AdamW optimizer and use the fused version if it is available
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device_type == 'cuda'
extra_args = dict(fused=True) if use_fused else dict()
optimizer = torch.optim.AdamW(optim_groups,
                              lr=learning_rate,
                              betas=(beta1, beta2),
                              weight_decay=weight_decay,
                              **extra_args)
print(f"using fused AdamW: {use_fused}")
checkpoint = None # free up memory

# ------------------------初始化 wandb-----------------------
if master_process:
    # project_name = "preco" if model_name == "preco" else "ttt_slim"
    project_name = "2048"
    wandb_config = {
            "model_name": model_name,
            "dataset": dataset,
            "block_size": block_size,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "max_iters": max_iters,
            "vocab_size": meta_vocab_size,
            "total_params": total_params,
        }
    
    # 添加模型特定配置
    if model_name == "preco":
        wandb_config.update({
            "longhorn_d_model": model_config.d_model,
            "longhorn_n_layer": model_config.n_layer,
            "ttt_num_heads": model_config.ttt_num_heads,
            "ttt_num_layers": model_config.ttt_num_layers,
            "d_state": model_config.d_state,
            "expand": model_config.expand,
            "dropout": model_config.dropout,
            "training_objective": "corrected_state_cross_entropy_only",
            "shared_lm_head": True,
            "shared_components": "longhorn_C + ttt_wo + preco_q",
            "parameter_sharing": "三合一共用設計",
            "shared_lm_head_params": f"{model_config.d_model * meta_vocab_size / 1e6:.1f}M",
            "parameter_savings": f"{2 * model_config.d_model * meta_vocab_size / 1e6:.1f}M",
        })
    elif model_name == "preco_nogain":
        wandb_config.update({
            "longhorn_d_model": model_config.d_model,
            "longhorn_n_layer": model_config.n_layer,
            "ttt_num_heads": model_config.ttt_num_heads,
            "ttt_num_layers": model_config.ttt_num_layers,
            "d_state": model_config.d_state,
            "expand": model_config.expand,
            "dropout": model_config.dropout,
            "longhorn_weight": model_config.longhorn_weight,
            "ttt_weight": model_config.ttt_weight,
            "training_objective": "simplified_weight_fusion",
            "shared_lm_head": True,
            "architecture_type": "preco_nogain_simplified",
            "shared_lm_head_params": f"{model_config.d_model * meta_vocab_size / 1e6:.1f}M",
            "parameter_savings": f"{2 * model_config.d_model * meta_vocab_size / 1e6:.1f}M",
        })
    else:
        wandb_config.update({
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
        })
    
    wandb.init(
        project=project_name,
        name=experiment_name,
        config=wandb_config
    )

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# 添加 PPL 計算函數
def calculate_ppl(loss):
    if loss > 700:  # exp(700) 接近 float 上限
        return float('inf')
    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')

# 修改 estimate_loss 函數
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        # 收集有效 batch 的 loss（顯示用）與 PPL 用的 CE/NLL
        losses_list: list = []
        ppl_losses_list: list = []
        # 追加：在評估時計算簡單診斷指標
        accs: list = []
        max_probs: list = []
        
        # PreCo 系列特定統計
        if model_name == "preco" or model_name == "preco_nogain":
            ce_losses: list = []
            longhorn_losses: list = []
            ttt_losses: list = []
            kalman_means: list = []
            kalman_stds: list = []
        bad_batches = 0
            
        for k in range(eval_iters):
            x, y = get_batch(split)
            # 評估顯式關閉 AMP，自動以 FP32 計算更穩定
            from torch.cuda.amp import autocast
            with autocast(enabled=False):
                if model_name == "ttt":
                    outputs = model(
                        input_ids=x, 
                        labels=y
                    )
                    # 檢查輸出格式
                    if isinstance(outputs, tuple):
                        logits, loss = outputs[0], outputs[1]
                    else:
                        logits, loss = outputs.logits, outputs.loss
                    # 追加：計算shift後的top-1 accuracy與平均最大機率
                    try:
                        shift_logits = logits[..., :-1, :].float()
                        shift_labels = y[..., 1:].long()
                        preds = shift_logits.argmax(dim=-1)
                        acc_val = (preds == shift_labels).float().mean().item()
                        # 平均最大機率
                        max_prob_val = torch.softmax(shift_logits, dim=-1).max(dim=-1)[0].mean().item()
                        accs.append(acc_val)
                        max_probs.append(max_prob_val)
                    except Exception:
                        pass
                    # 記錄 loss 與 PPL 用的 NLL（CE）
                    loss_val = float(loss.item()) if hasattr(loss, 'item') else float(loss)
                    if math.isfinite(loss_val):
                        losses_list.append(loss_val)
                        ppl_losses_list.append(loss_val)
                    else:
                        bad_batches += 1
                elif model_name == "preco" or model_name == "preco_nogain":
                    if model_name == "preco":
                        # ✨ PreCo 新版：使用 TTT 原生自適應學習率
                        logits, loss_dict = model(
                            x,
                            y,
                            compute_branch_loss=False,
                        )
                    else:  # preco_nogain
                        # PreCo NoGain 仍使用外部調度
                        ttt_lr_mult = get_ttt_lr_mult(iter_num)
                        logits, loss_dict = model(
                            x,
                            y,
                            ttt_lr_mult=ttt_lr_mult,
                            compute_branch_loss=False,
                        )
                    
                    # 確保所有 loss_dict 值都轉為 float（不覆蓋非有限值）
                    for key in list(loss_dict.keys()):
                        if hasattr(loss_dict[key], 'item'):
                            loss_dict[key] = loss_dict[key].item()
                        else:
                            loss_dict[key] = float(loss_dict[key])
                    total_loss_val = float(loss_dict.get('total_loss', float('nan')))
                    ce_loss_val = float(loss_dict.get('ce_loss', float('nan')))
                    # 僅在兩者有限時納入主統計；PPL 使用 CE loss
                    if math.isfinite(total_loss_val) and math.isfinite(ce_loss_val):
                        losses_list.append(total_loss_val)
                        ppl_losses_list.append(ce_loss_val)
                        # 記錄簡化版的損失統計
                        ce_losses.append(ce_loss_val)
                        longhorn_losses.append(float(loss_dict.get('longhorn_loss', 0.0)))
                        ttt_losses.append(float(loss_dict.get('ttt_loss', 0.0)))
                        # 根據模型類型使用不同的統計字段
                        if model_name == "preco":
                            kalman_means.append(float(loss_dict.get('kalman_mean', 0.0)))
                            kalman_stds.append(float(loss_dict.get('kalman_std', 0.0)))
                        else:
                            kalman_means.append(float(loss_dict.get('weight_mean', 0.0)))
                            kalman_stds.append(float(loss_dict.get('weight_std', 0.0)))
                    else:
                        bad_batches += 1
                else:
                    logits, loss = model(x, y)
                    # 其它模型：只要 loss 有限就計入，PPL 使用同一 loss
                    loss_val = float(loss.item()) if hasattr(loss, 'item') else float(loss)
                    if math.isfinite(loss_val):
                        losses_list.append(loss_val)
                        ppl_losses_list.append(loss_val)
                    else:
                        bad_batches += 1
            
        mean_loss = (sum(losses_list) / len(losses_list)) if len(losses_list) > 0 else float('inf')
        mean_ppl_loss = (sum(ppl_losses_list) / len(ppl_losses_list)) if len(ppl_losses_list) > 0 else float('inf')
        result = {
            'loss': float(mean_loss),
            'ppl': calculate_ppl(float(mean_ppl_loss))
        }
        # 回傳診斷指標
        if model_name == "ttt":
            if len(accs) > 0:
                result['acc'] = float(sum(accs) / len(accs))
            if len(max_probs) > 0:
                result['max_prob'] = float(sum(max_probs) / len(max_probs))
        
        # 添加 PreCo 系列統計
        if model_name == "preco" or model_name == "preco_nogain":
            result.update({
                'ce_loss': float(sum(ce_losses)/len(ce_losses)) if len(ce_losses)>0 else float('inf'),
                'longhorn_loss': float(sum(longhorn_losses)/len(longhorn_losses)) if len(longhorn_losses)>0 else float('inf'),
                'ttt_loss': float(sum(ttt_losses)/len(ttt_losses)) if len(ttt_losses)>0 else float('inf'),
                'kalman_mean': float(sum(kalman_means)/len(kalman_means)) if len(kalman_means)>0 else 0.0,
                'kalman_std': float(sum(kalman_stds)/len(kalman_stds)) if len(kalman_stds)>0 else 0.0,
                'dropped_batches': bad_batches,
            })
            
        out[split] = result
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) 線性 warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) 達到或超過衰減總步數：固定為 min_lr
    if it >= lr_decay_iters:
        return min_lr
    # 3) 中間區段：標準餘弦衰減
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    decay_ratio = max(0.0, min(1.0, decay_ratio))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# 添加 TTT 學習率調度 - 修正版
def get_ttt_lr_mult(it):
    """TTT 學習率調度 - 修正版，邏輯清晰"""
    if model_name == "ttt" or model_name == "preco" or model_name == "preco_nogain":
        base_mult = 1.0
        
        # 🔧 更平滑的 warmup 策略
        warmup_steps = 500  # 增加到 500 iter，更平滑
        
        if it < warmup_steps:
            # 使用餘弦 warmup，更平滑
            progress = it / warmup_steps
            # 從 0.1 開始，使用餘弦函數平滑到 1.0
            return 0.1 + 0.9 * (1.0 - math.cos(math.pi * progress)) / 2.0
        
        # 500-2000 iter：標準學習率
        elif it < 2000:
            return base_mult
        
        # 🔧 更平滑的衰減策略
        elif it >= 3000:  # 提前開始衰減
            decay_start = 3000
            decay_ratio = (it - decay_start) / (lr_decay_iters - decay_start)
            decay_ratio = min(decay_ratio, 1.0)
            
            # 使用餘弦衰減，更平滑
            min_ttt_mult = 0.3  # 降低最小值
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_ttt_mult + coeff * (base_mult - min_ttt_mult)
        
        # 2000-3000 iter：保持標準激活
        else:  # it >= 2000 and it < 3000
            return base_mult
        
    return 1.0

# 🔧 新增：Train/Val gap 監控和正則化機制
def calculate_train_val_gap(train_loss, val_loss):
    """計算 Train/Val gap 並返回過擬合指標"""
    gap = val_loss - train_loss
    gap_ratio = gap / train_loss if train_loss > 0 else 0
    return gap, gap_ratio

def get_adaptive_regularization(iter_num, train_loss, val_loss, base_weight_decay=1e-1):
    """根據 Train/Val gap 自適應調整正則化強度"""
    gap, gap_ratio = calculate_train_val_gap(train_loss, val_loss)
    
    # 基礎正則化權重
    adaptive_weight_decay = base_weight_decay
    
    # 🔧 2000+ iter 後啟用自適應正則化
    if iter_num >= 2000:
        # 如果 gap_ratio > 0.05 (5%)，增加正則化
        if gap_ratio > 0.05:
            adaptive_weight_decay = base_weight_decay * 2.0
        # 如果 gap_ratio > 0.1 (10%)，進一步增加正則化
        if gap_ratio > 0.1:
            adaptive_weight_decay = base_weight_decay * 3.0
        # 如果 gap_ratio > 0.15 (15%)，最大正則化
        if gap_ratio > 0.15:
            adaptive_weight_decay = base_weight_decay * 4.0
    
    return adaptive_weight_decay, gap, gap_ratio

# training loop
train_stats = {
    "experiment_name": experiment_name,
    "global_args": config,
    "model/config": vars(model_config),
    "model/params": total_params,
    "total_tokens": total_tokens,
    "iter": [],
    "train/loss": [],
    "val/loss": [],
    "train/ppl": [],
    "val/ppl": [],
    "lr": [],
}

x, y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process and do_eval:
        losses = estimate_loss()
        train_loss = losses['train']['loss']
        val_loss = losses['val']['loss']
        
        # 🔧 新增：計算 Train/Val gap 和自適應正則化
        adaptive_weight_decay, gap, gap_ratio = get_adaptive_regularization(
            iter_num, train_loss, val_loss, weight_decay
        )
        
        print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        print(f"train ppl {losses['train']['ppl']:.4f}, val ppl {losses['val']['ppl']:.4f}")
        
        # 🔧 新增：Train/Val gap 監控輸出
        print(f"📈 Train/Val Gap: {gap:.4f} ({gap_ratio*100:.2f}%)")
        if iter_num >= 2000:
            print(f"🔧 Adaptive Weight Decay: {adaptive_weight_decay:.6f} (base: {weight_decay:.6f})")
            if gap_ratio > 0.05:
                print(f"⚠️  過擬合警告: Gap ratio {gap_ratio*100:.2f}% > 5%, 已增加正則化")
        
        # PreCo 系列模型的詳細輸出
        if model_name == "preco":
            print(f"  📊 PreCo 詳細統計:")
            print(f"    CE Loss: {losses['train']['ce_loss']:.4f} / {losses['val']['ce_loss']:.4f}")
            print(f"    監控指標:")
            print(f"      Longhorn Loss: {losses['train']['longhorn_loss']:.4f} / {losses['val']['longhorn_loss']:.4f}")
            print(f"      TTT Loss: {losses['train']['ttt_loss']:.4f} / {losses['val']['ttt_loss']:.4f}")
            print(f"    Kalman Gain 分析:")
            print(f"      平均值: {losses['train']['kalman_mean']:.4f} / {losses['val']['kalman_mean']:.4f}")
            print(f"      標準差: {losses['train']['kalman_std']:.4f} / {losses['val']['kalman_std']:.4f}")
        elif model_name == "preco_nogain":
            print(f"  📊 PreCo NoGain 詳細統計:")
            print(f"    CE Loss: {losses['train']['ce_loss']:.4f} / {losses['val']['ce_loss']:.4f}")
            print(f"    監控指標:")
            print(f"      Longhorn Loss: {losses['train']['longhorn_loss']:.4f} / {losses['val']['longhorn_loss']:.4f}")
            print(f"      TTT Loss: {losses['train']['ttt_loss']:.4f} / {losses['val']['ttt_loss']:.4f}")
            print(f"    簡化權重分析:")
            print(f"      平均值: {losses['train']['kalman_mean']:.4f} / {losses['val']['kalman_mean']:.4f}")
            print(f"      標準差: {losses['train']['kalman_std']:.4f} / {losses['val']['kalman_std']:.4f}")
        
        # 🔧 新增：動態調整優化器的權重衰減
        if iter_num >= 2000 and adaptive_weight_decay != weight_decay:
            for param_group in optimizer.param_groups:
                if 'weight_decay' in param_group:
                    param_group['weight_decay'] = adaptive_weight_decay
        
        train_stats['iter'].append(iter_num)
        train_stats['train/loss'].append(losses['train']['loss'])
        train_stats['val/loss'].append(losses['val']['loss'])
        train_stats['train/ppl'].append(losses['train']['ppl'])
        train_stats['val/ppl'].append(losses['val']['ppl'])
        train_stats['lr'].append(lr)
        
        # 記錄到 wandb - 簡化版本
        if master_process:
            log_data = {
                "train/loss": losses['train']['loss'],
                "val/loss": losses['val']['loss'],
                "train/ppl": losses['train']['ppl'],
                "val/ppl": losses['val']['ppl'],
                "learning_rate": lr,
                "iter": iter_num,
            }
            
            # PreCo 系列模型的記錄
            if model_name == "preco":
                def _f(x):
                    try:
                        return float(x)
                    except Exception:
                        return x
                log_data.update({
                    "preco/train_loss": _f(losses['train']['loss']),
                    "preco/val_loss": _f(losses['val']['loss']),
                    "preco/train_ce_loss": _f(losses['train']['ce_loss']),
                    "preco/val_ce_loss": _f(losses['val']['ce_loss']),
                    "preco/train_longhorn_loss": _f(losses['train']['longhorn_loss']),
                    "preco/val_longhorn_loss": _f(losses['val']['longhorn_loss']),
                    "preco/train_ttt_loss": _f(losses['train']['ttt_loss']),
                    "preco/val_ttt_loss": _f(losses['val']['ttt_loss']),
                    "preco/train_kalman_mean": _f(losses['train']['kalman_mean']),
                    "preco/val_kalman_mean": _f(losses['val']['kalman_mean']),
                    "preco/train_kalman_std": _f(losses['train']['kalman_std']),
                    "preco/val_kalman_std": _f(losses['val']['kalman_std']),
                    "preco/adaptive_ttt_lr": "native",  # ✨ 標記使用 TTT 原生自適應學習率
                    "preco/branch_use": int(use_branch_losses),
                    "preco/branch_longhorn_w": float(branch_longhorn_weight),
                    "preco/branch_ttt_w": float(branch_ttt_weight),
                })
            elif model_name == "preco_nogain":
                log_data.update({
                    "preco_nogain/train_loss": losses['train']['loss'],
                    "preco_nogain/val_loss": losses['val']['loss'],
                    "preco_nogain/train_ce_loss": losses['train']['ce_loss'],
                    "preco_nogain/val_ce_loss": losses['val']['ce_loss'],
                    "preco_nogain/train_longhorn_loss": losses['train']['longhorn_loss'],
                    "preco_nogain/val_longhorn_loss": losses['val']['longhorn_loss'],
                    "preco_nogain/train_ttt_loss": losses['train']['ttt_loss'],
                    "preco_nogain/val_ttt_loss": losses['val']['ttt_loss'],
                    "preco_nogain/train_weight_mean": losses['train']['kalman_mean'],
                    "preco_nogain/val_weight_mean": losses['val']['kalman_mean'],
                    "preco_nogain/train_weight_std": losses['train']['kalman_std'],
                    "preco_nogain/val_weight_std": losses['val']['kalman_std'],
                    "preco_nogain/ttt_lr_mult": get_ttt_lr_mult(iter_num),
                })
            wandb.log(log_data)
        
        # 已移除 PPL 繪圖/保存
        
        torch.save(train_stats,
                   os.path.join(out_dir, f'{experiment_name}.stats'))

        if losses['val']['loss'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']['loss']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'global_args': config,
                    'model_config': vars(model_config),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                # 附加模型特定的輔助中繼資料（不重複保存參數）
                if model_name == "preco":
                    # 保存 PreCo 特定元信息 - 不重複保存參數（已在 model.state_dict() 中）
                    preco_meta = {
                        # 模型架構信息
                        'shared_components': ['longhorn_backbone', 'ttt_backbone', 'q_network'],
                        'parameter_sharing_info': 'Longhorn backbone + TTT backbone + q_network 專用輸出',
                        'architecture_type': 'kalman_filter_with_q_network',
                        # 組件映射信息（用於載入時驗證）
                        'component_mapping': {
                            'longhorn_output': 'q_network',
                            'ttt_output': 'q_network',
                            'final_output': 'q_network',
                        },
                        # PreCoNewModel 沒有 q_network 和 kalman_gain 屬性，移除相關統計
                    }
                    checkpoint['preco_meta'] = preco_meta
                elif model_name == "preco_nogain":
                    preco_nogain_meta = {
                        'shared_components': ['longhorn_backbone', 'ttt_backbone', 'q_network'],
                        'parameter_sharing_info': 'Longhorn backbone + TTT backbone + q_network 專用輸出',
                        'architecture_type': 'preco_nogain_simplified',
                        'weight_type': 'learnable',
                        'longhorn_weight': model_config.longhorn_weight,
                        'ttt_weight': model_config.ttt_weight,
                        'component_mapping': {
                            'longhorn_output': 'q_network',
                            'ttt_output': 'q_network',
                            'final_output': 'q_network',
                        },
                    }
                    checkpoint['preco_nogain_meta'] = preco_nogain_meta
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint,
                          os.path.join(out_dir, f'{experiment_name}_best.pt'))

    # 🔧 修改：使用可配置的檢查點間隔 - 🚀 優化：減少保存頻率
    checkpoint_interval = globals().get('checkpoint_interval', 1000)  # 從 500 改為 1000
    if iter_num % checkpoint_interval == 0 and iter_num > 0:
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'global_args': config,
            'model_config': vars(model_config),
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': config,
        }
        if model_name == "preco":
            # 保存 PreCo 特定元信息 - 不重複保存參數（已在 model.state_dict() 中）
            preco_meta = {
                # 模型架構信息
                'shared_components': ['longhorn_backbone', 'ttt_backbone', 'q_network'],
                'parameter_sharing_info': 'Longhorn backbone + TTT backbone + q_network 專用輸出',
                'architecture_type': 'kalman_filter_with_q_network',
                # 組件映射信息（用於載入時驗證）
                'component_mapping': {
                    'longhorn_output': 'q_network',
                    'ttt_output': 'q_network',
                    'final_output': 'q_network',
                },
            }
            checkpoint['preco_meta'] = preco_meta
        elif model_name == "preco_nogain":
            # 保存 PreCo NoGain 特定元信息
            preco_nogain_meta = {
                # 模型架構信息
                'shared_components': ['longhorn_backbone', 'ttt_backbone', 'q_network'],
                'parameter_sharing_info': 'Longhorn backbone + TTT backbone + q_network 專用輸出',
                'architecture_type': 'preco_nogain_simplified',
                'weight_type': 'learnable',
                'longhorn_weight': model_config.longhorn_weight,
                'ttt_weight': model_config.ttt_weight,
                # 組件映射信息（用於載入時驗證）
                'component_mapping': {
                    'longhorn_output': 'q_network',
                    'ttt_output': 'q_network',
                    'final_output': 'q_network',
                },
            }
            checkpoint['preco_nogain_meta'] = preco_nogain_meta
        print(f"saving periodic checkpoint to {out_dir}")
        torch.save(checkpoint,
                  os.path.join(out_dir, f'{experiment_name}_iter{iter_num}.pt'))

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            if model_name == "ttt":
                # TTT 模型：使用標準 Cross-Entropy Loss
                outputs = model(
                    input_ids=x, 
                    labels=y
                )
                # 檢查輸出格式
                if isinstance(outputs, tuple):
                    logits, loss = outputs[0], outputs[1]
                else:
                    logits, loss = outputs.logits, outputs.loss
                # 只使用主要的 Cross-Entropy loss
                loss = loss / gradient_accumulation_steps
            elif model_name == "preco":
                # PreCo 混合模型前向傳播
                # ✨ 新版：不再傳遞 ttt_lr_mult，使用 TTT 原生自適應學習率
                logits, loss_dict = model(
                    x,
                    y,
                    compute_branch_loss=False,
                )
                loss = loss_dict['total_loss'] / gradient_accumulation_steps
            elif model_name == "preco_nogain":
                # PreCo NoGain 簡化模型前向傳播
                ttt_lr_mult = get_ttt_lr_mult(iter_num)
                logits, loss_dict = model(x, y, ttt_lr_mult=ttt_lr_mult)
                loss = loss_dict['total_loss'] / gradient_accumulation_steps
            else:
                logits, loss = model(x, y)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        x, y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging - 🚀 優化：減少同步點和日誌開銷
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    # 🚀 優化：減少日誌頻率，只在關鍵時刻同步
    if iter_num % (log_interval * 2) == 0 and master_process:  # 從每 50 次改為每 100 次
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms", flush=True)
        
        # 🚀 優化：減少 wandb 同步頻率
        if iter_num % (wandb_log_interval * 2) == 0 and master_process:  # 從每 50 次改為每 100 次
            log_data = {
                "train/iter_loss": lossf,
                "train/iter_time": dt*1000,
                "iter": iter_num
            }
            # PreCo 系列模型的即時統計記錄
            if model_name == "preco":
                log_data["preco/adaptive_ttt_lr"] = "native"  # ✨ 標記使用原生自適應學習率
                # 如果有 loss_dict，記錄詳細損失（轉為純 float，避免 CUDA tensor 造成 0/NaN）
                if 'loss_dict' in locals():
                    def _to_float(v):
                        try:
                            return float(v.detach().float().item()) if hasattr(v, 'detach') else (float(v.item()) if hasattr(v, 'item') else float(v))
                        except Exception:
                            return float(v)
                    log_data.update({
                        "preco/iter_loss": float(lossf),
                        "preco/iter_ce_loss": _to_float(loss_dict['ce_loss']),
                        "preco/iter_longhorn_loss": _to_float(loss_dict['longhorn_loss']),
                        "preco/iter_ttt_loss": _to_float(loss_dict['ttt_loss']),
                        "preco/iter_kalman_mean": _to_float(loss_dict['kalman_mean']),
                        "preco/iter_kalman_std": _to_float(loss_dict['kalman_std']),
                    })
                print(f"  ✨ TTT 自適應學習率: 原生機制")
            elif model_name == "preco_nogain":
                log_data["preco_nogain/ttt_lr_mult"] = get_ttt_lr_mult(iter_num)
                # 如果有 loss_dict，記錄詳細損失
                if 'loss_dict' in locals():
                    log_data.update({
                        "preco_nogain/iter_loss": lossf,
                        "preco_nogain/iter_ce_loss": loss_dict['ce_loss'],
                        "preco_nogain/iter_longhorn_loss": loss_dict['longhorn_loss'],
                        "preco_nogain/iter_ttt_loss": loss_dict['ttt_loss'],
                        "preco_nogain/iter_weight_mean": loss_dict.get('longhorn_weight_mean', 0.0),
                        "preco_nogain/iter_weight_std": loss_dict.get('weight_std', 0.0),
                        "preco_nogain/iter_weight_entropy": loss_dict.get('weight_entropy', 0.0),
                    })
                print(f"  TTT lr mult: {get_ttt_lr_mult(iter_num):.4f}")
            wandb.log(log_data)



    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

# -----------------結束時關閉 wandb-----------------
if master_process:
    wandb.finish()
