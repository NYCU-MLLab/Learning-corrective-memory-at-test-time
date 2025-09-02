#!/bin/bash
#original
mkdir -p trainlogs

# 🎯 TTT 原始125m配置
# - TTT 125m: 12層×768維 (125M)
# - 原始論文配置：hidden_size=768, intermediate_size=2048, num_hidden_layers=12, num_attention_heads=12

# 共同配置
block_size=2048  # 匹配JAX main
batch_size=4   # 增加batch size提高GPU利用率
grad_accum=4  # 減少梯度累積步驟，有效batch size = 8×2=16
max_iters=9600
warmup=960
lr=2.5e-3  # 稍微保守，考慮更大的詞彙表
wd=0.1   # 調整為JAX main的weight decay

model=$1

if [ "$model" == "longhorn" ]; then
    echo "🚀 訓練 Longhorn (133M, 512維度) - 12層公平比較"
    python train.py \
      --master_seed=1337 \
      --block_size=$block_size \
      --eval_interval=200 \
      --model_name=longhorn \
      --compile=False \
      --n_head=12 \
      --n_embd=768 \
      --n_layer=12 \
      --batch_size=$batch_size \
      --gradient_accumulation_steps=$grad_accum \
      --max_iters=$max_iters \
      --lr_decay_iters=$max_iters \
      --learning_rate=$lr \
      --weight_decay=$wd \
      --warmup_iters=$warmup \
      --grad_clip=1.0 \
      > trainlogs/longhorn_133m.log 2>&1

elif [ "$model" == "ttt" ]; then
    echo "🚀 訓練 TTT 125m (原始論文配置)"
    python train.py \
      --master_seed=1337 \
      --block_size=$block_size \
      --eval_interval=200 \
      --model_name=ttt \
      --compile=False \
      --n_head=12 \
      --n_embd=768 \
      --n_layer=12 \
      --batch_size=$batch_size \
      --gradient_accumulation_steps=$grad_accum \
      --max_iters=$max_iters \
      --lr_decay_iters=$max_iters \
      --learning_rate=$lr \
      --weight_decay=$wd \
      --warmup_iters=$warmup \
      --ttt_base_lr=1.0 \
      --mini_batch_size=16 \
      --dropout=0.0 \
      --temperature=1.0 \
      --grad_clip=1.0 \
      --use_gate=False \
      --ttt_layer_type=linear \
      --scan_checkpoint_group_size=0 \
      --pre_conv=True \
      --conv_kernel=4 \
      > trainlogs/ttt_125m.log 2>&1

elif [ "$model" == "preco" ]; then
    echo "🚀 訓練 PreCo (127M, 512維度) - TTT 原生自適應學習率版本"
    python train.py \
      --master_seed=1337 \
      --block_size=$block_size \
      --model_name=preco \
      --compile=False \
      \
      --longhorn_d_model=512 \
      --longhorn_n_layer=12 \
      --longhorn_d_state=8 \
      --longhorn_ssm_expand=6 \
      \
      --ttt_hidden_size=512 \
      --ttt_num_layers=1 \
      --ttt_num_heads=8 \
      --ttt_base_lr=0.01 \
      --mini_batch_size=16 \
      --use_gate=True \
      --share_qk=True \
      --ttt_layer_type=linear \
      --pre_conv=True \
      --conv_kernel=4 \
      --scan_checkpoint_group_size=0 \
      \
      --max_iters=$max_iters \
      --lr_decay_iters=$max_iters \
      --learning_rate=$lr \
      --weight_decay=$wd \
      --warmup_iters=$warmup \
      --batch_size=$batch_size \
      --gradient_accumulation_steps=$grad_accum \
      --eval_interval=200 \
      --grad_clip=1.0 \
      --dropout=0.1 \
      > trainlogs/preco_127m.log 2>&1

elif [ "$model" == "preco_nogain" ]; then
    echo "🚀 訓練 PreCo NoGain (136M, 512維度, 簡化版)"
    python train.py \
      --master_seed=1337 \
      --block_size=$block_size \
      --model_name=preco_nogain \
      --compile=False \
      \
      --longhorn_d_model=512 \
      --longhorn_n_layer=12 \
      --longhorn_d_state=4 \
      --longhorn_ssm_expand=6 \
      \
      --ttt_hidden_size=512 \
      --ttt_num_layers=8 \
      --ttt_num_heads=8 \
      --ttt_base_lr=1.0 \
      --mini_batch_size=8 \
      --use_gate=True \
      --share_qk=True \
      --ttt_layer_type=linear \
      --pre_conv=True \
      --conv_kernel=2 \
      --scan_checkpoint_group_size=1 \
      \
      --max_iters=$max_iters \
      --lr_decay_iters=$max_iters \
      --learning_rate=$lr \
      --weight_decay=$wd \
      --warmup_iters=$warmup \
      --batch_size=$batch_size \
      --gradient_accumulation_steps=$grad_accum \
      --eval_interval=200 \
      --grad_clip=1.0 \
      --dropout=0.1 \
      > trainlogs/preco_nogain_136m.log 2>&1

else
    echo "❌ 未知的模型類型: $model"
    echo "請使用: ./run.sh [preco|preco_nogain|ttt|longhorn]"
    echo ""
    echo "可用選項:"
    echo "  preco        - PreCo 混合模型 (完整版：Longhorn + TTT + Kalman Gate)"
    echo "  preco_nogain - PreCo NoGain 混合模型 (簡化版：Longhorn + TTT + 可學習權重)"
    echo "  ttt          - TTT-Linear 模型 (125m原始配置)"
    echo "  longhorn     - Longhorn SSM 模型"
    exit 1
fi

echo "✅ 訓練完成！日誌文件：trainlogs/${model}.log"
