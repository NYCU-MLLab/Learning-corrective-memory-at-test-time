Longhorn parameters:
# python train.py config/train_slim.py \
#   --master_seed=1337 \
#   --block_size=$block \
#   --eval_interval=500 \
#   --model_name=$model \
#   --compile=False \
#   --n_head=6 --n_embd=1024 --n_layer=32 \
#   --batch_size=4 \
#   --gradient_accumulation_steps=32 \
#   --max_iters=20000 \
#   --lr_decay_iters=20000 \
#   --learning_rate=0.00003 \
#   > trainlogs/mini_${block}_${model}.log 2>&1

ttt parameters: