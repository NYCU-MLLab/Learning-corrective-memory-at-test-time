python finetune_preco_truthfulqa.py \
    --train-data train.jsonl \
    --val-data val.jsonl \
    --output-dir checkpoints/preco_truthfulqa \
    --model-config 512_12 \
    --pretrained-weights ../../longhorn/results/slim_results/preco_136m_8000iter_7.14_best/Slim_2.1_block1024_preco_best.pt \
    --batch-size 4 \
    --learning-rate 5e-5 \
    --epochs 3

# 3. 測試微調後的模型
python test_finetuned_model.py