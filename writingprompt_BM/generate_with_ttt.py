#!/usr/bin/env python3
"""
使用 TTT 模型生成 WritingPrompt 預測結果
"""

import json
import torch
import sys
import os

# 添加模型路徑
sys.path.append('../../longhorn')
from models.ttt import TTTConfig, TTT
from transformers import PreTrainedTokenizerFast

def load_ttt_model(model_path, device='cuda'):
    """載入 TTT 模型"""
    print(f"載入 TTT 模型從: {model_path}")
    
    # 載入權重
    checkpoint = torch.load(model_path, map_location=device)
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    # 從 checkpoint 中獲取正確的配置
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        print(f"📋 Using model config from checkpoint: {model_config}")
        
        config = TTTConfig(
            vocab_size=model_config.get('vocab_size', 50257),
            hidden_size=model_config.get('hidden_size', 512),
            num_hidden_layers=model_config.get('num_hidden_layers', 12),
            num_attention_heads=model_config.get('num_attention_heads', 8),
            max_position_embeddings=model_config.get('max_position_embeddings', 1024),
            ttt_base_lr=model_config.get('ttt_base_lr', 1.0),
            mini_batch_size=model_config.get('mini_batch_size', 8),
            use_gate=model_config.get('use_gate', True),
            share_qk=model_config.get('share_qk', False),
            ttt_layer_type=model_config.get('ttt_layer_type', 'linear'),
            pre_conv=model_config.get('pre_conv', True),
            conv_kernel=model_config.get('conv_kernel', 2),
            scan_checkpoint_group_size=model_config.get('scan_checkpoint_group_size', 1),
            mlp_ratio=model_config.get('mlp_ratio', 14),
            pad_token_id=model_config.get('pad_token_id', 0),
            bos_token_id=model_config.get('bos_token_id', 2),
            eos_token_id=model_config.get('eos_token_id', 3),
            dropout=model_config.get('dropout', 0.1),
        )
    else:
        # 使用預設配置
        print("📋 Using default TTT config")
        config = TTTConfig(
            vocab_size=50257,
            hidden_size=512,
            num_hidden_layers=12,
            num_attention_heads=8,
            max_position_embeddings=1024,
            ttt_base_lr=1.0,
            mini_batch_size=8,
            use_gate=True,
            share_qk=False,
            ttt_layer_type="linear",
            pre_conv=True,
            conv_kernel=2,
            scan_checkpoint_group_size=1,
            mlp_ratio=14,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=3,
            dropout=0.1,
        )
    
    print(f"使用 TTT 配置: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")
    
    # 建立模型
    model = TTT(config)
    
    # 載入權重
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    print("✅ TTT 模型載入成功")
    return model

def load_tokenizer(tokenizer_path):
    """載入 tokenizer"""
    print(f"載入 tokenizer 從: {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("✅ Tokenizer 載入成功")
    return tokenizer

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.8, top_p=0.9, device='cuda'):
    """使用 TTT 模型生成回答"""
    # 編碼輸入
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    input_length = inputs.shape[1]
    
    # 手動實現生成邏輯
    generated = inputs.clone()
    
    with torch.no_grad():
        for step in range(max_length - input_length):
            # TTT 模型前向傳播 - 使用不同的輸出格式
            try:
                logits, _, _ = model(
                    input_ids=generated, 
                    labels=None,
                    loss_masks=None,
                    ttt_lr_mult=1.0  # 推理時使用標準 TTT 學習率
                )
            except Exception as e:
                print(f"TTT 前向傳播錯誤: {e}")
                break
            
            # 只取最後一個 token 的 logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # 避免生成特殊字符
            for token_id in [179, 374, tokenizer.pad_token_id]:  # 換行符、tab、pad token
                if token_id is not None:
                    next_token_logits[token_id] = float('-inf')
            
            # 應用 nucleus sampling (top-p)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累積概率超過 top_p 的 token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 採樣下一個 token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 添加到生成序列
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # 檢查是否生成了結束符號
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # 檢查是否重複太多（改進的重複檢測）
            if step > 20:
                last_20 = generated[0, -20:].tolist()
                unique_tokens = len(set(last_20))
                if unique_tokens <= 5:  # 如果最後20個token中只有5個不同的token
                    break
                
                # 檢查是否有連續重複
                if step > 5:
                    last_5 = generated[0, -5:].tolist()
                    if len(set(last_5)) == 1:  # 如果最後5個token都相同
                        break
    
    # 解碼輸出（只取生成的部分）
    generated_text = tokenizer.decode(generated[0][input_length:], skip_special_tokens=True)
    return generated_text.strip()

def load_test_data(file_path):
    """載入測試資料"""
    prompts = []
    targets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data['prompt'])
            targets.append(data['target'])
    return prompts, targets

def save_predictions(prompts, predictions, targets, filename='ttt_predictions.json'):
    """保存預測結果"""
    data = []
    for i, (prompt, pred, target) in enumerate(zip(prompts, predictions, targets)):
        data.append({
            'id': i,
            'prompt': prompt,
            'prediction': pred,
            'target': target
        })
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ TTT 預測結果已保存到 {filename}")

def main():
    # 設定
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}")
    
    # TTT 模型和 tokenizer 路徑
    # 自動尋找最新的 TTT 微調結果
    import glob
    finetune_dirs = glob.glob("../../longhorn/finetune_results/ttt_finetune_*")
    if finetune_dirs:
        # 按時間排序，取最新的
        finetune_dirs.sort()
        latest_dir = finetune_dirs[-1]
        model_path = os.path.join(latest_dir, "best_checkpoint.pt")
        print(f"🔍 Found latest TTT finetune directory: {latest_dir}")
    else:
        print("⚠️  No TTT finetune results found, using default path")
        model_path = "../../longhorn/results/slim_results/ttt_137.9M_8000iter_7.17_best/Slim_2.1_block1024_ttt_best.pt"
    
    tokenizer_path = "../../tokenizer/slim_tokenizer"
    
    # 檢查檔案是否存在
    if not os.path.exists(model_path):
        print(f"❌ 找不到 TTT 模型檔案: {model_path}")
        print("請修改 model_path 為您的實際 TTT 模型路徑")
        return
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ 找不到 tokenizer 路徑: {tokenizer_path}")
        print("請修改 tokenizer_path 為您的實際 tokenizer 路徑")
        return
    
    # 載入模型和 tokenizer
    model = load_ttt_model(model_path, device)
    tokenizer = load_tokenizer(tokenizer_path)
    
    # 載入測試資料
    print("載入測試資料...")
    prompts, targets = load_test_data('test_50.jsonl')
    print(f"載入了 {len(prompts)} 個測試樣本")
    
    # 生成預測
    predictions = []
    print("\n開始使用 TTT 模型生成預測...")
    
    for i, prompt in enumerate(prompts):
        print(f"TTT 生成第 {i+1}/50 個回答...")
        try:
            response = generate_response(model, tokenizer, prompt, device=device)
            predictions.append(response)
            print(f"  完成: {response[:100]}...")
        except Exception as e:
            print(f"  錯誤: {e}")
            predictions.append("生成失敗")
    
    # 保存結果
    save_predictions(prompts, predictions, targets)
    
    # 顯示前幾個結果
    print("\n前 3 個 TTT 生成結果:")
    for i in range(min(3, len(predictions))):
        print(f"\n樣本 {i+1}:")
        print(f"Prompt: {prompts[i]}")
        print(f"TTT Generated: {predictions[i][:200]}...")
        print(f"Target: {targets[i][:200]}...")

if __name__ == "__main__":
    main() 