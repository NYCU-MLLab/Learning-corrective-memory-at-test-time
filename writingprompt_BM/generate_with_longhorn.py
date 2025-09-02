#!/usr/bin/env python3
"""
使用 Longhorn 模型生成 WritingPrompt 預測結果
"""

import json
import torch
import sys
import os

# 添加模型路徑
sys.path.append('../../longhorn')
from models.longhorn import LonghornConfig, LonghornLM
from transformers import PreTrainedTokenizerFast

def load_longhorn_model(model_path, device='cuda'):
    """載入 Longhorn 模型"""
    print(f"載入 Longhorn 模型從: {model_path}")
    
    # 載入權重
    checkpoint = torch.load(model_path, map_location=device)
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    # 從權重維度推斷正確的配置
    state_dict = checkpoint['model']
    for key, value in state_dict.items():
        if 'layers.0.mixer.x_proj.weight' in key:
            print(f"Found x_proj weight shape: {value.shape}")
            # x_proj 維度是 [dt_rank + d_state*2, d_inner]
            dt_rank_plus_2d_state, d_inner = value.shape
            print(f"dt_rank + 2*d_state = {dt_rank_plus_2d_state}, d_inner = {d_inner}")
            
            # 假設 d_model = 512, 計算 expand
            d_model = 512
            expand = d_inner // d_model
            print(f"Calculated expand = {expand}")
            
            # 假設 dt_rank = d_model // 16 = 32, 計算 d_state
            dt_rank = d_model // 16
            d_state = (dt_rank_plus_2d_state - dt_rank) // 2
            print(f"Calculated d_state = {d_state}")
            
            config = LonghornConfig(
                d_model=d_model,
                n_layer=12,
                vocab_size=50264,  # 從權重推斷
                ssm_cfg={
                    "d_state": d_state,
                    "d_conv": 3,  # 從權重推斷
                    "expand": expand
                }
            )
            break
    else:
        # 如果找不到，使用預設配置
        config = LonghornConfig(
            d_model=512,
            n_layer=12,
            vocab_size=50264,
            ssm_cfg={
                "d_state": 1,
                "d_conv": 3,
                "expand": 11
            }
        )
    
    print(f"使用訓練配置: d_model={config.d_model}, n_layer={config.n_layer}, d_state={config.ssm_cfg['d_state']}, expand={config.ssm_cfg['expand']}")
    
    # 建立模型
    model = LonghornLM(config)
    
    # 載入權重
    model.load_state_dict(checkpoint['model'], strict=False)
    
    model.to(device)
    model.eval()
    print("✅ 模型載入成功")
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
    """使用 Longhorn 模型生成回答"""
    # 編碼輸入
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    input_length = inputs.shape[1]
    
    # 手動實現生成邏輯
    generated = inputs.clone()
    
    with torch.no_grad():
        for step in range(max_length - input_length):
            # 前向傳播
            logits, _ = model(generated)
            
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

def save_predictions(prompts, predictions, targets, filename='longhorn_predictions.jsonl'):
    """保存預測結果為 JSONL 格式"""
    with open(filename, 'w', encoding='utf-8') as f:
        for i, (prompt, pred, target) in enumerate(zip(prompts, predictions, targets)):
            data = {
                'question': prompt,
                'answer': target,
                'pre': pred
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"✅ 預測結果已保存到 {filename}")
    print(f"📊 總共保存了 {len(predictions)} 筆資料")

def main():
    # 設定
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}")
    
    # 模型和 tokenizer 路徑
    model_path = "../../longhorn/results/slim_results/LH_135.1M_8000iter_7.16_best/Slim_2.1_block1024_longhorn_best.pt"
    # model_path = "../../longhorn/results/slim_results/LH_135.1M_8000iter_7.16_best/Slim_2.1_block1024_longhorn_iter5000.pt"
    tokenizer_path = "../../tokenizer/slim_tokenizer"
    
    # 檢查檔案是否存在
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型檔案: {model_path}")
        print("請修改 model_path 為您的實際模型路徑")
        return
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ 找不到 tokenizer 路徑: {tokenizer_path}")
        print("請修改 tokenizer_path 為您的實際 tokenizer 路徑")
        return
    
    # 載入模型和 tokenizer
    model = load_longhorn_model(model_path, device)
    tokenizer = load_tokenizer(tokenizer_path)
    
    # 載入測試資料
    print("載入測試資料...")
    prompts, targets = load_test_data('test_500.jsonl')
    print(f"載入了 {len(prompts)} 個測試樣本")
    
    # 生成預測
    predictions = []
    print("\n開始生成預測...")
    
    for i, prompt in enumerate(prompts):
        print(f"生成第 {i+1}/{len(prompts)} 個回答...")
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
    print("\n前 3 個生成結果:")
    for i in range(min(3, len(predictions))):
        print(f"\n樣本 {i+1}:")
        print(f"Prompt: {prompts[i]}")
        print(f"Generated: {predictions[i][:200]}...")
        print(f"Target: {targets[i][:200]}...")

if __name__ == "__main__":
    main() 