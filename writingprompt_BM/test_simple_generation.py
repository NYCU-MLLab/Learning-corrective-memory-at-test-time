#!/usr/bin/env python3
"""
簡單的 PreCo 模型生成測試
使用更保守的參數來測試基本生成能力
"""

import os
import sys
import torch
import argparse

# 添加模型路徑
sys.path.append('../../longhorn')
from models.PreCo import PreCoNewConfig, PreCoNewModel
from transformers import PreTrainedTokenizerFast

def load_model_and_tokenizer(model_path, tokenizer_path, device):
    """載入模型和 tokenizer"""
    print(f"🔄 載入模型: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model_config_dict = checkpoint['model_config']
    model_config = PreCoNewConfig(**model_config_dict)
    model = PreCoNewModel(model_config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"🔄 載入 tokenizer: {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

def simple_generate(model, tokenizer, prompt, max_new_tokens=20, temperature=1.0, device="cuda"):
    """簡單的生成函數"""
    # 編碼輸入
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"📝 輸入: {prompt}")
    print(f"🔢 輸入 tokens: {input_ids[0].tolist()}")
    
    # 生成
    with torch.no_grad():
        for i in range(max_new_tokens):
            # 前向傳播
            outputs = model(input_ids, ttt_lr_mult=0.1)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # 獲取最後一個 token 的 logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # 簡單的採樣
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 添加到序列
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # 解碼並顯示
            current_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            print(f"   Step {i+1}: {current_text}")
            
            # 檢查是否生成了 EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="簡單的 PreCo 生成測試")
    parser.add_argument("--model", type=str, 
                       default="../../longhorn/results/writingprompt_finetuned_full/writingprompt_best.pt",
                       help="模型路徑")
    parser.add_argument("--tokenizer", type=str, 
                       default="../../tokenizer/slim_tokenizer",
                       help="Tokenizer 路徑")
    parser.add_argument("--device", type=str, default="cuda", help="設備")
    
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    
    print(f"🔧 使用設備: {args.device}")
    
    try:
        # 載入模型和 tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model, args.tokenizer, args.device)
        
        # 測試 prompts
        test_prompts = [
            "Hello",
            "The story begins",
            "Once upon a time",
            "In the year 2024",
            "The scientist discovered"
        ]
        
        print("\n" + "="*60)
        print("🧪 簡單生成測試")
        print("="*60)
        
        for prompt in test_prompts:
            print(f"\n{'='*50}")
            result = simple_generate(model, tokenizer, prompt, max_new_tokens=10, temperature=1.0, device=args.device)
            print(f"\n✅ 最終結果: {result}")
            print(f"{'='*50}")
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 