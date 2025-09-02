#!/usr/bin/env python3
"""
測試 Longhorn 模型生成效果
"""

import os
import sys
import torch
import argparse

# 添加模型路徑
sys.path.append('../../longhorn')
from models.longhorn import LonghornConfig, LonghornLM
from transformers import PreTrainedTokenizerFast

def load_longhorn_model(checkpoint_path, tokenizer_path, device):
    """載入 Longhorn 模型"""
    print(f"🔄 載入 Longhorn 模型: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config_dict = checkpoint['model_config']
    model_config = LonghornConfig(**model_config_dict)
    model = LonghornLM(model_config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"🔄 載入 tokenizer: {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

def generate_with_longhorn(model, tokenizer, prompt, max_new_tokens=25, temperature=1.0, device="cuda"):
    """使用 Longhorn 模型生成文本"""
    # 編碼輸入
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"📝 輸入: {prompt}")
    print(f"🔢 輸入 tokens: {input_ids[0].tolist()}")
    
    # 生成
    with torch.no_grad():
        for i in range(max_new_tokens):
            # 前向傳播
            logits, _ = model(input_ids)
            
            # 獲取最後一個 token 的 logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # 採樣
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 添加到序列
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # 解碼並顯示
            current_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            print(f"   Step {i+1}: {current_text}")
            
            # 檢查是否生成了 EOS
            if next_token.item() == tokenizer.eos_token_id:
                print(f"   ✅ 生成了 EOS token，停止生成")
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="測試 Longhorn 模型生成")
    parser.add_argument("--model", type=str, 
                       default="../../longhorn/results/slim_results/LH_135.1M_8000iter_7.16_best/Slim_2.1_block1024_longhorn_best.pt",
                       help="Longhorn 模型路徑")
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
        model, tokenizer = load_longhorn_model(args.model, args.tokenizer, args.device)
        
        # 測試 prompts
        test_prompts = [
            "Hello",
            "The story begins",
            "Once upon a time",
            "In the year 2024",
            "The scientist discovered"
        ]
        
        print("\n" + "="*60)
        print("🧪 Longhorn 模型生成測試")
        print("="*60)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{'='*50}")
            print(f"🧪 測試 {i}: {prompt}")
            print(f"{'='*50}")
            
            result = generate_with_longhorn(model, tokenizer, prompt, max_new_tokens=25, temperature=1.0, device=args.device)
            print(f"\n✅ 最終結果: {result}")
            print(f"{'='*50}")
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 