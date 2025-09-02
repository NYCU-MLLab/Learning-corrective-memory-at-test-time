#!/usr/bin/env python3
"""
PreCo 模型 WritingPrompt 微調腳本
"""

import os
import sys
import torch
import torch.nn.functional as F
import json
import time
import argparse
from typing import List, Dict
import random

# 添加模型路徑
sys.path.append('../../longhorn')
from models.PreCo import PreCoNewConfig, PreCoNewModel
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

def load_preco_model(checkpoint_path: str, device: str = "cuda") -> tuple:
    """載入預訓練的 PreCo 模型"""
    print(f"🔄 載入預訓練模型: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config_dict = checkpoint['model_config']
    model_config = PreCoNewConfig(**model_config_dict)
    model = PreCoNewModel(model_config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    
    print("✅ 預訓練模型載入成功!")
    return model, model_config

def load_tokenizer(tokenizer_path: str = "../tokenizer/slim_tokenizer"):
    """載入 tokenizer"""
    print(f"🔄 載入 tokenizer: {tokenizer_path}")
    
    try:
        # 使用 PreTrainedTokenizerFast 載入
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        print("✅ Tokenizer 載入成功!")
    except Exception as e:
        print(f"⚠️  Tokenizer 載入失敗: {e}")
        # 如果失敗，嘗試使用 AutoTokenizer 作為備用
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 設置 EOS token ID
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.vocab_size - 1
    
    # 設置 padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    return tokenizer

def load_writingprompt_data(file_path: str) -> List[Dict]:
    """載入 WritingPrompt 數據，使用明確的分隔符格式"""
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            # 檢查數據格式
            if 'text' in data:
                # 已經處理過的格式
                data_list.append({
                    'text': data['text'],
                    'prompt_end': len(data['text'])  # 整個文本都是 target
                })
            elif 'prompt' in data and 'target' in data:
                # 原始 WritingPrompt 格式 - 使用明確分隔符
                prompt = data['prompt']
                target = data['target']
                # 使用明確的分隔符格式
                full_text = f"Prompt: {prompt}\nResponse: {target}"
                prompt_end = len(f"Prompt: {prompt}\nResponse: ")
                data_list.append({
                    'text': full_text,
                    'prompt_end': prompt_end
                })
            else:
                print(f"⚠️  未知數據格式: {list(data.keys())}")
                continue
    return data_list

def create_training_batch(data_list: List[Dict], tokenizer, block_size: int = 512, batch_size: int = 4) -> tuple:
    """創建訓練批次，只對 target 部分計算 loss"""
    # 隨機選擇樣本
    batch_data = random.sample(data_list, min(batch_size, len(data_list)))
    
    # 編碼文本
    input_ids_batch = []
    targets_batch = []
    prompt_end_positions = []
    
    for data in batch_data:
        text = data['text']
        prompt_end = data['prompt_end']
        
        # 編碼完整文本
        encoded = tokenizer.encode(text, max_length=block_size, truncation=True)
        input_ids_batch.append(encoded)
        
        # 創建 targets，只對 target 部分計算 loss
        targets = [-100] * len(encoded)  # -100 表示忽略該位置的 loss
        
        # 找到 prompt 結束位置對應的 token 位置
        prompt_text = text[:prompt_end]
        prompt_tokens = tokenizer.encode(prompt_text, max_length=block_size, truncation=True)
        target_start_pos = len(prompt_tokens)
        
        # 從 target 開始位置設置 targets
        for i in range(target_start_pos, len(encoded)):
            targets[i] = encoded[i]
        
        targets_batch.append(targets)
        prompt_end_positions.append(target_start_pos)
    
    # 填充到相同長度
    max_len = max(len(seq) for seq in input_ids_batch)
    padded_input_ids = []
    padded_targets = []
    
    for input_ids, targets in zip(input_ids_batch, targets_batch):
        # 填充 input_ids
        padded_input = input_ids + [tokenizer.pad_token_id] * (max_len - len(input_ids))
        padded_input_ids.append(padded_input)
        
        # 填充 targets
        padded_target = targets + [-100] * (max_len - len(targets))
        padded_targets.append(padded_target)
    
    # 轉換為 tensor
    input_ids = torch.tensor(padded_input_ids)
    targets = torch.tensor(padded_targets)
    
    return input_ids, targets

def train_epoch(model, tokenizer, train_data, optimizer, device, 
                block_size=512, batch_size=4, num_batches=100):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx in range(num_batches):
        # 創建批次
        input_ids, targets = create_training_batch(train_data, tokenizer, block_size, batch_size)
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        
        # 前向傳播
        optimizer.zero_grad()
        logits, loss_dict = model(input_ids, targets=targets, ttt_lr_mult=0.3)
        loss = loss_dict['total_loss']
        
        # 反向傳播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches

def evaluate_model(model, tokenizer, val_data, device, 
                  block_size=512, batch_size=4, num_batches=20):
    """評估模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            # 創建批次
            input_ids, targets = create_training_batch(val_data, tokenizer, block_size, batch_size)
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            # 前向傳播
            logits, loss_dict = model(input_ids, targets=targets, ttt_lr_mult=0.3)
            loss = loss_dict['total_loss']
            
            total_loss += loss.item()
    
    return total_loss / num_batches

def save_checkpoint(model, optimizer, epoch, loss, save_path: str):
    """保存檢查點"""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'model_config': {
            'vocab_size': model.config.vocab_size,
            'd_model': model.config.d_model,
            'n_layer': model.config.n_layer,
            'd_state': model.config.d_state,
            'd_conv': model.config.d_conv,
            'expand': model.config.expand,
            'ttt_num_heads': model.config.ttt_num_heads,
            'ttt_num_layers': model.config.ttt_num_layers,
            'dropout': model.config.dropout,
        }
    }
    
    torch.save(checkpoint, save_path)
    print(f"✅ 檢查點已保存: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="PreCo WritingPrompt 微調")
    parser.add_argument("--pretrained_model", type=str, 
                       default="../../longhorn/results/slim_results/preco_136M_8000iter_7.23_bestnoTB/Slim_2.1_block1024_preco_best.pt",
                       help="預訓練模型路徑")
    parser.add_argument("--input_data", type=str, 
                       default="../../dataset/writingprompts/filtered_wp_data.jsonl",
                       help="輸入數據路徑")
    parser.add_argument("--tokenizer", type=str, 
                       default="../../tokenizer/slim_tokenizer",
                       help="Tokenizer 路徑")
    parser.add_argument("--output_dir", type=str, 
                       default="../../longhorn/results/writingprompt_finetuned_1000",
                       help="輸出目錄")
    parser.add_argument("--epochs", type=int, default=1, help="訓練輪數")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--block_size", type=int, default=1024, help="序列長度")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="學習率")
    parser.add_argument("--device", type=str, default="cuda", help="設備")
    parser.add_argument("--conservative_mode", action="store_true", 
                       help="保守模式：使用更少的資料和更低的學習率")
    parser.add_argument("--max_samples", type=int, default=2000, help="最大訓練樣本數")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="驗證集比例")
    parser.add_argument("--train_samples", type=int, default=1600, help="訓練樣本數")
    parser.add_argument("--val_samples", type=int, default=400, help="驗證樣本數")
    
    args = parser.parse_args()
    
    # 保守模式設置
    if args.conservative_mode:
        print("🛡️  啟用保守模式：防止過擬合")
        args.learning_rate = 5e-6  # 更低的學習率
        args.max_samples = 1000    # 更少的資料
        args.train_samples = 800   # 800 訓練 + 200 驗證
        args.val_samples = 200
        args.epochs = 1            # 只訓練 1 輪
        args.batch_size = 1        # 更小的批次
    
    # 檢查設備
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，使用 CPU")
        args.device = "cpu"
    
    print(f"🔧 使用設備: {args.device}")
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 載入模型和 tokenizer
        model, model_config = load_preco_model(args.pretrained_model, args.device)
        tokenizer = load_tokenizer(args.tokenizer)
        
        # 載入並處理數據
        print(f"📖 載入原始數據: {args.input_data}")
        all_data = load_writingprompt_data(args.input_data)
        print(f"✅ 載入了 {len(all_data)} 個樣本")
        
        # 限制樣本數量以避免過擬合
        if len(all_data) > args.max_samples:
            print(f"🔄 限制樣本數量為 {args.max_samples}")
            all_data = all_data[:args.max_samples]
        
        # 創建訓練/驗證分割
        print(f"🔄 創建訓練/驗證分割 (驗證比例: {args.val_ratio})")
        
        # 隨機打亂資料
        random.shuffle(all_data)
        
        # 如果指定了具體的樣本數，使用指定的數量
        if args.train_samples > 0 and args.val_samples > 0:
            total_requested = args.train_samples + args.val_samples
            if total_requested > len(all_data):
                print(f"⚠️  請求的樣本數 ({total_requested}) 超過可用樣本數 ({len(all_data)})")
                print(f"   使用所有可用樣本")
                train_data = all_data[args.val_samples:]
                val_data = all_data[:args.val_samples]
            else:
                train_data = all_data[args.val_samples:args.val_samples + args.train_samples]
                val_data = all_data[:args.val_samples]
        else:
            # 使用比例切割
            val_size = int(len(all_data) * args.val_ratio)
            train_data = all_data[val_size:]
            val_data = all_data[:val_size]
        
        print(f"📊 數據統計:")
        print(f"  - 總樣本數: {len(all_data)}")
        print(f"  - 訓練樣本: {len(train_data)}")
        print(f"  - 驗證樣本: {len(val_data)}")
        
        # 設置優化器和學習率調度器
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # 早停機制
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        early_stop = False
        
        # 訓練循環
        print(f"\n🚀 開始微調訓練...")
        print(f"  - 訓練輪數: {args.epochs}")
        print(f"  - 批次大小: {args.batch_size}")
        print(f"  - 學習率: {args.learning_rate}")
        print(f"  - 序列長度: {args.block_size}")
        
        for epoch in range(args.epochs):
            print(f"\n📅 Epoch {epoch + 1}/{args.epochs}")
            print("-" * 40)
            
            # 計算每輪的批次數（基於數據集大小）
            num_train_batches = min(200, len(train_data) // args.batch_size)  # 最多200個批次
            num_val_batches = min(50, len(val_data) // args.batch_size)       # 最多50個批次
            
            print(f"  📊 批次設置:")
            print(f"    - 訓練批次: {num_train_batches}")
            print(f"    - 驗證批次: {num_val_batches}")
            
            # 訓練
            start_time = time.time()
            train_loss = train_epoch(
                model, tokenizer, train_data, optimizer, args.device,
                args.block_size, args.batch_size, num_batches=num_train_batches
            )
            train_time = time.time() - start_time
            
            # 驗證
            start_time = time.time()
            val_loss = evaluate_model(
                model, tokenizer, val_data, args.device,
                args.block_size, args.batch_size, num_batches=num_val_batches
            )
            val_time = time.time() - start_time
            
            print(f"  📊 結果:")
            print(f"    - 訓練損失: {train_loss:.4f} (時間: {train_time:.1f}s)")
            print(f"    - 驗證損失: {val_loss:.4f} (時間: {val_time:.1f}s)")
            
            # 檢查過擬合
            train_val_gap = train_loss - val_loss
            print(f"    - 訓練/驗證差距: {train_val_gap:.4f}")
            
            if train_val_gap < -0.1:  # 驗證損失比訓練損失低很多，可能過擬合
                print(f"    ⚠️  警告：可能過擬合 (差距: {train_val_gap:.4f})")
            
            # 早停機制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_path = os.path.join(args.output_dir, f"writingprompt_best.pt")
                save_checkpoint(model, optimizer, epoch, val_loss, save_path)
                print(f"    ✅ 新的最佳模型已保存")
            else:
                patience_counter += 1
                print(f"    📉 驗證損失未改善 ({patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    print(f"    🛑 早停：驗證損失連續 {patience} 次未改善")
                    early_stop = True
                    break
            
            # 更新學習率
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            print(f"    - 當前學習率: {current_lr:.2e}")
            
            # 保存定期檢查點
            if (epoch + 1) % 1 == 0:
                save_path = os.path.join(args.output_dir, f"writingprompt_epoch_{epoch + 1}.pt")
                save_checkpoint(model, optimizer, epoch, val_loss, save_path)
        
        print(f"\n🎉 微調完成!")
        print(f"  - 最佳驗證損失: {best_val_loss:.4f}")
        print(f"  - 模型保存在: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()

def format_prompt_for_generation(prompt: str) -> str:
    """格式化 prompt 用於生成，與訓練時保持一致"""
    return f"Prompt: {prompt}\nResponse:"

if __name__ == "__main__":
    main() 