#!/usr/bin/env python3
"""
改進的 PreCo 模型微調腳本 - 解決微調後性能下降問題
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../longhorn'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any

# 導入模型
from models.PreCo import PreCoNewModel, PreCoNewConfig
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


class TruthfulQADataset(Dataset):
    """TruthfulQA 數據集類"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # 載入數據
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        
        print(f"載入了 {len(self.data)} 條數據")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        completion = item['completion']
        
        # 組合完整文本
        full_text = prompt + completion
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 創建標籤（-100 表示忽略的位置）
        labels = encoding['input_ids'].clone()
        
        # 找到 prompt 的結束位置
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        prompt_length = prompt_tokens['input_ids'].shape[1]
        
        # 將 prompt 部分的標籤設為 -100
        labels[0, :prompt_length] = -100
        
        # 確保 PAD token 也被設為 -100
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[0, labels[0] == pad_token_id] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }


def train_epoch(model, dataloader, optimizer, device, epoch, scheduler=None):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # 移動數據到設備
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向傳播
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            targets=labels
        )
        
        # 處理輸出
        if isinstance(outputs, tuple):
            logits, loss_dict = outputs
            loss = loss_dict['total_loss']
            
            # 記錄詳細損失
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: CE={loss_dict['ce_loss']:.4f}, "
                      f"Longhorn={loss_dict['longhorn_loss']:.4f}, "
                      f"TTT={loss_dict['ttt_loss']:.4f}, "
                      f"Kalman_mean={loss_dict['kalman_mean']:.4f}")
        else:
            loss = outputs.loss
        
        total_loss += loss.item()
        
        # 反向傳播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 更新學習率（如果使用步進調度器）
        if scheduler is not None and hasattr(scheduler, 'step'):
            scheduler.step()
        
        # 更新進度條
        current_lr = optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
            'lr': f'{current_lr:.2e}'
        })
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, dataloader, device):
    """評估模型"""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                targets=labels
            )
            
            # 處理輸出
            if isinstance(outputs, tuple):
                logits, loss_dict = outputs
                loss = loss_dict['total_loss']
            else:
                loss = outputs.loss
            
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="改進的 PreCo 模型微調")
    parser.add_argument("--train-data", type=str, default="train.jsonl",
                       help="訓練數據路徑")
    parser.add_argument("--val-data", type=str, default="val.jsonl",
                       help="驗證數據路徑")
    parser.add_argument("--output-dir", type=str, default="checkpoints/preco_truthfulqa_improved",
                       help="輸出目錄")
    parser.add_argument("--pretrained-weights", type=str, 
                       default="../../longhorn/results/slim_results/preco_136m_8000iter_7.14_best/Slim_2.1_block1024_preco_best.pt",
                       help="預訓練權重路徑")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="學習率")
    parser.add_argument("--epochs", type=int, default=3,
                       help="訓練輪數")
    parser.add_argument("--max-length", type=int, default=256,
                       help="最大序列長度")
    parser.add_argument("--warmup-steps", type=int, default=100,
                       help="預熱步數")
    parser.add_argument("--freeze-embeddings", action="store_true",
                       help="凍結嵌入層")
    parser.add_argument("--freeze-layers", type=int, default=6,
                       help="凍結前幾層")
    
    args = parser.parse_args()
    
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 設置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    # 創建模型配置（與預訓練權重匹配）
    config = PreCoNewConfig(
        vocab_size=50257,
        d_model=512,
        n_layer=12,
        d_state=4,
        d_conv=3,
        expand=6,
        ttt_num_heads=8,
        ttt_num_layers=8,
        dropout=0.1
    )
    
    # 創建模型
    model = PreCoNewModel(config)
    
    # 載入預訓練權重
    if args.pretrained_weights and Path(args.pretrained_weights).exists():
        print(f"載入預訓練權重: {args.pretrained_weights}")
        checkpoint = torch.load(args.pretrained_weights, map_location=device)
        
        # 檢查權重格式
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("使用預訓練權重格式")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("使用微調檢查點格式")
        else:
            state_dict = checkpoint
            print("使用原始權重格式")
        
        # 載入權重
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print("成功載入預訓練權重")
            if missing_keys:
                print(f"缺失的鍵: {len(missing_keys)} 個")
            if unexpected_keys:
                print(f"意外的鍵: {len(unexpected_keys)} 個")
        except Exception as e:
            print(f"載入權重時出現錯誤: {e}")
            print("繼續使用部分載入的權重")
    else:
        print("未找到預訓練權重，使用隨機初始化")
    
    # 凍結部分層（可選）
    if args.freeze_embeddings:
        print("凍結嵌入層")
        model.embedding.requires_grad_(False)
    
    if args.freeze_layers > 0:
        print(f"凍結前 {args.freeze_layers} 層")
        for i in range(args.freeze_layers):
            if i < len(model.blocks):
                for param in model.blocks[i].parameters():
                    param.requires_grad_(False)
    
    model = model.to(device)
    
    # 創建 tokenizer
    raw_tokenizer = Tokenizer.from_file("/root/Thesis/tokenizer/slim_tokenizer/tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer)
    
    # 正確設置特殊 tokens
    vocab = tokenizer.get_vocab()
    
    if "</s>" in vocab:
        tokenizer.eos_token = "</s>"
        tokenizer.eos_token_id = vocab["</s>"]
    else:
        tokenizer.eos_token_id = tokenizer.vocab_size - 1
    
    if "<pad>" in vocab:
        tokenizer.pad_token = "<pad>"
        tokenizer.pad_token_id = vocab["<pad>"]
    else:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if "<s>" in vocab:
        tokenizer.bos_token = "<s>"
        tokenizer.bos_token_id = vocab["<s>"]
    
    if "<unk>" in vocab:
        tokenizer.unk_token = "<unk>"
        tokenizer.unk_token_id = vocab["<unk>"]
    
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"PAD token ID: {tokenizer.pad_token_id}")
    
    # 創建數據集
    train_dataset = TruthfulQADataset(args.train_data, tokenizer, args.max_length)
    val_dataset = TruthfulQADataset(args.val_data, tokenizer, args.max_length)
    
    # 創建數據加載器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # 創建優化器 - 使用不同的學習率策略
    no_decay = ['bias', 'LayerNorm.weight', 'RMSNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01,
            'lr': args.learning_rate
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': args.learning_rate
        }
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # 添加學習率調度器
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1)
    
    # 訓練循環
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        logging.info(f"開始訓練第 {epoch + 1} 輪")
        
        # 訓練
        train_loss = train_epoch(model, train_dataloader, optimizer, device, epoch + 1, scheduler)
        logging.info(f"第 {epoch + 1} 輪訓練損失: {train_loss:.4f}")
        
        # 驗證
        val_loss = evaluate(model, val_dataloader, device)
        logging.info(f"第 {epoch + 1} 輪驗證損失: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_loss': val_loss,
            }, output_dir / 'best_model.pth')
            logging.info(f"保存最佳模型，驗證損失: {val_loss:.4f}")
        
        # 保存檢查點
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, output_dir / f'checkpoint_epoch_{epoch + 1}.pth')
    
    # 保存最終模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer': tokenizer,
    }, output_dir / 'final_model.pth')
    
    logging.info("訓練完成！")


if __name__ == "__main__":
    main() 