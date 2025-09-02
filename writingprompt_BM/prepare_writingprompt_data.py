#!/usr/bin/env python3
"""
準備 WritingPrompt 微調數據
將 WritingPrompt 數據轉換為適合 PreCo 模型微調的格式
"""

import json
import os
import sys
from typing import List, Dict
import random

def load_writingprompt_data(file_path: str) -> List[Dict]:
    """載入 WritingPrompt 數據"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def convert_to_preco_format(writingprompt_data: List[Dict]) -> List[Dict]:
    """轉換為 PreCo 微調格式"""
    converted_data = []
    
    for item in writingprompt_data:
        prompt = item['prompt']
        target = item['target']
        
        # 創建訓練樣本
        # 格式：prompt + target，用於自回歸訓練
        full_text = f"{prompt} {target}"
        
        converted_data.append({
            'text': full_text,
            'prompt': prompt,
            'target': target
        })
    
    return converted_data

def save_converted_data(data: List[Dict], output_path: str):
    """保存轉換後的數據"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 已保存 {len(data)} 個樣本到 {output_path}")

def create_train_val_split(data: List[Dict], val_ratio: float = 0.1) -> tuple:
    """創建訓練/驗證分割"""
    random.shuffle(data)
    
    val_size = int(len(data) * val_ratio)
    train_data = data[val_size:]
    val_data = data[:val_size]
    
    return train_data, val_data

def main():
    # 設定路徑
    input_file = "test_50.jsonl"  # 當前目錄下的數據文件
    output_dir = "../../longhorn/data/writingprompt"  # 相對於 longhorn 目錄
    
    print("🔄 準備 WritingPrompt 微調數據")
    print("="*50)
    
    # 檢查輸入文件
    if not os.path.exists(input_file):
        print(f"❌ 找不到輸入文件: {input_file}")
        print("請確保 WritingPrompt 數據文件存在")
        return
    
    # 載入原始數據
    print(f"📖 載入原始數據: {input_file}")
    original_data = load_writingprompt_data(input_file)
    print(f"✅ 載入了 {len(original_data)} 個樣本")
    
    # 轉換格式
    print("🔄 轉換為 PreCo 格式...")
    converted_data = convert_to_preco_format(original_data)
    
    # 創建訓練/驗證分割
    print("🔄 創建訓練/驗證分割...")
    train_data, val_data = create_train_val_split(converted_data, val_ratio=0.1)
    
    # 保存數據
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    
    save_converted_data(train_data, train_path)
    save_converted_data(val_data, val_path)
    
    print(f"\n📊 數據統計:")
    print(f"  - 總樣本數: {len(converted_data)}")
    print(f"  - 訓練樣本: {len(train_data)}")
    print(f"  - 驗證樣本: {len(val_data)}")
    
    # 顯示樣本示例
    print(f"\n📝 樣本示例:")
    for i, sample in enumerate(converted_data[:2]):
        print(f"\n樣本 {i+1}:")
        print(f"  Prompt: {sample['prompt'][:100]}...")
        print(f"  Target: {sample['target'][:100]}...")
        print(f"  Full text: {sample['text'][:150]}...")

if __name__ == "__main__":
    main() 