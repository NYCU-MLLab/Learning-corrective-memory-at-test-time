#!/usr/bin/env python3
"""
測試 JSONL 格式輸出
"""

import json

def test_jsonl_format():
    """測試 JSONL 格式"""
    print("🧪 測試 JSONL 格式...")
    
    # 模擬數據
    prompts = ["Write a story about a cat.", "Describe a sunset."]
    predictions = ["Once upon a time, there was a cat...", "The sun slowly descended..."]
    targets = ["A cat lived in a magical forest...", "Golden rays painted the sky..."]
    
    # 保存為 JSONL
    filename = 'test_output.jsonl'
    with open(filename, 'w', encoding='utf-8') as f:
        for prompt, pred, target in zip(prompts, predictions, targets):
            data = {
                'question': prompt,
                'answer': target,
                'pre': pred
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"✅ 已保存到 {filename}")
    
    # 讀取並顯示
    print("\n📖 讀取並顯示內容:")
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            data = json.loads(line.strip())
            print(f"\n樣本 {i}:")
            print(f"  question: {data['question']}")
            print(f"  answer: {data['answer'][:50]}...")
            print(f"  pre: {data['pre'][:50]}...")
    
    print("\n✅ JSONL 格式測試完成!")

if __name__ == "__main__":
    test_jsonl_format() 