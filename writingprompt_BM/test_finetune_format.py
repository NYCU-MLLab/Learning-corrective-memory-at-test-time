#!/usr/bin/env python3
"""
測試新的微調格式
"""

import json
from finetune_writingprompt import load_writingprompt_data, format_prompt_for_generation

def test_data_format():
    """測試數據格式"""
    print("🧪 測試新的微調格式...")
    
    # 測試數據
    test_data = [
        {
            "prompt": "Write a story about a magical forest.",
            "target": "Once upon a time, there was a magical forest where trees whispered ancient secrets..."
        },
        {
            "prompt": "Describe a futuristic city.",
            "target": "The city of Neo-Tokyo rose like a metallic phoenix from the ashes of the old world..."
        }
    ]
    
    # 保存測試數據
    with open('test_format.jsonl', 'w', encoding='utf-8') as f:
        for data in test_data:
            f.write(json.dumps(data) + '\n')
    
    # 載入並測試
    loaded_data = load_writingprompt_data('test_format.jsonl')
    
    print("\n📊 載入的數據格式:")
    for i, data in enumerate(loaded_data):
        print(f"\n樣本 {i+1}:")
        print(f"文本: {data['text']}")
        print(f"Prompt 結束位置: {data['prompt_end']}")
        
        # 測試 prompt 格式化
        original_prompt = test_data[i]['prompt']
        formatted = format_prompt_for_generation(original_prompt)
        print(f"格式化後的 prompt: {formatted}")
    
    print("\n✅ 格式測試完成!")

if __name__ == "__main__":
    test_data_format() 