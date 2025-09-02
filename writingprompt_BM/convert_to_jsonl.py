#!/usr/bin/env python3
"""
將 JSON 格式的預測結果轉換為 JSONL 格式
"""

import json
import os

def convert_json_to_jsonl(input_file, output_file):
    """將 JSON 檔案轉換為 JSONL 格式"""
    print(f"🔄 轉換 {input_file} 為 {output_file}")
    
    # 檢查輸入檔案是否存在
    if not os.path.exists(input_file):
        print(f"❌ 找不到輸入檔案: {input_file}")
        return False
    
    try:
        # 讀取 JSON 檔案
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📖 讀取了 {len(data)} 筆資料")
        
        # 轉換為 JSONL 格式
        converted_count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                # 轉換欄位名稱
                jsonl_item = {
                    'question': item.get('prompt', ''),
                    'answer': item.get('target', ''),
                    'pre': item.get('prediction', '')
                }
                
                # 寫入 JSONL 格式（每行一個 JSON 物件）
                f.write(json.dumps(jsonl_item, ensure_ascii=False) + '\n')
                converted_count += 1
        
        print(f"✅ 成功轉換 {converted_count} 筆資料")
        print(f"📁 輸出檔案: {output_file}")
        
        # 顯示前幾筆資料作為範例
        print("\n📊 前 3 筆資料範例:")
        with open(output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                data = json.loads(line.strip())
                print(f"\n樣本 {i+1}:")
                print(f"  question: {data['question'][:50]}...")
                print(f"  answer: {data['answer'][:50]}...")
                print(f"  pre: {data['pre'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 轉換失敗: {e}")
        return False

def main():
    """主函數"""
    # 輸入和輸出檔案路徑
    input_files = [
        'preco_predictions.json',
        'longhorn_predictions.json',
        'ttt_predictions.json'
    ]
    
    print("🚀 JSON 轉 JSONL 轉換工具")
    print("=" * 50)
    
    for input_file in input_files:
        if os.path.exists(input_file):
            output_file = input_file.replace('.json', '.jsonl')
            print(f"\n📝 處理檔案: {input_file}")
            success = convert_json_to_jsonl(input_file, output_file)
            if success:
                print(f"✅ {input_file} 轉換成功")
            else:
                print(f"❌ {input_file} 轉換失敗")
        else:
            print(f"⚠️  檔案不存在: {input_file}")
    
    print("\n🎉 轉換完成!")

if __name__ == "__main__":
    main() 