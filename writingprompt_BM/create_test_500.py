#!/usr/bin/env python3
"""
創建 500 筆測試資料 - 從 WritingPrompt 資料集中提取
格式與 test_50.jsonl 相同
"""

import json
import random
import os
import sys

def create_test_500(input_file, output_file="test_500.jsonl", num_samples=500, seed=42):
    """
    從 WritingPrompt 資料集中提取指定數量的測試樣本
    
    Args:
        input_file: 輸入檔案路徑
        output_file: 輸出檔案路徑
        num_samples: 要提取的樣本數量
        seed: 隨機種子
    """
    
    if not os.path.exists(input_file):
        print(f"❌ 找不到輸入檔案: {input_file}")
        return False
    
    # 設定隨機種子
    random.seed(seed)
    
    print(f"📖 正在讀取 {input_file}...")
    
    # 讀取所有資料
    all_data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    # 檢查格式
                    if 'prompt' in data and 'target' in data:
                        all_data.append(data)
                    else:
                        print(f"⚠️  第 {line_num} 行格式不正確，跳過")
                except json.JSONDecodeError:
                    print(f"⚠️  第 {line_num} 行 JSON 格式錯誤，跳過")
                    continue
        
        print(f"✅ 成功讀取 {len(all_data)} 筆資料")
        
    except Exception as e:
        print(f"❌ 讀取檔案時發生錯誤: {e}")
        return False
    
    # 檢查資料量
    if len(all_data) < num_samples:
        print(f"⚠️  資料集只有 {len(all_data)} 筆，少於要求的 {num_samples} 筆")
        num_samples = len(all_data)
    
    # 隨機抽樣
    print(f"🎲 隨機抽取 {num_samples} 筆資料...")
    selected_data = random.sample(all_data, num_samples)
    
    # 寫入輸出檔案
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for data in selected_data:
                json.dump(data, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✅ 成功創建 {output_file}")
        
        # 統計資訊
        prompt_lengths = [len(data['prompt']) for data in selected_data]
        target_lengths = [len(data['target']) for data in selected_data]
        
        print(f"\n📊 統計資訊:")
        print(f"  📝 Prompt 平均長度: {sum(prompt_lengths) / len(prompt_lengths):.1f} 字")
        print(f"  📄 Target 平均長度: {sum(target_lengths) / len(target_lengths):.1f} 字")
        print(f"  📝 Prompt 長度範圍: {min(prompt_lengths)} - {max(prompt_lengths)} 字")
        print(f"  📄 Target 長度範圍: {min(target_lengths)} - {max(target_lengths)} 字")
        
        return True
        
    except Exception as e:
        print(f"❌ 寫入檔案時發生錯誤: {e}")
        return False

def main():
    """主函數"""
    
    print("🚀 WritingPrompt 測試資料提取工具")
    print("=" * 50)
    
    # 檢查命令列參數
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "test_500.jsonl"
        num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    else:
        # 互動模式
        print("請輸入以下資訊：")
        input_file = input("📁 輸入檔案路徑: ").strip()
        output_file = input("📁 輸出檔案路徑 (預設: test_500.jsonl): ").strip() or "test_500.jsonl"
        num_samples = int(input("📊 樣本數量 (預設: 500): ") or "500")
    
    # 執行提取
    success = create_test_500(input_file, output_file, num_samples)
    
    if success:
        print(f"\n🎉 完成！測試資料已保存到 {output_file}")
        print(f"💡 您現在可以使用這個檔案來測試您的模型了")
    else:
        print("\n❌ 提取失敗，請檢查輸入檔案和路徑")

if __name__ == "__main__":
    main() 