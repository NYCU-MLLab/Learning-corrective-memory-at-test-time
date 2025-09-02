import json

def save_model_predictions(prompts, predictions, targets, filename='model_predictions.json'):
    """保存模型預測結果到 JSON 檔案"""
    data = []
    for i, (prompt, pred, target) in enumerate(zip(prompts, predictions, targets)):
        data.append({
            'id': i,
            'prompt': prompt,
            'prediction': pred,
            'target': target
        })
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ 預測結果已保存到 {filename}")
    print(f"📊 共保存了 {len(predictions)} 個預測結果")

def load_model_predictions(filename='model_predictions.json'):
    """從 JSON 檔案載入模型預測結果"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        prompts = [item['prompt'] for item in data]
        predictions = [item['prediction'] for item in data]
        targets = [item['target'] for item in data]
        
        print(f"✅ 已載入 {len(predictions)} 個預測結果")
        return prompts, predictions, targets
    except FileNotFoundError:
        print(f"❌ 找不到檔案 {filename}")
        return None, None, None

# 使用範例：
if __name__ == "__main__":
    # 假設您有模型預測結果
    # prompts = ["問題1", "問題2", ...]
    # predictions = ["回答1", "回答2", ...]
    # targets = ["標準答案1", "標準答案2", ...]
    
    # 保存預測結果
    # save_model_predictions(prompts, predictions, targets)
    
    # 載入預測結果
    # loaded_prompts, loaded_predictions, loaded_targets = load_model_predictions()
    
    print("這個腳本用於保存和載入模型預測結果")
    print("使用方法：")
    print("1. 將您的模型輸出放入 predictions 列表")
    print("2. 調用 save_model_predictions() 保存結果")
    print("3. 下次使用 load_model_predictions() 載入結果") 