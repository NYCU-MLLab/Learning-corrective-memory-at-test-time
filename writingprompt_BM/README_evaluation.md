# WritingPrompt 評估腳本使用說明

## 檔案說明

- `evaluate_writing_prompt.py`: 主要的評估腳本
- `save_predictions.py`: 專門用於保存/載入預測結果的腳本
- `test_50.jsonl`: 前 50 題的測試資料
- `model_predictions.json`: 保存的模型預測結果（運行後會自動生成）

## 使用方法

### 1. 第一次使用（需要模型預測）

1. 將您的模型輸出放入 `predictions` 列表中
2. 取消註解以下代碼：
```python
# 評估和保存
results = evaluate_model(predictions, targets)
save_predictions(prompts, predictions, targets)
```

3. 運行腳本：
```bash
python3 evaluate_writing_prompt.py
```

### 2. 後續使用（使用已保存的預測）

1. 直接運行腳本，會自動檢測是否有已保存的預測結果
2. 選擇 'y' 使用已保存的結果進行評估

### 3. 快速評估（僅評估已保存的結果）

```python
from evaluate_writing_prompt import quick_evaluate_with_saved
results = quick_evaluate_with_saved()
```

## 評估指標

腳本會計算以下指標：

1. **ROUGE-L**: 衡量生成文本與參考文本的長度最長公共子序列
2. **BERTScore**: 使用 BERT 嵌入計算語義相似度
3. **Diversity**: 計算文本的多樣性（使用 n-gram 方法）
4. **MAUVE**: 衡量生成分布與真實分布的相似度

## 資料格式

### 輸入格式
```json
{
  "prompt": "問題描述",
  "target": "標準答案"
}
```

### 保存格式
```json
{
  "id": 0,
  "prompt": "問題描述",
  "prediction": "模型預測",
  "target": "標準答案"
}
```

## 注意事項

1. 首次運行需要安裝相關套件：
```bash
pip install rouge-score bert-score nltk numpy mauve-text
```

2. 確保 `test_50.jsonl` 檔案存在於同一目錄

3. 預測結果會自動保存為 `model_predictions.json`，下次可以直接使用

4. 如果修改了模型，可以刪除 `model_predictions.json` 重新生成 