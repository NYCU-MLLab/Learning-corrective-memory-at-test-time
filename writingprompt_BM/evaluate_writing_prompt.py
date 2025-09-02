import json
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from collections import Counter
import re
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import nltk
from mauve import compute_mauve

# 下載必要的 NLTK 資料
nltk.download("punkt")

def load_data(file_path):
    """載入測試資料"""
    prompts = []
    targets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data['prompt'])
            targets.append(data['target'])
    return prompts, targets

def calculate_rouge_l(predictions, references):
    """計算 ROUGE-L 分數"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(pred, ref)
        scores.append(score['rougeL'].fmeasure)
    
    return np.mean(scores)

def calculate_bertscore(predictions, references):
    """計算 BERTScore"""
    P, R, F1 = bert_score(predictions, references, lang='en', verbose=True)
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }

def compute_diversity(text: str, n_values=(2, 3, 4)) -> float:
    """計算單個文本的多樣性"""
    tokens = nltk.word_tokenize(text)
    diversity_scores = []

    for n in n_values:
        n_gram_list = list(ngrams(tokens, n))
        total_n_grams = len(n_gram_list)

        if total_n_grams == 0:
            continue  
        unique_n_grams = len(set(n_gram_list))
        diversity = unique_n_grams / total_n_grams
        diversity_scores.append(diversity)

    if not diversity_scores:
        return 0.0 

    return np.prod(diversity_scores) ** (1 / len(diversity_scores))

def compute_average_diversity(text_list: list[str], n_values=(2, 3, 4)) -> float:
    """計算文本列表的平均多樣性"""
    if not text_list:
        return 0.0

    scores = [compute_diversity(text, n_values=n_values) for text in text_list]
    return np.mean(scores)

def calculate_mauve(predictions, references):
    """計算 MAUVE 分數"""
    # 將文本轉換為特徵向量 (這裡使用簡單的詞頻向量作為示例)
    def text_to_features(texts):
        # 簡單的詞頻特徵
        all_words = set()
        for text in texts:
            text = re.sub(r'<newline>', ' ', text)
            words = word_tokenize(text.lower())
            all_words.update(words)
        
        # 限制詞彙表大小以避免維度問題
        word_counts = Counter()
        for text in texts:
            text = re.sub(r'<newline>', ' ', text)
            words = word_tokenize(text.lower())
            word_counts.update(words)
        
        # 只保留最常見的詞彙（限制在 1000 個以內）
        vocab_size = min(1000, len(word_counts))
        most_common_words = [word for word, count in word_counts.most_common(vocab_size)]
        word_to_idx = {word: idx for idx, word in enumerate(most_common_words)}
        
        features = []
        for text in texts:
            text = re.sub(r'<newline>', ' ', text)
            words = word_tokenize(text.lower())
            feature_vector = [0] * vocab_size
            for word in words:
                if word in word_to_idx:
                    feature_vector[word_to_idx[word]] += 1
            features.append(feature_vector)
        
        return np.array(features)
    
    try:
        # 合併所有文本來建立統一的詞彙表
        all_texts = predictions + references
        p_features = text_to_features(predictions)
        q_features = text_to_features(references)
        
        # 確保兩個特徵矩陣有相同的維度
        if p_features.shape[1] != q_features.shape[1]:
            min_dim = min(p_features.shape[1], q_features.shape[1])
            p_features = p_features[:, :min_dim]
            q_features = q_features[:, :min_dim]
        
        print(f"MAUVE 特徵維度: P={p_features.shape}, Q={q_features.shape}")
        
        mauve_score = compute_mauve(p_features, q_features)
        return mauve_score.mauve
    except Exception as e:
        print(f"MAUVE 計算錯誤: {e}")
        return None

def evaluate_model(predictions, references):
    """評估模型輸出"""
    print("開始評估...")
    
    # ROUGE-L
    print("計算 ROUGE-L...")
    rouge_l_score = calculate_rouge_l(predictions, references)
    print(f"ROUGE-L: {rouge_l_score:.4f}")
    
    # BERTScore
    print("計算 BERTScore...")
    bert_scores = calculate_bertscore(predictions, references)
    print(f"BERTScore Precision: {bert_scores['precision']:.4f}")
    print(f"BERTScore Recall: {bert_scores['recall']:.4f}")
    print(f"BERTScore F1: {bert_scores['f1']:.4f}")
    
    # Diversity
    print("計算 Diversity...")
    diversity_score = compute_average_diversity(predictions, n_values=(2, 3, 4))
    print(f"Diversity Score: {diversity_score:.4f}")
    
    # MAUVE
    print("計算 MAUVE...")
    mauve_score = calculate_mauve(predictions, references)
    if mauve_score is not None:
        print(f"MAUVE: {mauve_score:.4f}")
    else:
        print("MAUVE: 計算失敗")
    
    return {
        'rouge_l': rouge_l_score,
        'bertscore': bert_scores,
        'diversity': diversity_score,
        'mauve': mauve_score
    }

def save_predictions(prompts, predictions, targets, filename='model_predictions.json'):
    """保存模型預測結果"""
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
    print(f"預測結果已保存到 {filename}")

def load_predictions(filename='ttt_predictions.json'):
    """載入已保存的模型預測結果"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        prompts = [item['prompt'] for item in data]
        predictions = [item['prediction'] for item in data]
        targets = [item['target'] for item in data]
        
        print(f"已載入 {len(predictions)} 個預測結果")
        return prompts, predictions, targets
    except FileNotFoundError:
        print(f"找不到檔案 {filename}")
        return None, None, None

def main():
    # 載入測試資料
    print("載入測試資料...")
    prompts, targets = load_data('test_50.jsonl')
    print(f"載入了 {len(prompts)} 個測試樣本")
    
    # 檢查是否有已保存的預測結果
    saved_prompts, saved_predictions, saved_targets = load_predictions()
    
    if saved_predictions is not None:
        print("找到已保存的預測結果，是否要使用？(y/n)")
        use_saved = input().lower().strip()
        
        if use_saved == 'y':
            prompts, predictions, targets = saved_prompts, saved_predictions, saved_targets
            print("使用已保存的預測結果進行評估...")
            results = evaluate_model(predictions, targets)
            return
    
    # 如果沒有保存的結果或選擇重新生成
    print("\n=== 請在這裡放入您的模型預測結果 ===")
    
    # 請將您的模型輸出替換下面的示例
    predictions = [
        "這是示例回答1 - 請替換為您的模型實際輸出",
        "這是示例回答2 - 請替換為您的模型實際輸出",
        # ... 請繼續添加，直到有50個回答
    ]
    
    # 確保有足夠的預測結果
    if len(predictions) < len(prompts):
        print(f"⚠️  警告：您只有 {len(predictions)} 個預測，但需要 {len(prompts)} 個")
        print("請確保您的 predictions 列表包含所有 50 個回答")
    else:
        # 只取前50個
        predictions = predictions[:len(prompts)]
        print(f"✅ 準備評估 {len(predictions)} 個預測結果")
        
        # 執行評估和保存
        results = evaluate_model(predictions, targets)
        save_predictions(prompts, predictions, targets)
    
    # 顯示前幾個樣本供參考
    print("\n前 3 個測試樣本:")
    for i in range(min(3, len(prompts))):
        print(f"\n樣本 {i+1}:")
        print(f"Prompt: {prompts[i]}")
        print(f"Target: {targets[i][:200]}...")

def quick_evaluate_with_saved():
    """快速評估已保存的預測結果"""
    prompts, predictions, targets = load_predictions()
    if predictions is not None:
        results = evaluate_model(predictions, targets)
        return results
    else:
        print("沒有找到已保存的預測結果")
        return None

if __name__ == "__main__":
    main() 