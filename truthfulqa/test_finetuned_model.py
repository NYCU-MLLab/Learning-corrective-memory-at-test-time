#!/usr/bin/env python3
"""
測試微調後的 PreCo 模型
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../longhorn'))

import torch
import json
import re
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from models.PreCo import PreCoNewModel, PreCoNewConfig

# BERTScore 評估
try:
    from bert_score import score
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("警告: bert-score 未安裝，將跳過 BERTScore 評估")
    print("請運行: pip install bert-score")
    BERTSCORE_AVAILABLE = False


def load_model(model_path: str, device):
    """載入微調後的模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # 檢查權重文件格式
    if 'model_config' in checkpoint:
        # 使用權重文件中的配置
        config_dict = checkpoint['model_config']
        config = PreCoNewConfig(
            vocab_size=config_dict.get('vocab_size', 50257),
            d_model=config_dict.get('d_model', 512),
            n_layer=config_dict.get('n_layer', 12),
            d_state=config_dict.get('d_state', 8),
            d_conv=config_dict.get('d_conv', 3),
            expand=config_dict.get('expand', 6),
            ttt_num_heads=config_dict.get('ttt_num_heads', 8),
            ttt_num_layers=config_dict.get('ttt_num_layers', 1),
            dropout=config_dict.get('dropout', 0.1)
        )
    else:
        # 使用預設配置
        config = PreCoNewConfig()
    
    model = PreCoNewModel(config)
    
    # 載入模型權重
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model


def generate_answer(model, tokenizer, question: str, max_length: int = 20, device='cpu'):
    """生成答案"""
    prompt = f"Q: {question}\nA:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    
    # 自回歸生成
    model.eval()
    with torch.no_grad():
        generated_ids = input_ids.clone()
        
        for step in range(max_length):
            # 前向傳播
            outputs = model(input_ids=generated_ids, targets=None)
            
            # 處理輸出
            if isinstance(outputs, tuple):
                logits, _ = outputs
            else:
                logits = outputs
            
            # 獲取下一個 token
            next_token_logits = logits[0, -1, :]
            
            # 過濾特殊 tokens
            special_token_ids = []
            if tokenizer.pad_token_id is not None:
                special_token_ids.append(tokenizer.pad_token_id)
            if tokenizer.eos_token_id is not None:
                special_token_ids.append(tokenizer.eos_token_id)
            if tokenizer.bos_token_id is not None:
                special_token_ids.append(tokenizer.bos_token_id)
            if tokenizer.unk_token_id is not None:
                special_token_ids.append(tokenizer.unk_token_id)
            
            for token_id in special_token_ids:
                if token_id is not None:
                    next_token_logits[token_id] = float('-inf')
            
            # 應用溫度採樣（更保守的設定）
            temperature = 0.05
            next_token_logits = next_token_logits / temperature
            
            # 應用 top-k 採樣（更少的選項）
            top_k = 3
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 應用 top-p 採樣（更保守）
            top_p = 0.4
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')
            
            # 採樣下一個 token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 添加到生成序列
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
            
            # 檢查是否生成了結束符號
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # 檢查是否生成了換行符（可能表示答案結束）
            if next_token.item() in [10, 13]:  # \n 或 \r
                break
            
            # 簡單的重複檢測（只檢查連續相同 token）
            if step > 2:
                if generated_ids[0, -1] == generated_ids[0, -2]:
                    break
    
    # 解碼
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # 提取答案部分
    answer = generated_text[len(prompt):].strip()
    
    return answer


def calculate_rouge_l(prediction, reference):
    """計算 ROUGE-L 分數"""
    def get_lcs_length(str1, str2):
        """計算最長公共子序列長度"""
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    # 清理文本（保留更多信息）
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()
    
    pred_clean = clean_text(prediction)
    ref_clean = clean_text(reference)
    
    if not pred_clean or not ref_clean:
        return 0.0
    
    lcs_length = get_lcs_length(pred_clean, ref_clean)
    
    if lcs_length == 0:
        return 0.0
    
    # 計算 ROUGE-L F1 分數
    precision = lcs_length / len(pred_clean.split())
    recall = lcs_length / len(ref_clean.split())
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def load_truthfulqa_val_data(file_path: str, max_samples: int = None):
    """載入 TruthfulQA 驗證集數據"""
    questions = []
    references = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
                
            data = json.loads(line.strip())
            # 從 prompt 中提取問題
            prompt = data['prompt']
            if prompt.startswith('Q: ') and '\nA:' in prompt:
                question = prompt[3:prompt.find('\nA:')].strip()
            else:
                question = prompt.strip()
            
            # 獲取標準答案
            reference = data['completion'].strip()
            
            questions.append(question)
            references.append(reference)
    
    return questions, references


def main():
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    # 載入模型
    model_path = "/root/Thesis/longhorn/results/slim_results/preco_136m_8000iter_7.14_best/Slim_2.1_block1024_preco_best.pt"
    print(f"載入模型: {model_path}")
    model = load_model(model_path, device)
    
    # 載入 tokenizer
    raw_tokenizer = Tokenizer.from_file("/root/Thesis/tokenizer/slim_tokenizer/tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer)
    
    # 正確設置特殊 tokens
    vocab = tokenizer.get_vocab()
    
    # 設置 EOS token
    if "</s>" in vocab:
        tokenizer.eos_token = "</s>"
        tokenizer.eos_token_id = vocab["</s>"]
    else:
        tokenizer.eos_token_id = tokenizer.vocab_size - 1
    
    # 設置 PAD token
    if "<pad>" in vocab:
        tokenizer.pad_token = "<pad>"
        tokenizer.pad_token_id = vocab["<pad>"]
    else:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 設置其他特殊 tokens
    if "<s>" in vocab:
        tokenizer.bos_token = "<s>"
        tokenizer.bos_token_id = vocab["<s>"]
    
    if "<unk>" in vocab:
        tokenizer.unk_token = "<unk>"
        tokenizer.unk_token_id = vocab["<unk>"]
    
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"PAD token ID: {tokenizer.pad_token_id}")
    
    # 載入 TruthfulQA 驗證集
    val_file_path = "val.jsonl"
    print(f"載入驗證集: {val_file_path}")
    
    # 可以設置最大樣本數來控制測試時間
    max_samples = 50  # 可以調整：None=全部，50=前50個，100=前100個
    test_questions, test_references = load_truthfulqa_val_data(val_file_path, max_samples)
    
    print(f"載入 {len(test_questions)} 個測試問題")
    
    print("\n=== 測試微調後的 PreCo 模型 ===\n")
    
    total_rouge_l = 0.0
    num_questions = len(test_questions)
    successful_generations = 0
    
    for i, (question, reference) in enumerate(zip(test_questions, test_references), 1):
        print(f"問題 {i}/{num_questions}: {question}")
        print(f"標準答案: {reference}")
        
        try:
            # 生成答案
            answer = generate_answer(model, tokenizer, question, max_length=20, device=device)
            print(f"生成答案: {answer}")
            
            # 計算 ROUGE-L 分數
            rouge_l_score = calculate_rouge_l(answer, reference)
            total_rouge_l += rouge_l_score
            successful_generations += 1
            print(f"ROUGE-L 分數: {rouge_l_score:.4f}")
            
        except Exception as e:
            print(f"生成失敗: {e}")
            rouge_l_score = 0.0
            total_rouge_l += rouge_l_score
        
        print("-" * 50)
        
        # 每 10 個問題顯示一次進度
        if i % 10 == 0:
            current_avg = total_rouge_l / i
            print(f"進度: {i}/{num_questions}, 當前平均 ROUGE-L: {current_avg:.4f}")
    
    # 計算最終平均 ROUGE-L 分數
    avg_rouge_l = total_rouge_l / num_questions
    print(f"\n=== 評估結果 ===")
    print(f"總問題數: {num_questions}")
    print(f"成功生成數: {successful_generations}")
    print(f"平均 ROUGE-L 分數: {avg_rouge_l:.4f}")
    
    if successful_generations > 0:
        successful_avg = total_rouge_l / successful_generations
        print(f"成功生成的平均 ROUGE-L 分數: {successful_avg:.4f}")
    
    # BERTScore 評估
    if BERTSCORE_AVAILABLE and successful_generations > 0:
        print(f"\n=== BERTScore 評估 ===")
        try:
            # 收集所有成功生成的預測和參考
            all_predictions = []
            all_references = []
            
            for i, (question, reference) in enumerate(zip(test_questions, test_references), 1):
                try:
                    answer = generate_answer(model, tokenizer, question, max_length=20, device=device)
                    all_predictions.append(answer)
                    all_references.append(reference)
                except Exception as e:
                    print(f"BERTScore 評估時生成失敗 (問題 {i}): {e}")
                    continue
            
            if all_predictions and all_references:
                # 計算 BERTScore
                P, R, F1 = score(all_predictions, all_references, lang="en", model_type="microsoft/deberta-large-mnli")
                
                # 顯示平均結果
                print(f"BERTScore Precision: {P.mean().item():.4f}")
                print(f"BERTScore Recall:    {R.mean().item():.4f}")
                print(f"BERTScore F1:        {F1.mean().item():.4f}")
                
                # 顯示詳細統計
                print(f"BERTScore F1 標準差: {F1.std().item():.4f}")
                print(f"BERTScore F1 最小值: {F1.min().item():.4f}")
                print(f"BERTScore F1 最大值: {F1.max().item():.4f}")
            else:
                print("沒有成功的生成結果可供 BERTScore 評估")
                
        except Exception as e:
            print(f"BERTScore 評估失敗: {e}")
    elif not BERTSCORE_AVAILABLE:
        print("\n=== BERTScore 評估 ===")
        print("跳過 BERTScore 評估 (bert-score 未安裝)")


if __name__ == "__main__":
    main() 