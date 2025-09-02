# TTT model inference with test time training
import torch
import pickle
import torch.nn.functional as F
from transformers import AutoTokenizer
from models.ttt import TTTConfig, TTT

# 1. 載入 tokenizer
print("=" * 50)
print("開始載入 TTT 模型推理...")
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained("/root/Thesis/tokenizer/slim_tokenizer")
print(f"✓ Tokenizer 載入成功，詞彙表大小: {tokenizer.vocab_size}")

# 2. 載入 meta.pkl 取得 vocab_size
with open("data/Slim/meta.pkl", "rb") as f:
    meta = pickle.load(f)
vocab_size = meta["vocab_size"]
print(f"✓ Meta 文件載入成功，詞彙表大小: {vocab_size}")

# 3. 設定 TTT 模型 config（要跟訓練時一致）
config = TTTConfig(
    vocab_size=vocab_size,
    hidden_size=768,              # n_embd
    num_hidden_layers=12,         # n_layer
    num_attention_heads=12,       # n_head
    max_position_embeddings=512,  # block_size
    ttt_base_lr=0.1,
    mini_batch_size=8,
    use_gate=True,
    ttt_layer_type="linear",
    scan_checkpoint_group_size=4,
    dropout=0.1
)

print(f"✓ TTT 配置創建成功:")
print(f"  - 隱藏層大小: {config.hidden_size}")
print(f"  - 層數: {config.num_hidden_layers}")
print(f"  - 注意力頭數: {config.num_attention_heads}")
print(f"  - 最大位置編碼: {config.max_position_embeddings}")
print(f"  - TTT 基礎學習率: {config.ttt_base_lr}")

model = TTT(config)
print(f"✓ TTT 模型初始化完成")

# 4. 載入checkpoint（使用最佳模型）
checkpoint_path = "results/slim_results/TTT_131.3M_10000iter/Slim_0.7_block512_ttt_best.pt"
print(f"\n嘗試載入 checkpoint: {checkpoint_path}")

try:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    print(f"✓ 成功載入checkpoint: {checkpoint_path}")
    print(f"  Checkpoint keys: {list(ckpt.keys())}")
    
    # 載入模型權重
    model.load_state_dict(ckpt["model"])
    print(f"✓ 模型參數載入成功")
    print(f"  - 訓練迭代數: {ckpt['iter_num']}")
    print(f"  - 最佳驗證損失: {ckpt.get('best_val_loss', 'N/A')}")
    if 'val_loss' in ckpt:
        print(f"  - 當前驗證損失: {ckpt['val_loss']:.4f}")
    
    # 如果有 TTT 特定參數，也載入它們
    if 'ttt_params' in ckpt:
        print("✓ 載入 TTT 特定參數...")
        ttt_param_count = 0
        for layer_name, params in ckpt['ttt_params'].items():
            layer_idx = int(layer_name.split('_')[1])
            attn = model.transformer.h[layer_idx].attn
            attn.ttt_dense_0.data = params['ttt_dense_0']
            attn.ttt_bias_0.data = params['ttt_bias_0']
            attn.learnable_token_idx.data = params['learnable_token_idx']
            attn.learnable_ttt_lr.load_state_dict(params['learnable_ttt_lr'])
            attn.ttt_norm.load_state_dict(params['ttt_norm'])
            attn.post_norm.load_state_dict(params['post_norm'])
            ttt_param_count += 1
        print(f"✓ TTT 特定參數載入完成，共 {ttt_param_count} 層")
    else:
        print("⚠ 未找到 TTT 特定參數，使用默認初始化")
    
except FileNotFoundError:
    print(f"✗ 找不到 checkpoint: {checkpoint_path}")
    # 如果沒有best模型，嘗試載入最新的checkpoint
    checkpoint_path = "results/slim_results/TTT_131.3M_10000iter/Slim_0.7_block512_ttt_iter10000.pt"
    print(f"嘗試載入備用 checkpoint: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"✓ 載入備用checkpoint成功")
    except FileNotFoundError:
        print("✗ 找不到任何可用的checkpoint文件，請檢查路徑")
        exit(1)

# 5. 移到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✓ 使用設備: {device}")
if device.type == 'cuda':
    print(f"  - GPU 名稱: {torch.cuda.get_device_name()}")
    print(f"  - GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

model.to(device)

# 確保模型使用正確的數據類型（使用 float32 以避免類型問題）
model = model.to(torch.float32)
print(f"✓ 模型數據類型設置為: torch.float32")

# 計算模型參數數量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✓ 模型參數統計:")
print(f"  - 總參數數: {total_params:,}")
print(f"  - 可訓練參數數: {trainable_params:,}")

# 6. 設置 TTT 推理模式（啟用 test time training）
model.train()  # 保持訓練模式以啟用 TTT
print(f"✓ 模型設置為訓練模式以啟用 Test Time Training")

# 7. 輸入 prompt
# 改進 prompt 格式，添加更清晰的對話結構
input_text = "Human: Hello, I'm sad. I don't know how to talk to my friend. Can you help me?\n\nAssistant:"

# 檢查是否有 BOS token，如果有就添加
if tokenizer.bos_token_id is not None:
    # 手動添加 BOS token
    bos_tokens = torch.tensor([[tokenizer.bos_token_id]], device=device)
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(device)
    input_ids = torch.cat([bos_tokens, input_ids], dim=1)
    print(f"✓ 添加了 BOS token (ID: {tokenizer.bos_token_id})")
else:
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(device)
    print("⚠ 沒有 BOS token")

print(f"\n" + "=" * 50)
print("輸入處理:")
print("=" * 50)
print(f"輸入文本: '{input_text}'")
print(f"輸入token數量: {input_ids.shape[1]}")
print(f"輸入token IDs: {input_ids[0].tolist()}")
print(f"解碼驗證: '{tokenizer.decode(input_ids[0])}'")

# 檢查特殊 tokens
print(f"\n特殊 Token 信息:")
print(f"  - BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
print(f"  - EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
print(f"  - PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
print(f"  - UNK token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")

# 8. top-k decoding function
def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k)
    logits[logits < v[:, [-1]]] = -float('Inf')
    return logits

# 9. Test Time Training 生成函數
def generate_with_ttt(model, input_ids, max_new_tokens=100, k=100, temperature=0.8, ttt_lr_mult=1.0):
    """
    使用 Test Time Training 進行文本生成
    """
    print(f"\n開始生成，參數設置:")
    print(f"  - max_new_tokens: {max_new_tokens}")
    print(f"  - top_k: {k}")
    print(f"  - temperature: {temperature}")
    print(f"  - ttt_lr_mult: {ttt_lr_mult}")
    
    generated = input_ids.clone()
    
    for i in range(max_new_tokens):
        # 獲取當前序列（限制在最大位置範圍內）
        current_seq = generated[:, -config.max_position_embeddings:]
        
        # 創建目標序列（下一個token預測）
        if current_seq.shape[1] > 1:
            targets = current_seq[:, 1:].clone()  # 移位一位作為目標
            inputs = current_seq[:, :-1].clone()  # 輸入序列
        else:
            # 如果序列太短，使用當前序列
            targets = None
            inputs = current_seq
        
        # 顯示序列長度信息
        if i < 5 or i % 20 == 0:  # 前5步和每20步顯示詳細信息
            print(f"\nStep {i+1} 詳細信息:")
            print(f"  - 當前序列長度: {current_seq.shape[1]}")
            print(f"  - 輸入序列長度: {inputs.shape[1]}")
            if targets is not None:
                print(f"  - 目標序列長度: {targets.shape[1]}")
        
        # TTT 前向傳播（包含 test time training）
        with torch.set_grad_enabled(True):  # 啟用梯度計算
            if targets is not None:
                # 有目標時，進行 TTT 學習
                logits, loss, ttt_loss = model(inputs, targets, ttt_lr_mult=ttt_lr_mult)
                print(f"Step {i+1}: TTT Loss = {ttt_loss.item():.4f}, LM Loss = {loss.item():.4f}")
            else:
                # 沒有目標時，只進行推理
                logits, _, _ = model(inputs, ttt_lr_mult=ttt_lr_mult)
                print(f"Step {i+1}: 僅推理模式（無目標序列）")
        
        # 生成下一個token
        next_logits = logits[:, -1, :] / temperature
        
        # 顯示 logits 統計信息
        if i < 3:  # 前3步顯示詳細的 logits 信息
            print(f"  - Logits 統計: min={next_logits.min().item():.3f}, max={next_logits.max().item():.3f}, mean={next_logits.mean().item():.3f}")
        
        next_logits = top_k_logits(next_logits, k)
        probs = F.softmax(next_logits, dim=-1)
        
        # 顯示概率分佈信息
        if i < 3:
            top_probs, top_indices = torch.topk(probs, 5)
            print(f"  - Top 5 tokens: {[(tokenizer.decode([idx.item()]), prob.item()) for idx, prob in zip(top_indices[0], top_probs[0])]}")
        
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 顯示生成的 token
        generated_token_text = tokenizer.decode([next_token.item()])
        print(f"  - 生成 token: '{generated_token_text}' (ID: {next_token.item()})")
        
        # 添加到生成序列
        generated = torch.cat((generated, next_token), dim=1)
        
        # 檢查是否生成結束符號
        if next_token.item() == tokenizer.eos_token_id:
            print(f"✓ 在第 {i+1} 步生成結束符號，提前停止")
            break
        
        # 每5步顯示當前生成的文本
        if (i + 1) % 5 == 0:
            current_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"\n--- Step {i+1} 當前生成 ---")
            print(f"'{current_text}'")
            print("--- End ---\n")
    
    return generated

# 10. 執行 TTT 生成
k = 20           # 更小的 top-k
temperature = 0.3  # 更低的溫度
max_new_tokens = 50  # 減少生成數量以便觀察

print(f"\n" + "=" * 50)
print(f"開始 Test Time Training 生成")
print(f"參數: k={k}, temperature={temperature}, max_tokens={max_new_tokens}")
print("=" * 50)

generated = generate_with_ttt(
    model=model,
    input_ids=input_ids,
    max_new_tokens=max_new_tokens,
    k=k,
    temperature=temperature,
    ttt_lr_mult=0.1  # 從 1.0 降到 0.1
)

# 11. 輸出結果
output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
generated_text = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)

print("\n" + "=" * 50)
print("最終結果:")
print("=" * 50)
print(f"\n完整輸出:\n'{output_text}'")
print(f"\n僅生成部分:\n'{generated_text}'")
print(f"\n統計信息:")
print(f"  - 原始輸入長度: {input_ids.shape[1]} tokens")
print(f"  - 生成長度: {generated.shape[1] - input_ids.shape[1]} tokens")
print(f"  - 總長度: {generated.shape[1]} tokens")

print(f"\n✓ Test Time Training 完成！")

# 測試 tokenizer 功能
print(f"\n" + "=" * 30)
print("Tokenizer 測試:")
print("=" * 30)
test_text = "Hello, how are you?"
tokens = tokenizer(test_text, return_tensors="pt")
decoded = tokenizer.decode(tokens["input_ids"][0])
print(f"原始文本: '{test_text}'")
print(f"編碼後解碼: '{decoded}'")
print(f"Token IDs: {tokens['input_ids'][0].tolist()}")