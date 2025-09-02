import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
import pickle
import math

num_proc = 10 # 可根據 CPU 核心調整(指令:nproc)

if __name__ == "__main__":
    # === Step 1: 只載入 train 和 val 資料 ===
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    data_files = {
        "train": os.path.join(root, "dataset/SlimPajama-1B/SlimPajama-1B_train.jsonl"),
        "val": os.path.join(root, "dataset/SlimPajama-1B/SlimPajama-1B_validation.jsonl"),
    }
    dataset = load_dataset("json", data_files=data_files)

    # === Step 2: 載入你自己的 tokenizer ===
    raw_tokenizer = Tokenizer.from_file("/root/Thesis/tokenizer/slim_tokenizer/tokenizer.json")
    print("raw_tokenizer:", raw_tokenizer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer)
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>"
    })
    print("Special tokens:", tokenizer.special_tokens_map)
    print(tokenizer.all_special_tokens)
    print(tokenizer.vocab_size)
    
    # 驗證token ID映射
    print(f"pad_token_id: {tokenizer.pad_token_id}")
    print(f"bos_token_id: {tokenizer.bos_token_id}")
    print(f"eos_token_id: {tokenizer.eos_token_id}")
    print(f"unk_token_id: {tokenizer.unk_token_id}")
    
    # 驗證與TTT配置的一致性
    expected_mapping = {
        "pad_token_id": 0,
        "bos_token_id": 2, 
        "eos_token_id": 3,
        "unk_token_id": 1
    }
    
    actual_mapping = {
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "unk_token_id": tokenizer.unk_token_id
    }
    
    for key, expected in expected_mapping.items():
        actual = actual_mapping[key]
        if actual != expected:
            print(f"⚠️ 警告: {key} = {actual}, 期望 = {expected}")
        else:
            print(f"✅ {key} = {actual} (正確)")
    
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer is missing eos_token_id. Please check tokenizer training.")
    print("eos_id:", eos_id)

    # === Step 3: 定義 tokenization 處理函式 ===
    def process(example):
        text = example["text"].strip()
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(eos_id)
        return {"ids": ids, "len": len(ids)}

    # === Step 4: 對 train 和 val 做 tokenize ===
    tokenized = dataset.map(
        process,
        remove_columns=["text"],
        desc="Tokenizing slim splits",
        num_proc=num_proc
    )

    # === Step 5: 儲存成JAX風格的長陣列格式 ===
    # 🔧 修改：移除預切分，只保存長陣列
    block_size = 2048  # 匹配JAX main和訓練配置
    for split in ["train", "val"]:
        dset = tokenized[split]
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = f"{split}.bin"  # 只保存長陣列
        dtype = np.uint16

        if tokenizer.vocab_size >= 2**16:
            raise ValueError("Tokenizer vocab too large for uint16. Use uint32 instead.")

        # 創建 memmap 檔案 - 只保存長陣列
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 128

        # 寫入 tokenized 資料
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        
        print(f"✅ {split} 長陣列已保存: {filename}")
        print(f"  總tokens: {arr_len:,}")
        print(f"  預估序列數: {arr_len // block_size:,} (每序列{block_size} tokens)")

    # === Step 6: 建立 meta.pkl ===
    meta = {
        "vocab_size": tokenizer.vocab_size,
        "block_size": block_size
    }
    with open("meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"✅ JAX風格資料格式已生成:")
    print(f"  - train.bin: 長陣列格式")
    print(f"  - val.bin: 長陣列格式") 
    print(f"  - meta.pkl: 包含vocab_size={tokenizer.vocab_size}, block_size={block_size}")
    print(f"  - 支援滑動窗口切分，與JAX main一致")