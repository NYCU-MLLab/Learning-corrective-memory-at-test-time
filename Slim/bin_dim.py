import numpy as np
import pickle

with open("meta.pkl", "rb") as f:
    meta = pickle.load(f)
    vocab_size = meta["vocab_size"]

print(f"vocab_size: {vocab_size}") # 50257

train = np.memmap("train.bin", dtype=np.uint16, mode="r")
val = np.memmap("val.bin", dtype=np.uint16, mode="r")

print(f"train.bin 長度: {len(train):,} tokens") # 長度: 943,635,020 tokens
print(f"val.bin 長度:   {len(val):,} tokens") # 長度: 9,101,108 tokens

# 印出前幾個 token ids 看看
print("train 前 10 個 tokens:", train[:10]) 
print("val   前 10 個 tokens:", val[:10])
print("max token id in train.bin:", train.max()) # 50256