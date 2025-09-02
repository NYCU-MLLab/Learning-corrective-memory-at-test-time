import argparse
import json
import math
import os
import sys
from typing import Optional, Tuple, Any, List

import torch


def maybe_get_device(device_arg: Optional[str]) -> str:
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_dtype(model, prefer_bf16: bool = True):
    # Keep weights as-is; use autocast for speed/precision if needed.
    # Explicit casting is optional; Longhorn training使用 bf16/fp16/32 都可。
    return


def load_model(ckpt_path: str, device: str = "cuda"):
    # 確保專案根目錄在 sys.path 中，才能匯入 longhorn 模組
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from longhorn.models.longhorn import LonghornConfig, LonghornLM

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    # 期待 train.py 存了 'model_config' 與 'model'
    model_cfg_dict = checkpoint.get("model_config")
    if model_cfg_dict is None:
        raise ValueError("model_config not found in checkpoint")

    config = LonghornConfig(**model_cfg_dict)
    model = LonghornLM(config)
    model.to(device)

    state = checkpoint.get("model")
    if state is None:
        raise ValueError("state_dict 'model' not found in checkpoint")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0:
        print(f"[warn] missing keys: {len(missing)} (showing first 5) -> {missing[:5]}")
    if len(unexpected) > 0:
        print(f"[warn] unexpected keys: {len(unexpected)} (showing first 5) -> {unexpected[:5]}")

    model.eval()
    return model, config


def get_tokenizer():
    # 保留 GPT-2 BPE 後備方案（若專案 tokenizer 不存在時）。
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("gpt2")

        class Tok:
            def encode(self, s: str):
                return enc.encode(s)

            def decode(self, ids):
                return enc.decode(ids)

        return Tok()
    except Exception:
        return None


def load_project_tokenizer(tokenizer_dir: str):
    """優先從專案提供的 tokenizer 目錄載入。

    1) 嘗試 transformers.AutoTokenizer（若可用）
    2) 退回 tokenizers.Tokenizer（僅用 tokenizer.json）

    回傳 (tok, eos_id)；若無法載入則回傳 (None, None)
    tok 需具備 encode(text)->ids, decode(ids)->text 介面。
    """
    resolved_dir = tokenizer_dir
    if not os.path.isabs(resolved_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        cand = os.path.join(project_root, tokenizer_dir)
        if os.path.isdir(cand):
            resolved_dir = cand

    if not os.path.isdir(resolved_dir):
        return None, None

    # 讀 special tokens map 取得 eos 內容
    eos_content = None
    stm_path = os.path.join(resolved_dir, "special_tokens_map.json")
    if os.path.exists(stm_path):
        try:
            with open(stm_path, "r", encoding="utf-8") as f:
                stm = json.load(f)
            eos = stm.get("eos_token")
            if isinstance(eos, dict):
                eos_content = eos.get("content")
        except Exception:
            pass

    # 1) transformers
    try:
        from transformers import AutoTokenizer  # type: ignore

        hf_tok = AutoTokenizer.from_pretrained(resolved_dir, use_fast=True)
        eos_id = None
        if eos_content is not None:
            try:
                eos_id = hf_tok.convert_tokens_to_ids(eos_content)
            except Exception:
                eos_id = hf_tok.eos_token_id

        class HfTok:
            def encode(self, s: str):
                return hf_tok.encode(s, add_special_tokens=False)

            def decode(self, ids):
                return hf_tok.decode(ids, skip_special_tokens=True)

        return HfTok(), eos_id
    except Exception:
        pass

    # 2) tokenizers
    try:
        from tokenizers import Tokenizer  # type: ignore
        tok_json = os.path.join(resolved_dir, "tokenizer.json")
        if not os.path.exists(tok_json):
            return None, None
        tk = Tokenizer.from_file(tok_json)

        eos_id = None
        if eos_content is not None:
            try:
                eos_id = tk.token_to_id(eos_content)
            except Exception:
                eos_id = None

        class TkTok:
            def encode(self, s: str):
                return tk.encode(s).ids

            def decode(self, ids):
                try:
                    return tk.decode(ids)
                except Exception:
                    # 最簡回退
                    return " ".join(map(str, ids))

        return TkTok(), eos_id
    except Exception:
        pass

    return None, None


def _apply_repetition_penalty(logits: torch.Tensor, generated_ids: List[int], penalty: float,
                              window: Optional[int] = None):
    if penalty is None or penalty <= 1.0:
        return logits
    if not generated_ids:
        return logits
    window_ids = generated_ids if window is None else generated_ids[-window:]
    unique_ids = list(set(window_ids))
    # 在 logits 上減去 log(penalty)，等效於降低這些 token 的機率
    penalty_val = math.log(penalty)
    logits[..., unique_ids] -= penalty_val
    return logits


def _enforce_no_repeat_ngram(logits: torch.Tensor, generated_ids: List[int], n: int):
    if n is None or n <= 0:
        return logits
    seq_len = len(generated_ids)
    if seq_len < n - 1:
        return logits
    prefix = tuple(generated_ids[-(n - 1):])
    blocked: set = set()
    for i in range(seq_len - n + 1):
        if tuple(generated_ids[i:i + n - 1]) == prefix:
            blocked.add(generated_ids[i + n - 1])
    if blocked:
        logits[..., list(blocked)] = -float("inf")
    return logits


def _top_k_filter(logits: torch.Tensor, top_k: int):
    if top_k is None or top_k <= 0:
        return logits
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float("inf")
    return logits


def _top_p_filter(logits: torch.Tensor, top_p: float):
    if top_p is None or top_p <= 0.0 or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)
    # 保留累積機率 <= top_p 的 tokens
    mask = cumprobs > top_p
    # 確保至少保留一個 token
    mask[..., 0] = False
    sorted_logits[mask] = -float("inf")
    # 還原原始順序
    logits.fill_(-float("inf"))
    logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return logits


@torch.no_grad()
def generate(model, prompt_ids: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0,
             top_k: Optional[int] = None, top_p: Optional[float] = None,
             device: str = "cuda", context_window: int = 1024,
             eos_id: Optional[int] = None, repetition_penalty: float = 1.0,
             no_repeat_ngram_size: int = 0, repetition_window: int = 256):
    # model: LonghornLM（外層）
    # 貪婪/溫度+top-k 取樣；每步全長前向，簡單穩定。
    model.eval()
    x = prompt_ids.to(device)
    generated_ids: List[int] = prompt_ids[0].tolist()
    for _ in range(max_new_tokens):
        # 只保留最近 context_window 的 token
        x_cond = x[:, -context_window:]
        logits, _ = model(x_cond, targets=None)
        logits = logits[:, -1, :]  # 最後一個 token 的分佈

        # 重複懲罰與 n-gram 禁止
        logits = _apply_repetition_penalty(logits, generated_ids, repetition_penalty, window=repetition_window)
        logits = _enforce_no_repeat_ngram(logits, generated_ids, no_repeat_ngram_size)

        # top-k / top-p 過濾
        logits = _top_k_filter(logits, top_k)
        logits = _top_p_filter(logits, top_p)

        # 溫度縮放與採樣
        if temperature == 0.0:
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_id), dim=1)
        generated_ids.append(int(next_id.item()))
        if eos_id is not None and int(next_id.item()) == int(eos_id):
            break
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint .pt (e.g., longhorn/results/..._best.pt)")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--prompt", type=str, default="Hello, my name is")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0, help="0 表示不啟用 top-k")
    parser.add_argument("--top_p", type=float, default=0.9, help="核采樣機率閾值 (0 表示不啟用)")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help=">1.0 懲罰重複 token")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="禁止重複的 n-gram 大小 (0 不啟用)")
    parser.add_argument("--repetition_window", type=int, default=256, help="重複懲罰的視窗大小")
    parser.add_argument("--context_window", type=int, default=1024)
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer/slim_tokenizer",
                        help="專案 tokenizer 目錄（含 tokenizer.json 等）")
    args = parser.parse_args()

    device = maybe_get_device(args.device)
    print(f"[info] using device: {device}")

    model, config = load_model(args.ckpt, device)
    print(f"[info] vocab_size={config.vocab_size}, d_model={config.d_model}, n_layer={config.n_layer}")

    tok, eos_id = load_project_tokenizer(args.tokenizer_dir)
    if tok is None:
        print(f"[warn] 專案 tokenizer 目錄不可用: {args.tokenizer_dir}，改用 GPT-2 BPE 後備方案。")
        tok = get_tokenizer()
        if tok is None:
            print("[warn] 無 tokenizer 可用。將以 token id 形式進行推理與輸出。建議安裝 transformers 或 tokenizers。\n")
            prompt_ids = torch.tensor([[50256]], dtype=torch.long)
        else:
            prompt_ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
    else:
        prompt_ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
        if prompt_ids.numel() == 0:
            prompt_ids = torch.tensor([[0]], dtype=torch.long)

    out_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=(args.top_k if args.top_k > 0 else None),
        device=device,
        context_window=args.context_window,
        eos_id=eos_id,
        top_p=(args.top_p if args.top_p and args.top_p > 0 else None),
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_window=args.repetition_window,
    )

    # 若 tok 有 decode 介面就解碼
    try:
        text = tok.decode(out_ids[0].tolist())
        print("\n========== Generated Text ==========")
        print(text)
        print("===================================\n")
    except Exception:
        print("\n[info] generated token ids:")
        print(out_ids[0].tolist())


if __name__ == "__main__":
    main()

