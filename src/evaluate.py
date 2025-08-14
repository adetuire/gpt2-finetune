# src/evaluate.py
import json, math, os, typer
from pathlib import Path
from typing import List
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def _ppl_token_level(model, tok, texts: List[str], max_len: int, batch_size: int) -> float:
    """Token-level NLL average -> perplexity (no external 'evaluate' pkg)."""
    model.eval()
    device = next(model.parameters()).device
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            labels = input_ids.masked_fill(attn.eq(0), -100)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            non_ignored = (labels != -100).sum().item()
            total_nll += out.loss.item() * non_ignored
            total_tokens += non_ignored
    return float(math.exp(total_nll / max(1, total_tokens)))

def main(
    model: str = typer.Option(..., help="HF Hub id or local path, e.g. adetuire1/gpt2-imdb-tuned"),
    dataset_name: str = "imdb",
    split: str = "test",
    n_texts: int = 2000,
    max_len: int = 512,
    batch_size: int = 4,
    cache_dir: str = ".cache",
    out_json: str = "results/ppl.json",
):
    # Cache dirs work on Colab and local
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)

    tok = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
    # GPT-2 has no pad token; set it so padding works
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    mod = AutoModelForCausalLM.from_pretrained(model, cache_dir=cache_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mod.to(device)

    ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    texts = ds["text"][:n_texts]

    ppl = _ppl_token_level(mod, tok, texts, max_len=max_len, batch_size=batch_size)
    out = {
        "mean_perplexity": ppl,
        "method": "token_level_fallback",
        "n_texts": n_texts,
        "max_len": max_len,
        "batch_size": batch_size,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    typer.run(main)
