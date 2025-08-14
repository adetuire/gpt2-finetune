# src/evaluate.py  (or evaluate.py at repo root)
import json, math, os, typer
from pathlib import Path
from typing import List
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = typer.Typer(help="Evaluate perplexity for a Causal LM")

def _ppl_fallback(model, tok, texts: List[str], max_len: int, batch_size: int) -> float:
    """Token-level NLL average → perplexity, no `evaluate` dependency."""
    model.eval()
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            # labels = input_ids, ignore padding
            labels = input_ids.masked_fill(attn.eq(0), -100)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            # out.loss is mean over non-ignored tokens
            # recover sum over tokens: loss * num_non_ignored
            non_ignored = (labels != -100).sum().item()
            total_nll += out.loss.item() * non_ignored
            total_tokens += non_ignored
    return float(math.exp(total_nll / max(1, total_tokens)))

@app.command()
def perplexity(
    model: str = typer.Option(..., help="Local path or Hub id (e.g., gpt2_export or username/model)"),
    dataset_name: str = "imdb",
    split: str = "test",
    n_texts: int = 2000,
    max_len: int = 512,
    batch_size: int = 4,
    stride: int = 256,   # kept for API compat; unused by fallback
    cache_dir: str = ".cache",
    out_json: str = "metrics.json",
):
    # Let user control caches both in Colab (/content/cache) and locally
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)

    # Load model/tokenizer (works for local folders or Hub ids)
    tok = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
    mod = AutoModelForCausalLM.from_pretrained(model, cache_dir=cache_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mod.to(device)

    ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    texts = ds["text"][:n_texts]

    result = {"n_texts": n_texts, "max_len": max_len, "batch_size": batch_size}

    # Try `evaluate` first; if it fails (e.g., Drive path quirks), use fallback
    try:
        import evaluate as _evaluate  # lazy import
        metric = _evaluate.load("perplexity", module_type="metric", cache_dir=cache_dir)
        # evaluate.perplexity loads the model internally by id/path
        res = metric.compute(
            model_id=str(model),
            add_start_token=True,   # GPT-2 has no BOS
            batch_size=batch_size,
            max_length=max_len,
            stride=stride,
            data=texts,
        )
        mean_ppl = float(sum(res["perplexities"]) / len(res["perplexities"]))
        method = "evaluate.perplexity"
    except Exception as e:
        mean_ppl = _ppl_fallback(mod, tok, texts, max_len=max_len, batch_size=batch_size)
        method = f"fallback_token_level ({type(e).__name__}: {e})"

    out = {"mean_perplexity": mean_ppl, "method": method, **result}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    app()
