# src/evaluate.py
import json, typer
from pathlib import Path
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM

app = typer.Typer()

@app.command()
def perplexity(
    model: str = typer.Option(..., help="Local path or HF Hub id"),
    n_texts: int = 2000,
    max_len: int = 512,
    stride: int = 256,
    batch_size: int = 4,
    out_json: str = "metrics.json",
):
    # Smoke-load (also supports local path)
    _ = AutoModelForCausalLM.from_pretrained(model)
    _ = AutoTokenizer.from_pretrained(model)
    ds = load_dataset("imdb", split="test", cache_dir="/content/cache")
    texts = ds["text"][:n_texts]

    metric = evaluate.load("perplexity", module_type="metric")
    res = metric.compute(
        model_id=str(model),
        add_start_token=True,       # GPT-2 has no <BOS>
        batch_size=batch_size,
        max_length=max_len,
        stride=stride,
        data=texts,
    )
    mean_ppl = float((sum(res["perplexities"]) / len(res["perplexities"])) ** 1)  # keep raw mean

    out = {"mean_perplexity": mean_ppl, "n_texts": n_texts, "max_len": max_len, "stride": stride}
    Path(out_json).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    app()
