# GPT-2 Fine-Tuning Starter

Fine-tune GPT-2 on any text corpus using Transformers & Datasets.

## Quickstart

### Train (IMDB → GPT-2)
```bash
python -m src.train \
  --dataset-name imdb \
  --model-name gpt2 \
  --epochs 3 \
  --out-dir results

# Evaluate perplexity on IMDB test using a Hub model
python -m src.evaluate \
  --model adetuire1/gpt2-imdb-tuned \
  --dataset-name imdb --split test \
  --n-texts 2000 --batch-size 4 --max-len 512 \
  --cache-dir /content/cache \
  --out-json results/ppl.json
```
**Notes**
GPT-2 has no pad token; the evaluator sets `pad_token = eos_token` for safe batching.

Artifacts write to `results/` (see “Viewing results” below).


### Generate
```bash
python -m src.generate \
  --model ./results/checkpoint-final \
  --prompt "Once upon a time" \
  --max-len 100
```
### Generate

```bash
# The script expects --model-dir (or --model)
python -m src.generate \
  --model-dir ./results/checkpoint-final \
  --prompt "Once upon a time" \
  --max-len 100
```

If you pass a Hugging Face Hub model id, use it as the value of `--model-dir`, e.g.
`--model-dir adetuire1/gpt2-imdb-tuned`.

### Viewing results
Perplexity metrics are written to `results/ppl.json`. View them with:

Linux/macOS: `cat results/ppl.json`

Windows (PowerShell): `Get-Content results/ppl.json`