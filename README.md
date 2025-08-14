# GPT-2 Fine-Tuning Starter

Fine-tune GPT-2 on any text corpus using Transformers & Datasets.

## Quickstart
```bash
# train
python -m src.train --dataset-name imdb --model-name gpt2 --epochs 1 --batch 4 --max-len 256 --out-dir results

# export
python - <<'PY'
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from pathlib import Path
CKPT, EXP = Path("results/checkpoint-final"), Path("gpt2_export"); EXP.mkdir(exist_ok=True)
GPT2LMHeadModel.from_pretrained(CKPT).save_pretrained(EXP)
GPT2TokenizerFast.from_pretrained("gpt2").save_pretrained(EXP)
print("exported to", EXP)
PY

# evaluate
python -m src.evaluate perplexity --model gpt2_export --n-texts 2000 --out-json results/ppl.json

# generate
python -m src.generate --model gpt2_export --prompt "Once upon a time"
```