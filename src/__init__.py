python -m src.evaluate perplexity \
  --model /content/gpt2_export \
  --n-texts 2000 \
  --batch-size 4 \
  --cache-dir /content/cache \
  --out-json results/ppl.json
