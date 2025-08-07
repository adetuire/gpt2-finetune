# GPT-2 Fine-Tuning Starter

Fine-tune GPT-2 on any text corpus using Transformers & Datasets.

## Quickstart
```bash
conda env create -f environment.yml   # or pip install -r requirements.txt
python -m src.train --dataset_name imdb --epochs 3
python -m src.generate --prompt "Once upon a time"
