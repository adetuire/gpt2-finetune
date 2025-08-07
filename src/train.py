#!/usr/bin/env python
import typer, torch, random, numpy as np
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
app = typer.Typer(help="Fine-tune GPT-2 with HuggingFace")

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

@app.command()
def main(
    dataset_name: str = "imdb",
    model_name: str = "gpt2",
    out_dir: str = "./results",
    epochs: int = 3,
    batch: int = 1,
    max_len: int = 512,
    seed: int = 42,
):
    seed_everything(seed)

    ds = load_dataset(dataset_name)
    tok = GPT2Tokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token                               

    def tokenize(ex):
        return tok(ex["text"], truncation=True, padding="max_length", max_length=max_len)

    ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])

    model = GPT2LMHeadModel.from_pretrained(model_name)

    args = TrainingArguments(
        output_dir = out_dir,
        num_train_epochs = epochs,
        gradient_accumulation_steps = 8,
        per_device_train_batch_size = batch,

        per_device_eval_batch_size=2,
        warmup_steps=500,

        save_total_limit = 3,
        learning_rate    = 2e-5,
        weight_decay     = 0.01,
        fp16             = torch.cuda.is_available(),
        save_steps = 500,
        logging_steps = 50,
        save_total_limit=3,
        seed             = seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["test"],
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    trainer.train()
    trainer.save_model(f"{out_dir}/checkpoint-final")
    tok.save_pretrained(f"{out_dir}/checkpoint-final")

if __name__ == "__main__":
    app()
