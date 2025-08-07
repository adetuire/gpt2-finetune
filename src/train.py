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

app = typer.Typer(help="Fine-tune GPT-2 with HuggingFace")

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

@app.command()
def main(
    dataset_name: str = "imdb",
    model_name: str = "gpt2",
    out_dir: str = "./results",
    epochs: int = 3,
    batch: int = 2,
    max_len: int = 512,
    seed: int = 42,
):
    seed_everything(seed)

    ds = load_dataset(dataset_name)
    tok = GPT2Tokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token                               # GPT-2 has no PAD

    def tokenize(ex):
        return tok(ex["text"], truncation=True, padding="max_length", max_length=max_len)

    ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])

    model = GPT2LMHeadModel.from_pretrained(model_name)

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
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
