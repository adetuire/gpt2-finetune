# src/generate.py
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
import typer, torch
app = typer.Typer()

@app.command()
def main(
    model: str = typer.Option("./results/checkpoint-final", "--model", "--model-dir", help="HF Hub id or local path"),
    prompt: str = "Once upon a time",
    max_len: int = 100
):
    tok = GPT2Tokenizer.from_pretrained(model)
    mod = GPT2LMHeadModel.from_pretrained(model, torch_dtype=torch.float16 if torch.cuda.is_available() else None)
    gen = pipeline("text-generation", model=mod, tokenizer=tok, device=0 if torch.cuda.is_available() else -1)
    print(gen(prompt, max_length=max_len)[0]["generated_text"])


if __name__ == "__main__":
    app()
