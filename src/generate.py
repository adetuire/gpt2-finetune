from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
import typer, torch
app = typer.Typer()

@app.command()
def main(model_dir="./results/checkpoint-final", prompt="Once upon a time", max_len=100):
    tok  = GPT2Tokenizer.from_pretrained(model_dir)
    mod  = GPT2LMHeadModel.from_pretrained(model_dir, torch_dtype=torch.float16 if torch.cuda.is_available() else None)
    gen  = pipeline("text-generation", model=mod, tokenizer=tok, device=0 if torch.cuda.is_available() else -1)
    print(gen(prompt, max_length=max_len)[0]["generated_text"])

if __name__ == "__main__":
    app()
