import os
import torch

from tokenizer import CharacterTokenizer
from model import GPT
from config import *

base_dir = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(
    base_dir,
    "data",
    "stories.txt"
)

checkpoint_path = os.path.join(
    base_dir,
    "checkpoints",
    "gpt_step_90000.pth"
)

print("Building vocabulary...")

all_chars = set()

with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        all_chars.update(line)

vocab_text = "".join(sorted(all_chars))

tokenizer = CharacterTokenizer(vocab_text)

print(f"Vocabulary Size: {tokenizer.vocab_size}")

model = GPT(
    vocab_size=tokenizer.vocab_size,
    embed_dim=EMBED_DIM,
    block_size=BLOCK_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location="cpu"
    )
)

model.eval()

print("Model Loaded Successfully")

while True:

    prompt = input("\nEnter Prompt (or quit): ")

    if prompt.lower() == "quit":
        break

    encoded = tokenizer.encode(prompt)

    x = torch.tensor(
        [encoded],
        dtype=torch.long
    )

    with torch.no_grad():

        output = model.generate(
            x,
            max_new_tokens=400,
            temperature=0.8,
            top_k=20
        )

    generated_text = tokenizer.decode(
        output[0].tolist()
    )

    print("\n")
    print("=" * 80)
    print(generated_text)
    print("=" * 80)