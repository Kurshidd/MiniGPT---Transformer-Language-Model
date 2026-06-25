import os
import torch
from torch.utils.data import IterableDataset, DataLoader

from tokenizer import CharacterTokenizer
from model import GPT
from config import *

# =====================================================
# Streaming Dataset
# =====================================================

class StreamingTextDataset(IterableDataset):

    def __init__(self, file_path, tokenizer, block_size):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):

        buffer = []

        with open(self.file_path, "r", encoding="utf-8") as f:

            for line in f:

                if not line.strip():
                    continue

                tokens = self.tokenizer.encode(line)

                buffer.extend(tokens)

                while len(buffer) > self.block_size:

                    chunk = buffer[: self.block_size + 1]

                    x = torch.tensor(
                        chunk[: self.block_size],
                        dtype=torch.long
                    )

                    y = torch.tensor(
                        chunk[1 : self.block_size + 1],
                        dtype=torch.long
                    )

                    yield x, y

                    buffer = buffer[self.block_size :]


# =====================================================
# Paths
# =====================================================

base_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(
    base_dir,
    "data",
    "stories.txt"
)

# =====================================================
# Build Vocabulary Safely
# =====================================================

print("Scanning dataset vocabulary...")

all_chars = set()

with open(file_path, "r", encoding="utf-8") as f:

    for line in f:
        all_chars.update(line)

vocab_text = "".join(sorted(all_chars))

tokenizer = CharacterTokenizer(vocab_text)

print(f"Vocabulary Size: {tokenizer.vocab_size}")

# =====================================================
# Dataset & Loader
# =====================================================

dataset = StreamingTextDataset(
    file_path=file_path,
    tokenizer=tokenizer,
    block_size=BLOCK_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE
)

# =====================================================
# Model
# =====================================================

model = GPT(
    vocab_size=tokenizer.vocab_size,
    embed_dim=EMBED_DIM,
    block_size=BLOCK_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

print(f"Using device: {device}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# =====================================================
# Checkpoints Folder
# =====================================================

checkpoint_dir = os.path.join(
    base_dir,
    "checkpoints"
)

os.makedirs(
    checkpoint_dir,
    exist_ok=True
)

# =====================================================
# Training
# =====================================================

model.train()

for epoch in range(EPOCHS):

    print(f"\n===== Epoch {epoch + 1} =====")

    for step, (x, y) in enumerate(loader):

        x = x.to(device)
        y = y.to(device)

        logits, loss = model(x, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if step % 100 == 0:

            print(
                f"Epoch {epoch + 1} | "
                f"Step {step} | "
                f"Loss {loss.item():.4f}"
            )

        if step % 5000 == 0 and step > 0:

            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"gpt_step_{step}.pth"
            )

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print(
                f"Checkpoint saved: "
                f"{checkpoint_path}"
            )

print("\nTraining Complete!")

# =====================================================
# Final Save
# =====================================================

final_model_path = os.path.join(
    checkpoint_dir,
    "gpt.pth"
)

torch.save(
    model.state_dict(),
    final_model_path
)

print(
    f"Final Model Saved: "
    f"{final_model_path}"
)