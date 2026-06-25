import os
import json
import numpy as np
from tokenizer import CharacterTokenizer

# 1. Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(base_dir, "data", "TinyStories_all_data")
output_file = os.path.join(base_dir, "data", "train_tokens.bin")

# 2. First, build a complete tokenizer by reading a sample to get all characters
print("Building tokenizer...")
sample_path = os.path.join(input_folder, "data00.json")
with open(sample_path, "r", encoding="utf-8") as f:
    sample_data = json.load(f)
# Combine a mix of stories to capture all possible characters
sample_text = "".join([story["story"] for story in sample_data[:1000]])
tokenizer = CharacterTokenizer(sample_text)
print(f"Vocabulary size: {tokenizer.vocab_size}")

# 3. Process all JSON files and stream tokens into a binary file
print("Tokenizing full dataset to binary... (This may take a few minutes)")
all_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.json')])

with open(output_file, "wb") as bin_file:
    for filename in all_files:
        file_path = os.path.join(input_folder, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract stories, tokenize them, and write directly to disk
        for item in data:
            story_text = item.get("story", "")
            if story_text:
                tokens = tokenizer.encode(story_text)
                # Convert to uint16 (handles vocab sizes up to 65,535, saves 50% disk space)
                token_array = np.array(tokens, dtype=np.uint16)
                bin_file.write(token_array.tobytes())
                
        print(f"Processed {filename}")

print(f"\nSuccess! Full data tokenized and saved to: {output_file}")