from tokenizer import CharacterTokenizer

with open("data/stories.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharacterTokenizer(text)

print("Vocabulary Size:", tokenizer.vocab_size)

with open("data/vocab.txt", "w", encoding="utf-8") as f:
    for ch in tokenizer.stoi:
        f.write(ch + "\n")

print("Vocabulary saved")