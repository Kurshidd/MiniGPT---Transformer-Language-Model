class CharacterTokenizer:

    def __init__(self, text):

        chars = sorted(list(set(text)))

        # Add unknown token
        chars = ["<UNK>"] + chars

        self.stoi = {
            ch: i for i, ch in enumerate(chars)
        }

        self.itos = {
            i: ch for i, ch in enumerate(chars)
        }

        self.vocab_size = len(chars)

    def encode(self, text):

        unk = self.stoi["<UNK>"]

        return [
            self.stoi.get(c, unk)
            for c in text
        ]

    def decode(self, tokens):

        return ''.join(
            self.itos.get(i, "")
            for i in tokens
        )