import re
import unicodedata

"""
A Character converts text into a sequence of characters and back 
Works the same way as a word tokenizer but with characters of instead of words 
"""

class Character_Tokenizer:
    # Class-level constant: shared by all uses of this tokenizer
    SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
    # <pad> -- padding token, <unk> -- unknown token,
    # <bos> -- beginning-of-sentence token, <eos> -- end-of-sentence token

    def __init__(self) -> None:
        # These will hold the vocabulary once build_vocab is called
        self.token_to_id = {}
        self.id_to_token = {}

    def normalize(self, text):
        """
        Normalize text:
          - Unicode normalize (NFKC)
          - lowercase
          - strip leading/trailing whitespace
          - collapse multiple spaces to one
        """
        text = unicodedata.normalize("NFKC", text)
        text = text.lower()
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def tokenize(self, text):
        """
        Split text into individual characters.
        """
        text = self.normalize(text)
        tokens = list(text)
        return tokens

    def build_vocab(self, corpus):
        """
        Creates a vocabulary based on the input corpus.
        Steps:
          - start with SPECIAL_TOKENS
          - for each text in corpus: normalize + tokenize
          - if a token is new, add it to the vocabulary
        Fills:
          self.token_to_id and self.id_to_token
        """
        vocab_tokens = list(self.SPECIAL_TOKENS)
        seen = set(vocab_tokens)

        for text in corpus:
            norm = self.normalize(text)
            tokens = self.tokenize(norm)

            for token in tokens:
                if token not in seen:
                    seen.add(token)
                    vocab_tokens.append(token)

        self.token_to_id = {}
        self.id_to_token = {}

        for idx, token in enumerate(vocab_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        return self.token_to_id

    def encode(self, text):
        """
        Given text, convert it to a list of IDs using self.token_to_id.
        Steps:
          - normalize
          - tokenize
          - add <bos> and <eos>
          - map tokens to ids, using <unk> when not found
        """
        if not self.token_to_id:
            raise ValueError("Vocabulary is empty. Call build_vocab() first.")

        text = self.normalize(text)
        tokens = self.tokenize(text)
        tokens = ["<bos>"] + tokens + ["<eos>"]

        unk_id = self.token_to_id["<unk>"]
        ids = []
        for tok in tokens:
            idx = self.token_to_id.get(tok, unk_id)
            ids.append(idx)
        return ids

    def decode(self, ids):
        """
        Convert IDs back to text using self.id_to_token.
        Steps:
          - map ids -> tokens (with "<unk>" fallback)
          - remove special tokens
          - join characters directly
        """
        if not self.id_to_token:
            raise ValueError("id_to_token is empty. Call build_vocab() first.")

        tokens = []
        for i in ids:
            token = self.id_to_token.get(i, "<unk>")
            tokens.append(token)

        # remove special tokens
        tokens = [t for t in tokens if t not in self.SPECIAL_TOKENS]

        text = "".join(tokens)
        return text


# Example
if __name__ == "__main__":
    tokenizer = Character_Tokenizer()

    corpus = ["Hello, world!"]
    tokenizer.build_vocab(corpus)

    text = "Hello, world!"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print("Original:", text)
    print("IDs:", ids)
    print("Decoded:", decoded)