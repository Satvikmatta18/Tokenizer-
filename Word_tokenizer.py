import re
import unicodedata

"""
A word tokenizer converts text into a sequence of tokens and back into text.
First it normalizes text by removing accents and converting to lowercase.
Then it splits text into words and punctuation in the tokenize function.
The build_vocab method creates a mapping of tokens to IDs.
The encode method converts text to IDs using that mapping;
    here we add <bos> and <eos> tokens to the beginning and end of the text,
    and for unknown tokens we use <unk>.
The decode method converts IDs back to text.
"""


class WordTokenizer:
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
        r"""
        Split text into word tokens and punctuation tokens.
          - \w+     : one or more word characters (a "word")
          - [^\w\s] : a single character that is not a word character
                     and not whitespace (punctuation)
        """
        text = self.normalize(text)
        tokens = re.findall(r"\w+|[^\w\s]", text)
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
          - join tokens, attaching punctuation to previous word
        """
        if not self.id_to_token:
            raise ValueError("id_to_token is empty. Call build_vocab() first.")

        tokens = []
        for i in ids:
            token = self.id_to_token.get(i, "<unk>")
            tokens.append(token)

        # remove special tokens
        tokens = [t for t in tokens if t not in self.SPECIAL_TOKENS]

        text_parts = []
        for tok in tokens:
            if not text_parts:
                text_parts.append(tok)
                continue

            # if it's punctuation, add it to the end of the last word
            if re.match(r"[^\w\s]", tok):
                text_parts[-1] = text_parts[-1] + tok
            else:
                text_parts.append(tok)

        text = " ".join(text_parts)
        return text


# Example
if __name__ == "__main__":
    tokenizer = WordTokenizer()

    corpus = ["Hello, world!"]
    tokenizer.build_vocab(corpus)

    text = "Hello, world!"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print("Original:", text)
    print("IDs:", ids)
    print("Decoded:", decoded)