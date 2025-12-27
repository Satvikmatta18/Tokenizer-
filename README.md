# Tokenizers from Scratch 

This repo contains **four tokenizers** implemented from scratch in Python, showing how LLMs turn **text into integers** and back.

## Included Tokenizers

- `Word_tokenizer.py` – word-level tokenizer  
- `Character_tokenizer.py` – character-level tokenizer  
- `tokenizer_BPE.py` – subword BPE over characters + `</w>`  
- `Byte_level_BPE.py` – byte-level BPE (GPT-style idea)

Each tokenizer includes:

- A **training phase** (build vocabulary / learn BPE merges)
- An **encode** method (`text → [ids]`)
- A **decode** method (`[ids] → text`)

---

## 1. Word-Level Tokenizer (`Word_tokenizer.py`)

### What it does

- Treats tokens as **words + punctuation**
- Demonstrates the classic NLP pipeline:
  - normalize → tokenize → build vocab → encode → decode

### Main Methods

- `normalize(text)`  
  Unicode NFKC normalization, lowercase, strip, collapse spaces

- `tokenize(text)`  
  Regex split into words and punctuation

- `build_vocab(corpus)`  
  Builds:
  - `token_to_id`
  - `id_to_token`  
  Includes `<pad>`, `<unk>`, `<bos>`, `<eos>`

- `encode(text)`  
  Converts text → `[ids]` with `<bos>` and `<eos>`

- `decode(ids)`  
  Converts `[ids]` → text  
  Removes special tokens and fixes spacing/punctuation

### Example

```bash
python Word_tokenizer.py
