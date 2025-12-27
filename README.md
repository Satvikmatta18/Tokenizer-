
---

# Tokenizers from Scratch

This repository implements **four tokenizers**, showing how text is converted into integers and reconstructed back into text.

Each tokenizer follows the same high-level pattern:

```
text → tokens → token IDs → token IDs → tokens → text
```

What changes is **what counts as a token**.

---

## 1. Word-Level Tokenizer (`Word_tokenizer.py`)

### Example: Input → Output

**Training corpus**

```
["Hello, world!"]
```

**Vocabulary (simplified)**

```
<pad> → 0
<unk> → 1
<bos> → 2
<eos> → 3
hello → 4
, → 5
world → 6
! → 7
```

**Encoding**

```
Input text : "Hello, world!"
Tokens     : ["hello", ",", "world", "!"]
IDs        : [2, 4, 5, 6, 7, 3]
```

**Decoding**

```
Input IDs  : [2, 4, 5, 6, 7, 3]
Output text: "hello, world!"
```

---

### How it works (simple explanation)

* The tokenizer **splits text into words and punctuation**
* Every unique word becomes its own token
* Each token is assigned a fixed ID
* During encoding:

  * Text is split into tokens
  * Tokens are replaced with their IDs
  * `<bos>` and `<eos>` are added
* During decoding:

  * IDs are converted back to tokens
  * Special tokens are removed
  * Tokens are joined back into text

**Key limitation**:
If the model sees a word it never saw during training, it becomes `<unk>`.

---

## 2. Character-Level Tokenizer (`Character_tokenizer.py`)

### Example: Input → Output

**Training corpus**

```
["Hello, world!"]
```

**Vocabulary (partial)**

```
<pad> → 0
<unk> → 1
<bos> → 2
<eos> → 3
h → 4
e → 5
l → 6
o → 7
, → 8
(space) → 9
w → 10
r → 11
d → 12
! → 13
```

**Encoding**

```
Input text : "Hello, world!"
Characters : ["h","e","l","l","o",","," ","w","o","r","l","d","!"]
IDs        : [2, 4, 5, 6, 6, 7, 8, 9, 10, 7, 11, 6, 12, 13, 3]
```

**Decoding**

```
Input IDs  : [2, 4, 5, 6, 6, 7, 8, 9, 10, 7, 11, 6, 12, 13, 3]
Output text: "hello, world!"
```

---

### How it works (simple explanation)

* Every **single character** is treated as a token
* The vocabulary is just:

  * letters
  * numbers
  * punctuation
  * spaces
* Encoding converts each character to an ID
* Decoding converts IDs back to characters and joins them

**Big advantage**:
Almost never needs `<unk>`

**Big downside**:
Sequences get very long (bad for transformers)

---

## 3. Subword BPE Tokenizer (`tokenizer_BPE.py`)

### Example: Input → Output

**Training corpus**

```
["low", "lower", "lowest"]
```

**Learned subword tokens (simplified)**

```
low</w>
er</w>
est</w>
```

**Encoding**

```
Input text : "low lower lowest"
Subwords   : ["low</w>", "low", "er</w>", "low", "est</w>"]
IDs        : [2, 5, 7, 6, 8, 3]
```

**Decoding**

```
Input IDs  : [2, 5, 7, 6, 8, 3]
Output text: "low lower lowest"
```

---

### How it works (simple explanation)

* Starts by splitting words into **characters**
* Learns which character pairs appear together most often
* Repeatedly **merges frequent pairs into new tokens**
* Common pieces like `low`, `er`, `est` become single tokens
* Rare words are broken into smaller known pieces

**Why this matters**:

* Vocabulary stays small
* Model can handle unseen words
* This is the foundation of most classic NLP tokenizers

---

## 4. Byte-Level BPE Tokenizer (`Byte_level_BPE.py`)

### Example: Input → Output

**Input text**

```
"hello, BPE on bytes!"
```

**UTF-8 bytes (conceptual)**

```
[104, 101, 108, 108, 111, 44, 32, 66, 80, 69, ...]
```

**Encoding**

```
Input text : "hello, BPE on bytes!"
IDs        : [2, 11, 5, 15, 4, 14, 13, 6, 17, 7, 3]
```

**Decoding**

```
Input IDs  : [2, 11, 5, 15, 4, 14, 13, 6, 17, 7, 3]
Output text: "hello, bpe on bytes!"
```

---

### How it works (simple explanation)

* Text is first converted into **raw UTF-8 bytes**
* Each byte (0–255) becomes a base token
* BPE merges learn common **byte sequences**
* This guarantees:

  * no `<unk>` tokens
  * support for any Unicode text

This is the **core idea behind GPT-style tokenizers**.

---

## Summary 

* **Word-level**: “Each word gets a number”
* **Character-level**: “Each character gets a number”
* **Subword BPE**: “Learn common word pieces automatically”
* **Byte-level BPE**: “Learn common byte patterns — never break”
