import re 
from sre_parse import SPECIAL_CHARS
import unicodedata
from collections import defaultdict 


"""
BPE_Tokenizer: convert text <-> subword tokens using Byte Pair Encoding.

Why BPE:
  - Word-level: few tokens but huge vocab and many <unk>.
  - Char-level: tiny vocab, no <unk>, but very long sequences.
  - BPE: learns subwords from data → medium vocab, medium sequence length, almost no <unk>.

Training (train):
  - Normalize corpus, split into words.
  - Represent each word as characters + '</w>'.
  - Repeatedly:
      * count adjacent symbol pairs in all words,
      * pick the most frequent pair,
      * merge it into a new symbol everywhere,
      * record the merge order.
  - Build:
      * pair2rank (pair -> merge order),
      * token_to_id / id_to_token (final subword vocab + special tokens).

Encoding (encode):
  - Normalize text, split into words.
  - For each word:
      * start from chars + '</w>',
      * repeatedly apply merges using pair2rank (earliest merges first),
      * get BPE subword tokens.
  - Add '<bos>' and '<eos>', map tokens -> ids.

Decoding (decode):
  - Map ids -> tokens, drop '<pad>', '<unk>', '<bos>', '<eos>'.
  - Use '</w>' suffix to know where each word ends.
  - Join reconstructed words with spaces.
"""

class BPE_Tokenizer: 
    SPECIAL_TOKENS = ["<pad>" , "<unk>", "<bos>", "<eos>"]

    def __init__(self, num_merges): 
        self.num_merges = num_merges # how many bpe merge operations during training, larger -- bigger subword vocab, fewer tokens per word
        self.token_to_id ={} # tokens to ids 
        self.id_to_token= {} # mapping from ids to tokens 
        self._bpe_vocab = {} # bpe training vocab: mapping from tuple(symbools) --> frequency
        self.merges = [] # list of merges during training in order, each merge pair (a, b)
        self.pair2rank = {} # rank of the pair in the merges list 
    

    def normalize(self, text): 
        text = unicodedata.normalize("NFKC", text)
        text = text.lower()
        text = text.strip()
        text = re.sub(r"\s+", " ", text) # collapse multiple spaces to one
        return text 

    def _get_bpe_vocab_from_corpus(self, corpus):
        f"""
        Initial BPE vocab from corpus 
        normalize each line, split words by whitespace 
        represent each word as a list(characters) + ["</w>'] (marks end of word)
        Count frequency of each word form appears 
        eg. output: 
      ('l','o','w','</w>'): 2,
      ('l','o','w','e','r','</w>'): 1,
      ...
        
        """
        bpe_vocab = defaultdict(int)
        
        for line in corpus:
            norm = self.normalize(line)
            
            for word in norm.split(): 
                if not word: continue
                # characters  + end of word marker
                symbols = list(word)+["</w>"]
                bpe_vocab[tuple(symbols)]+=1 

        return dict(bpe_vocab) 

    @staticmethod 
    def _get_pair_stats(bpe_vocab): 
        """
        Count how often each adjacent paif of symbols appear in the current vocab 
        return pair_counts: dict mapping (sym1, sym2)--> freq 
        bpe_vocab = {
                ('l', 'o', 'w', '</w>'): 2,
                ('l', 'o', 'w', 'e', 'r', '</w>'): 1
            }
        Then the adjacent pairs are:
                ('l','o'), ('o','w'), ('w','</w>')          in the first word
                ('l','o'), ('o','w'), ('w','e'),
                ('e','r'), ('r','</w>')                      in the second word
        return pair_counts: dict mapping (sym1, sym2)--> freq 
        """
        pair_counts = defaultdict(int)

        for word, freq in bpe_vocab.items():
            if len(word)<2: continue 

            for i in range(len(word)-1): 
                pair = (word[i], word[i+1])
                pair_counts[pair]+=freq  
        return dict(pair_counts)

    @staticmethod
    def _merge_vocab(pair, bpe_vocab): 
        """
        Merge a pair of symbols (a, b) into a single symbol in the BPE vocab        
        Input:
            pair: (sym1, sym2) to merge, e.g. ('l','o')
            bpe_vocab: dict mapping tuple(symbols) -> frequency
        Example:
            pair = ('l','o')
            bpe_vocab = {
                ('l', 'o', 'w', '</w>'): 2,
                ('l', 'o', 'w', 'e', 'r', '</w>'): 1
            }
            After merging ('l','o') -> 'lo', every ('l','o') in the keys
            becomes 'lo':
                ('l','o','w','</w>')          -> ('lo','w','</w>')
                ('l','o','w','e','r','</w>')  -> ('lo','w','e','r','</w>')
            So the new_bpe_vocab is:
                {
                    ('lo', 'w', '</w>'): 2,
                    ('lo', 'w', 'e', 'r', '</w>'): 1
                }
        Returns:
            new_bpe_vocab: updated mapping after merging the pair everywhere
        """
        new_bpe_vocab =defaultdict(int)
        first, second = pair 
        merged_symbol = first+second 

        for word, freq in bpe_vocab.items():
            new_word = []
            i = 0 
            while i <len(word): 
                if i<len(word)-1 and word[i] == first and word[i+1] == second: 
                    new_word.append(merged_symbol)
                    i+=2
                else: 
                    new_word.append(word[i])
                    i+=1 
            new_bpe_vocab[tuple(new_word)]+=freq

        return new_bpe_vocab 
    
    def train(self, corpus): 
        """ 
        Train BPE tokenizer from corpus 
        1. build intial char + </w> vocab from corpus
        2. Repeat up to num_merges times 
            Count all adjacent pairs 
            pick most freq pairs 
            merge the pair in vocab 
            record merge 
        3.  build: pair2rank( pair --> rank merge order), token_to_id/id_to token maps
        """ 
        self._bpe_vocab = self._get_bpe_vocab_from_corpus(corpus)
        # Merge frequent pairs 
        for _ in range(self.num_merges): 
            pair_counts = self._get_pair_stats(self._bpe_vocab)
            if not pair_counts: break 

            best_pair= max(pair_counts, key=pair_counts.get)
            
            if pair_counts[best_pair]<2: break 

            self.merges.append(best_pair)
            self._bpe_vocab = self._merge_vocab(best_pair, self._bpe_vocab)

        self.pair2rank = {pair: rank for rank , pair in enumerate(self.merges) }

        symbol_set = set()
        for word in self._bpe_vocab.keys(): 
            for sym in word: 
                symbol_set.add(sym) 
        
        vocab_list = list(self.SPECIAL_TOKENS) + sorted(symbol_set)

        self.token_to_id = {}
        self.id_to_token = {}
        for idx, token in enumerate(vocab_list): 
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token 
        
        return self.token_to_id

    def _encode_word(self, word): 
        """
        Apply BPE merges to a single word 
        1. Represent word as characters + </w> 
        2. Repeat merge pairs according to pair2rank 
        3. stop when no more mergable paids remain 
        4. return final list of bpe symbols of the word
        """
        symbols  = list(word)+["</w>"]

        if not self.pair2rank: 
            return symbols 
        
        while True: 
            candidate = []
            for i in range(len(symbols)-1): 
                pair = (symbols[i], symbols[i+1])
                if pair in self.pair2rank: 
                    rank = self.pair2rank[pair]
                    candidate.append((rank, i, pair))
            
            if not candidate: 
                break 

            _, _, best_pair = min(candidate) 

            new_symbols =[]
            i = 0 
            while i <len(symbols): 
                if(i<len(symbols)-1 and symbols[i] == best_pair[0] and symbols[i+1] == best_pair[1]): 
                    new_symbols.append(symbols[i]+symbols[i+1]) 
                    i+=2
                else: 
                    new_symbols.append(symbols[i])
                    i+=1 
            symbols = new_symbols 

        return symbols 
    
    def encode(self, text): 
        """
        Encode raw tst into list of IDS using BPE merges and token_to_id
        1. normalize text 
        2. split text words by whitespace
        3.apply bpe merges of each word 
        4.add <bos> and <eos> tokens 
        5.Map each token to id and if missing use <unk>
        """

        if not self.token_to_id: 
            raise ValueError("Vocabulary is empty. Call train(corpus) first.")

        norm = self.normalize(text)
        words = [w for w in norm.split() if w]
        tokens = []
        for word in words: 
            word_bpe = self._encode_word(word) 
            tokens.extend(word_bpe)
        tokens = ["<bos>"] + tokens + ["<eos>"] 
        unk_id = self.token_to_id["<unk>"]
        ids = []
        for tok in tokens: 
            idx = self.token_to_id.get(tok, unk_id)
            ids.append(idx)
        return ids
        
    def decode(self, ids):
        """
        decode list of IDS back into text
        1. map ids --> tokens 
        2. remove special tokens 
        3. use </w> to reconstruct words
        3. Join words with spaces 
        """ 

        if not self.id_to_token: 
            raise ValueError("Vocabulary is empty. Train the tokenizer first.")

        tokens = []
        for i in ids: 
            token = self.id_to_token.get(i, "<unk>")
            tokens.append(token)
        
        special = set(self.SPECIAL_TOKENS)
        tokens= [t for t in tokens if t not in special]

        words= []
        current_word =""
        for tok in tokens: 
            if tok.endswith("</w>"): 
                peice = tok[:-4]
                current_word+=peice 
                words.append(current_word) 
                current_word =""
            else: 
                current_word+=tok
        
        if current_word: 
            words.append(current_word) 

        text = " ".join(words)
        return text 
        
        

if __name__ == "__main__":
    corpus = [
        "low",
        "lower",
        "lowest",
        "low low lower",
    ]
    tokenizer = BPE_Tokenizer(num_merges=20)

    # Show initial char+`</w>` vocab
    initial_vocab = tokenizer._get_bpe_vocab_from_corpus(corpus)
    print("Initial BPE vocab (word as symbols -> freq):")
    for word, freq in initial_vocab.items():
        print(" ", word, ":", freq)
    print()

    # Train BPE
    tokenizer.train(corpus)

    # Show learned merges
    print("Learned merges (in order):")
    for i, pair in enumerate(tokenizer.merges):
        print(f"  {i}: {pair}")
    print()

    # Show some of the final vocab
    print("Final token_to_id (first 50 entries):")
    for tok, idx in list(tokenizer.token_to_id.items())[:50]:
        print(f"  {idx}: {tok}")
    print()

    # Test encode/decode on a sentence
    text = "low lower lowest"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print("Text   :", text)
    print("IDs    :", ids)
    print("Decoded:", decoded)