import re 
from sre_parse import SPECIAL_CHARS
import unicodedata
from collections import defaultdict 

"""
Byte-level BPE tokenizer:
    - Training: learn merges over UTF-8 bytes.
    - Encoding: apply merges to new text, map subword bytes to IDs.
    - Decoding: map IDs back to bytes, then decode UTF-8.
"""

class ByteBPE_Tokenizer: 
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
        text = text.lower() # optional
        text = text.strip()
        text = re.sub(r"\s+", " ", text) # collapse multiple spaces to one
        return text 

    def _get_bpe_vocab_from_corpus(self, corpus):
        """
        Build initial BPE vocab from corpus as:
          tuple(byte_tokens) -> frequency

        Here byte_tokens is a sequence of 1-byte tokens (each a bytes object of length 1).
        We treat each line as one sequence of bytes.
        
        """ 

        bpe_vocab = defaultdict(int)
        
        for line in corpus:
            norm = self.normalize(line)

            b = norm.encode("utf-8")
            
            symbols = [bytes([byte]) for byte in b]
            if not symbols: continue 
            bpe_vocab[tuple(symbols)] +=1 

        return dict(bpe_vocab) 

    @staticmethod 
    def _get_pair_stats(bpe_vocab): 
        """
        Count how often each adjacent paif of symbols appear in the current vocab 
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
        Merge a pair of byte-symbols (a, b) into a single symbol (a+b) in the BPE vocab.
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
        Train byte-level BPE on a text corpus.
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

    def _encode_bytes(self, byte_seq): 
        """
        Apply BPE merges to a sequence of byte-symbols (list of bytes objects).
        """

        symbols  = [bytes([b]) for b in byte_seq]

        if not self.pair2rank: 
            return symbols 
        
        while True: 
            candidates = []
            for i in range(len(symbols)-1): 
                pair = (symbols[i], symbols[i+1])
                if pair in self.pair2rank: 
                    rank = self.pair2rank[pair]
                    candidates.append((rank, i, pair))
            
            if not candidates: 
                break 

            _, _, best_pair = min(candidates) 

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
        Encode raw text into list of IDS using BPE merges and token_to_id
        """

        if not self.token_to_id: 
            raise ValueError("Vocabulary is empty. Call train(corpus) first.")

        norm = self.normalize(text)
        words = norm.encode("utf-8")
        byte_tokens = self._encode_bytes(words)
        tokens = ["<bos>"] + byte_tokens + ["<eos>"] 
        unk_id = self.token_to_id["<unk>"]
        ids = []
        for tok in tokens: 
            idx = self.token_to_id.get(tok, unk_id)
            ids.append(idx)
        return ids
        
    def decode(self, ids):
        """
        decode list of IDS back into text
        """ 

        if not self.id_to_token: 
            raise ValueError("Vocabulary is empty. Train the tokenizer first.")

        tokens = []
        for i in ids: 
            token = self.id_to_token.get(i, "<unk>")
            tokens.append(token)
        
        special = set(self.SPECIAL_TOKENS)
        byte_pieces= [t for t in tokens if t not in special]
        byte_stream = b"".join(byte_pieces)
        text = byte_stream.decode("utf-8", errors="replace")
        return text 
        
 
if __name__ == "__main__":
    corpus = [
        "hello, world!",
        "hello, byte BPE!",
        "BPE on bytes.",
    ]

    tokenizer = ByteBPE_Tokenizer(num_merges=50)
    tokenizer.train(corpus)

    print("Vocab size:", len(tokenizer.token_to_id))

    text = "hello, BPE on bytes!"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print("Text   :", text)
    print("IDs    :", ids)
    print("Decoded:", decoded)