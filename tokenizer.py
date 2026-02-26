import torch

class Tokenizer:

    def __init__(self):

        self.eos_token = '<EOS>'
        self.pad_token  = '<PAD>'
        self.unk_token = '<UNK>'

        self.chars = [self.eos_token, self.pad_token, self.unk_token] + \
            list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:()[]{}-_=+@#$%^&*<>/\\|`~"\' ')
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}

        self.eos_token_id = self.char_to_idx[self.eos_token]
        self.pad_token_id = self.char_to_idx[self.pad_token]
        self.unk_token_id = self.char_to_idx[self.unk_token]

        self.vocab_size = len(self.chars)

    def encode(self, text, return_tensor=True):
        tokens =  [self.char_to_idx.get(char, self.unk_token_id) for char in text] + [self.eos_token_id]   
         
        if return_tensor:
            tokens =  torch.tensor(tokens, dtype=torch.long)
        return tokens

    def decode(self, token_ids, include_special_tokens=False):

        chars = []
        for token_id in token_ids:
            char = self.idx_to_char.get(token_id, self.unk_token)
            if include_special_tokens or char not in [self.eos_token, self.pad_token, self.unk_token]:
                chars.append(char)

        return ''.join(chars)