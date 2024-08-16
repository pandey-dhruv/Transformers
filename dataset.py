import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        #now it is possible that the vocabulary size is larger than 32 bits. In such a case, the id that is assigned to these tokens
        #could be greater than what a 32 bit number can hold. Thus we use a 64 bit integer to hold the values
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype = torch.int64)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]
        #now we have the source(english) sentence and we have the target(french) sentence. Now we must tokenize them (assign id)
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        #also find out the number of padding tokens that must be added to the sentence to each the pre-determined context length
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  #because we add the SOS and EOS tokens also
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  #because we only add the SOS/EOS token here
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("The sentence is too long to fit the context")
        
        #Add SOS and EOS token to the source text
        #the source text would be a one dimensional array of indices (Depicting the position of the token int he vocabulary)
        #then we simply add the SOS token in the start, EOS token in the end and Padding tokens if required
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        #in the decoder, we only add the start of sentence token
        #in the label, we only add the end of sentence token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        #for double measure.
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        #when we append the pad tokens to the sentence, we do not want them to participate in the attention calculation as they do not
        #actually exist in the sentence. Thus we make those values negative infinity before the softmax so that the contribution is zero
        #for encoder mask, we unsqueeze twice to add the batch and the 
        """for the encoder msak - 
        encoder_input != self.pad_token would return an array (or tensor) containing True at the indices where the condition is true
        Now, unsqueeze two time will add two dimensions in the first place. Thus, shape becomes (1, 1, seq_len)
        Now this ust be done seq_len times (to get a seq_len * seq_len matrix that can be added to the original attention calculation)
        And this must be done for all the sentences in the batch (which is nothing but the batch_dimension)
        Thus the final dimension would simply be equal to (batch_Size, seq_len, seq_len)
        """
        return {
            "encoder_input" : encoder_input,
            "decoder_input" : decoder_input,
            "encoder_mask"  : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask"  : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label" : label,
            "src_text" : src_text,
            "tgt_text" : tgt_text
        }
        
def causal_mask(size):
    #here, we want to return an upper triangular matrix that contains negative infinity and the values below the diagonal are 0
    #since .triu returns an upper triangular matrix, we do mask==0 to ensure that the lower triangle becomes True and the upper = False
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
