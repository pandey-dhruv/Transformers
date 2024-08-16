import pandas as pd
import numpy as np
import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import math
import torch 
import torch.nn as nn

class InputEmbedding(nn.Module):
    #d_model is the dimension of the inout embeddings (the size of the numerical encoding of the vectors)
    #vocab_size is the total number of words in the vocabulary
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)  

    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """we need to give the size of the embedding vectors to the positional embedding layer because the size of the positional embedding layer must be equal to the 
    size of the input embeddings as the two must be added together to get the final input + positionl embedding. I must remind you that this still does not have
    any reference to the context and is still, just based upon the word and would be the same each time that words appears (if we do not consider positonal embedding)
    Also, we must pass in the sequence length here because the positional encoding is only calculated for the entire sequence that is passed into the model at a time
    This is because the model can only process a pre-determined context length at max and this is the sequence length. The positonal encoding is only calculated
    for words that are within the sequence length as only they are passed to the model at a single timestep"""
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        #this essentially creates a Dropout layer that prevents overfitting to the input data
        #now we create a matrix that will store the positional encoding (only calculated once, not learned through backprop, used again when needed)
        pe = torch.zeros(seq_len, d_model)          #creates a matrix of size, say (6,512), if sequence length = 6 and embedding dimension = 512
        #create a vector of shape (sequence length,1)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000)/d_model))
        #applying this to the even positions (sin term) and the odd positions (cos terms)
        #[:, 0::2] means apply this to all the rows, and for the colums, start from 0, and move by 2 steps upto the end
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #adding the batch dimension in the first position (1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        #when the model state is saved, the positional encodings will also be saved along with it
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        '''requires grad (false) is used when we do not want this to get update by backpropagation
        x.shape[1] would basically give you the number of words that have been fed in as the context
        the first : is for the batch dimension
        the second means that it must be added for all the rows that are present in the x (input embedding matrix)
        the last means that we must add all the corresponding column values together (thus all the values)'''
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    """This is the layer normalization"""
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        #defining something as Parameter makes it learnable through backpropagation
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1))
        #this is only for numerical stability so that we do not end up dividing by zero somewhere
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha*(x - mean)/(std**2 + self.epsilon) + self.bias

class FeedForwardBlock(nn.Module):
    """since the feed forward is applied to each word in the input sequence parallel, you basically have a feed forward network with d_model number of imputs
    in the first layer (the input layer). These inputs are then densly connected to a layer containing d_ff number of neurons and thus, we use the first linear_1
    layer that takes d_model number of inputs and produces d_ff number of outputs"""
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        #thus the basica transformation can be summarized as - 
        #(batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
    
    def forward(self, x):
        #linear -> relu activation -> dropout regularization -> linear_2
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        #we must divide the embedding vector along the embedding dimension into h sections (for each head). Thus, if the embedding dimension is 512 and h = 4,
        #the original embedding gets divided into 4 fragments each of size 128. Thus d_model should be divisible by h
        assert d_model % self.h == 0, "The dimension of the embedding vector is not divisible by the number of heads"
        self.d_k = int(self.d_model//self.h)
        self.w_q = nn.Linear(d_model, d_model)  #this represents Wq in multi-head attention
        self.w_k = nn.Linear(d_model, d_model)  #this represents Wk in multi-head attention
        self.w_v = nn.Linear(d_model, d_model)  #this represents Wv in multi-head attention
        self.w_o = nn.Linear(d_model, d_model)  #this represents Wo in multi-head attention
    
    #this is a decorator so that we can simply call this function without instantiating an object of the MultiHeadAttention class
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (Batch_size, h, seq_len, d_k)*(Batch_size, h, d_k, seq_len) -> (Batch_size, h, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
        #now we must add the mask before applying the softmax so that we can cut-off the interactions that are unwanted (LOOK INTO THIS. NOT SO CLEAR)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)

        #applying the dropout layer if provided
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return torch.matmul(attention_scores, value), attention_scores
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        #here the transformation is (seq_len, d_model) -> (seq_len, d_model) because the layers have d_model number of inputs (one for each in the embedding) and the same number of outputs
        '''here the transformation is (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) (this is basically dividing the last dimension which is
        the total d_model sized embedding of the word into h fractions each of size d_k)
        now we take the transpose to finally get (batch_size, h, seq_len, d_k) because we want each head h to see (seq_len, d_k).
        thus, each head can see the entire sentence but only a fraction of the original embedding'''
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2)  #because we need to bring this back to the original dimension (Batch_size, seq_len, h, d_k)
        #now we concatenate the above mentioned matrices along h
        #now why -1 here, because we only want the last one to predict the next coming word in the sequence. This last encoding contains all the relevant information
        #from the previous sequence
        x = x.contiguous().view(x.shape[0], -1, x.shape[2]*x.shape[3])
        #convert this to (Batch_size, seq_len, d_model). This would then be passed to a linear layer of size = vocabulary size which then assigns a number to each
        #word in the vocabulary. This is then passed through a softmax which assigns a probaility to each word in the vocabulary as the next word in the sequence
        return self.w_o(x)

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    def forward(self, x, sublayer):
        #think of x as the input to the multihead attention. Thus MHA becomes the sublayer. We normalize, pass this through the sublayer and then, we finally
        #add it to the initial values. (This is simple a rediual connection)
        x = x + self.dropout(sublayer(self.norm(x)))
        return x
        
"""In python, containers are classes or data structures that are designed to hold and organize neural network components such as layers, modules, parameters and other sub networks.
You have the Sequential container, wherein the layers are added in sequentially as they are passed
ModuleList allows us to store a list of Modules. Basically, it is more flexible in use than the Sequential container
nn.ModuleList is just a python list, but one where the weights are registered in Pytorch. It itself does not define a forward function as it does not
define the neural network architecture by itself. So, it is just a container to store the modules that we might use later in our Neural Network"""
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)
        #this is because the entire Encoder block has 2 Residual connections. Here, we just store the Residual connection module in a list for future use
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        #when you initialize the ENcoder block, you will pass in the layers of the block itself, Thus we do not initialize the layers here. Simply use the forward fucntion
        """lambda x: ko remove kar ke dekhna"""
        assert x is not None
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        #this should receive x as the input in feed forward block
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

#the above was to define a single Encoder block, Now we define the entire encoder which can contain multiple of these encoder blocks
class Encoder(nn.Module):
    def __init__(self, layer: nn.ModuleList):
        super().__init__()
        self.layers = layer
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        assert x is not None
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    #the encoder_output us need for the step with the cross-multihead attention
    #src_mask is applied to the encoder, target mask is applied to the decoder
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

#A single decoder block can simply hold in multiple decoder units. Thus we build the entire decoder block. This is very similar to the encoder block with 
#very slight changes, namely that it also uses the output from the enoder block, and cross multihead attention.
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

#the ouput of the decoder block would simply be (Batch_size, seq_len, d_model).
#However, we want to project this to the vocabulary so that we can get a prediction probability for each word in the vocabulary
#thus, we build this projection layer that simply uses a Linear layer to map this to (Batch_size, seq_len, vocab_size)
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.linear_layer(x)
        x = torch.log_softmax(x, dim=-1)
        return x
        #this x would contain the probability corresponding to every word in the vocabulary as potential next word in the sequence


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    #this is for encoding the source language (say English).
    def encode(self, src, src_mask):
        assert src is not None
        assert src_mask is not None
        src = self.src_embed(src)
        src = self.src_pos(src)
        assert src is not None
        assert src_mask is not None
        return self.encoder(src, src_mask)
    
    #this takes in the corresponding sequence in the target language as tgt. Thus we need to convert the tokens in this sequence into the corresponding
    #embeddings and then the positional encoding. Finally, all this data is fed into the decoder with the target mask, the source maks and the input from the
    #encoder for the cross multihead attention
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)

#source sequence length and the target sequence length are not the sequence length of the current sentence that is being supplied to the transformer
#it is the maximum context length that the transformer network can support. Thus the model can take this many inputs at once.
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len, d_model: int = 512, N: int = 5, h: int = 4, dropout: float = 0.1, d_ff:int = 2048):
    #first we build the embedding layer to embed the source sentence and that target sentence
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    #now we need to create the positional encoding layer which simply encodes the position according to some formula and them adds this information to the
    #word embeddings. These final embeddings contain the meaning of the word and also contain information about the word in the sequence of inputs. 
    #However, this is only upto the context length as we only feed in words upto the context length, we only calculate the psotional encodings upto the context length
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    #now we create the encoder blocks - 
    encoder_blocks = list()
    for _ in range(N):
        #we are creating the layers here because for the initialization of the encoder block, we must pass in the layers themselves
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    #now we create the decoder blocks
    decoder_blocks = list()
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    #now we create the final Encoder and the Decoder which contain all the blocks here
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    #now we create the layer for the projection from the embedding vectors to the tgt_vocab_size
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    #now we finally create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    #initializing the parameters
    for p in transformer.parameters():
        if(p.dim() > 1):
            nn.init.xavier_uniform_(p)
    
    return transformer