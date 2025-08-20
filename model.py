import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    """
    Configuration for the Transformer model.
    """
    embedding_dimension: int = 512
    num_attention_heads: int = 8
    attention_dropout_p: float = 0.0
    hidden_dropout_p: float = 0.0
    mlp_ratio: int = 4
    encoder_depth: int = 6
    decoder_depth: int = 6
    
    src_vocab_size: int = 30522
    tgt_vocab_size: int = 32000
    
    max_src_length: int = 512
    max_tgt_length: int = 512
    learn_pos_embed: bool = False
    


class PositionalEncoding(nn.Module):
    
    """
    Sin/Cosine (non-learnable) encodings proposed in Attention is All You Need

    Args:
        max_len: Maximum number of tokens possible in a sequence
        embed_dim: Embedding dimension of each token
    """
    
    def __init__(self,max_length, embedding_dim, require_grad=False):
        
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.require_grad = require_grad
        
        self.encoding = self._positional_encoding()
        
    def _positional_encoding(self):
        
        encodings = torch.zeros(self.max_length,self.embedding_dim, dtype=torch.float32)
        position_idx = torch.arange(0, self.max_length, dtype=torch.float32).reshape(-1, 1)
        embed_dim_skip_idx = torch.arange(0, self.embedding_dim, step=2, dtype=torch.float32)
        ##print(embed_dim_skip_idx.shape)
        ##print(position_idx.shape)
        
        encodings[:, 0::2] = torch.sin(position_idx / (10000 ** (embed_dim_skip_idx / self.embedding_dim)))
        encodings[:, 1::2] = torch.cos(position_idx / (10000 ** (embed_dim_skip_idx / self.embedding_dim)))
        
        encodings = nn.Parameter(encodings, requires_grad=self.require_grad)
        
        ##print(encodings.shape)
        ##print(encodings)
        return encodings
    
    def forward(self, x):
        ###(B x Seq_len x Embedding_dim) = x.shape
        seq_len = x.shape[1]
        
        encodings = self.encoding[: seq_len]
        
        x = x + encodings
        
        return x 
    
class Embeddings(nn.Module):
    
    def __init__(self, config):
        
        super(Embeddings, self).__init__()
        
        self.src_embedding = nn.Embedding(config.src_vocab_size, config.embedding_dimension)
        self.tgt_embedding = nn.Embedding(config.tgt_vocab_size, config.embedding_dimension)
        
        self.src_positional_encoding = PositionalEncoding(config.max_src_length,
                                                          config.embedding_dimension,
                                                          config.learn_pos_embed)
        
        self.tgt_positional_encoding = PositionalEncoding(config.max_tgt_length,
                                                          config.embedding_dimension,
                                                          config.learn_pos_embed)
    
    
    ## Architecture: # Input -> Embedding -> Positional Encoding -> Output {Positional Encodings are added to the embeddings which provides the positional information}
        
    def forward_src(self, input_ids):
        
        encodings = self.src_embedding(input_ids)
        encodings = self.src_positional_encoding(encodings)
        
        return encodings
    
    def forward_tgt(self, input_ids):
        
        encodings = self.tgt_embedding(input_ids)
        encodings = self.tgt_positional_encoding(encodings)
        
        return encodings
        
class Attention(nn.Module):
    
    def __init__(self, config):
        """
        Regular Self-Attention but in this case we utilize flash_attention
        incorporated in the F.scaled_dot_product_attention to speed up our training. 
        """
        super(Attention, self).__init__()
        
        self.config = config
        
        assert config.embedding_dimension % config.num_attention_heads == 0, "Double Check the embedding dimension and number of attention heads"
        
        self.head_dimension =  config.embedding_dimension // config.num_attention_heads
        
        self.q_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.k_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.v_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        
        self.out_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        
    def forward(self, src, tgt = None, attention_mask = None, causal=False):
        
        """
        This forward function handles all the cases we need. Lets pretend we are doing English to French
        
            - We can provide English as the src along with its padding mask for Encoder self-attention
            - We can provide French as the src along with its padding mask and causal as True for decoder self-attention
            - We can provide English as src and French as tgt along with the src padding_mask for cross attention

        ### ATTENTION MASK FOR SELF-ATTENTION ###

        Attention Mask is in (Batch x Sequence Length) where we have False for tokens we don't want to attend to (from F.SDPA in PyTorch) ###
        F.scaled_dot_product_attention expects a mask of the shape (Batch x ..., x Seq_len x Seq_len) ###
        the "..." in this case is any extra dimensions (such as heads of attention). lets expand our mask to (Batch x 1 x Seq_len x Seq_len) ###
        The 1 in this case refers to the number of heads of attention we want, so it is a dummy index to broadcast over ###
        In each (Seq_len x Seq_len) matrix for every batch, we want False for all columns corresponding to padding tokens ### 

        ### ATTENTION MASK FOR CROSS-ATTENTION ###

        When doing cross attention, our French will be (Batch x french_len x embed_dim) and our English will be (Batch x english_len x embed_dim)
        In typical cross attention fashion, the queries will be the thing we want and Keys/Values will be the thing we are crossing with. In our 
        Decoder Cross Attention, we want to learn how our generated French is related to the encoded english from the Encoder. So our Queries will be
        French and Keys/Values will be the encoded English. 

        Q @ K^T will then give a shape (Batch x ... x french_len x english_len). This means our attention mask also has to have this shape! Just like
        before, we want to mask out the columns of the attention mask, so our french tokens dont attend to any english padding tokens. We can then take
        our english padding mask which is (Batch x english_len), add extra dimensions for head and src_len dimension which will give a 
        (Batch x 1 x 1 x english_len) and then repeat the mask for the source length (batc x 1 x french_len x english_len)

        """
        
        batch_size, src_len, embed_dim = src.shape
        ## Self Attention Case
        if tgt is None:
            ## break the Embedding Dimension into num_attention_heads and head_dimension
            ## lets see we have the shape of (B * Seq_len * Embedding_dim) - but we need to reshape it to (B x Seq_len x num_attention_heads x head_dimension)
            q = self.q_proj(src).reshape(batch_size, src_len, self.config.num_attention_heads, self.head_dimension).transpose(1,2).contiguous()
            k = self.k_proj(src).reshape(batch_size, src_len, self.config.num_attention_heads, self.head_dimension).transpose(1,2).contiguous()
            v = self.v_proj(src).reshape(batch_size, src_len, self.config.num_attention_heads, self.head_dimension).transpose(1,2).contiguous()
            
            if attention_mask is not None:
                
                ## [B * Seq_len] -> [B * 1 * Seq_len * Seq_len]
                ## attention_mask = attention_mask.unsqueeze(1).unsqueeze(1) [Adds 2 dummy dimensions so here we repeat the mask for num_attention_heads and src_len]
                attention_mask =  attention_mask.bool()
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1,1,src_len,1)
            if causal:
                attention_output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1, is_causal=True)
            else:
                attention_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.1)

            # attention_output = F.scaled_dot_product_attention(q,k,v,
            #                                                   attn_mask=attention_mask,
            #                                                   dropout_p=self.config.attention_dropout_p if self.training else 0.0,
            #                                                   is_causal=causal)
        ## Cross Attention Case    
        else:
            #if tgt is not None:
            # (B * tgt_len(seq_len) * embed_dim) = tgt.shape here we use the shape[1] for storing the shape dim
            tgt_len = tgt.shape[1]
            
            q = self.q_proj(tgt).reshape(batch_size, tgt_len, self.config.num_attention_heads, self.head_dimension).transpose(1,2).contiguous()
            k = self.k_proj(src).reshape(batch_size, src_len, self.config.num_attention_heads, self.head_dimension).transpose(1,2).contiguous()
            v = self.v_proj(src).reshape(batch_size, src_len, self.config.num_attention_heads, self.head_dimension).transpose(1,2).contiguous()
            
            if attention_mask is not None:
                ## [B * Seq_len] -> [B * 1 * tgt_len * src_len]
                attention_mask = attention_mask.bool()
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).repeat(1,1,tgt_len,1)
                
            attention_output = F.scaled_dot_product_attention(q,k,v,
                                                           attn_mask=attention_mask, 
                                                           dropout_p=self.config.attention_dropout_p if self.training else 0.0, 
                                                           is_causal=False)
            
        attention_out = attention_output.transpose(1,2).flatten(2)
        attention_out = self.out_proj(attention_out)
        #print(attention_out.shape)
        return attention_out

class FeedForward(nn.Module):
    
    def __init__(self, config):
        
        super(FeedForward, self).__init__()
        
        self.hidden_size = config.embedding_dimension * config.mlp_ratio

        self.intermediate_hidden_dense = nn.Linear(config.embedding_dimension, self.hidden_size)
        self.activation = nn.GELU()
        self.intermediate_dropout = nn.Dropout(config.hidden_dropout_p)
        
        self.output_dense = nn.Linear(self.hidden_size, config.embedding_dimension)
        self.output_dropout = nn.Dropout(config.hidden_dropout_p)
        
    def forward(self, x):
        
        x = self.intermediate_hidden_dense(x)
        x = self.activation(x)
        x = self.intermediate_dropout(x)
        
        x = self.output_dense(x)
        x = self.output_dropout(x)
        
        return x
    
class TransformerEncoder(nn.Module):
    
    """
    Stacks together a Self-Attention module and MLP Layer
    """
    
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        
        self.enc_attention = Attention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_p)
        self.layer_norm = nn.LayerNorm(config.embedding_dimension)
        
        self.feed_forward = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.embedding_dimension)
        
    def forward(self, x, attention_mask=None):
        
        attention_out = self.enc_attention(x, attention_mask=attention_mask, causal=False)
        x = x + self.dropout(attention_out)
        x = self.layer_norm(x)
        
        x = x + self.feed_forward(x)
        x = self.final_layer_norm(x)
        
        return x
            
class TransformerDecoder(nn.Module):
    
    """
    Stacks together a Self-Attention module, Cross-Attention module and MLP Layer
    """
    
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
    
        self.dec_attention = Attention(config)
        self.dec_attention_drop = nn.Dropout(config.hidden_dropout_p)
        self.dec_attention_layernorm = nn.LayerNorm(config.embedding_dimension)
        
        self.cross_attention = Attention(config)
        self.cross_attention_drop = nn.Dropout(config.hidden_dropout_p)
        self.cross_attention_layernorm = nn.LayerNorm(config.embedding_dimension)
        
        self.feed_forward = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.embedding_dimension)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        
        tgt = tgt + self.dec_attention_drop(self.dec_attention(src=tgt, attention_mask=tgt_mask, causal=True))
        tgt = self.dec_attention_layernorm(tgt)
        
        tgt = tgt + self.cross_attention_drop(self.cross_attention(src=src, tgt=tgt, attention_mask=src_mask))
        tgt = self.cross_attention_layernorm(tgt)
        
        tgt = tgt + self.feed_forward(tgt)
        tgt = self.final_layer_norm(tgt)
        
        return tgt
    
class Transformer(nn.Module):
    
    def __init__(self, config):
        super(Transformer, self).__init__()
        
        self.config = config
        
        self.embeddings = Embeddings(config)
        
        self.encoder = nn.ModuleList(
            [
                TransformerEncoder(config) for _ in range(config.encoder_depth)
            ]
        )
        
        self.decoder = nn.ModuleList(
            [
                TransformerDecoder(config) for _ in range(config.decoder_depth)
            ]
        )
        
        self.output_layer = nn.Linear(config.embedding_dimension, config.tgt_vocab_size)
        
        self.apply(_init_weights_)
        
    def forward(self, src_ids, tgt_ids, src_attention_mask=None, tgt_attention_mask=None):
        
        src_embeddings = self.embeddings.forward_src(src_ids)
        tgt_embeddings = self.embeddings.forward_tgt(tgt_ids)
        
        ##print(src_embeddings.shape,tgt_embeddings.shape)
        
        for layer in self.encoder:
            src_embeddings = layer(src_embeddings, src_attention_mask)
            
        for layer in self.decoder:
            tgt_embeddings = layer(src_embeddings, tgt_embeddings, src_attention_mask, tgt_attention_mask)
        
        pred = self.output_layer(tgt_embeddings)
        
        return pred    
        #   print(src_embeddings.shape)    
        #   print(tgt_embeddings.shape)
        
    def inference(self,
                  src_ids,
                  tgt_start_id=2,
                  tgt_end_id=3,
                  max_len=512):
        ## inference function is used to generate the output sequence given the input sequence(not a batch a single sequence)
        tgt_ids = torch.tensor([tgt_start_id], device=src_ids.device).reshape(1,1)
        
        src_embeddings = self.embeddings.forward_src(src_ids)
        
        for layer in self.encoder:
            src_embeddings = layer(src_embeddings)
            
        for i in range(max_len-1):
            
            tgt_embeddings = self.embeddings.forward_tgt(tgt_ids)
            
            for layer in self.decoder:
                
                tgt_embeddings = layer(src_embeddings, tgt_embeddings)
                
            #print(tgt_embeddings.shape)
            ## we need only the last token for the next prediction
            tgt_embeddings = tgt_embeddings[:,-1]    
            
            pred = self.output_layer(tgt_embeddings)
            
            pred = pred.argmax(axis=1).unsqueeze(0)# (1, 1)
            #print(tgt_ids.shape, pred.shape)
            tgt_ids = torch.cat([tgt_ids,pred], axis=-1)
            
            #print(tgt_ids)
            if torch.all(pred == tgt_end_id):
                break
            
            #print(tgt_ids)
        
        return tgt_ids.unsqueeze().cpu().tolist()
    
def _init_weights_(module):

    """
    Simple weight intialization taken directly from the huggingface
    `modeling_roberta.py` implementation! 
    """
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
        

if __name__ == "__main__":
    config = TransformerConfig()
    transformer = Transformer(config)
    
    english = torch.randint(low=0, high=1000, size=(1,32))
    french = torch.randint(low=0, high=1000, size=(2,48))
    
    transformer.inference(english)
    #print(english)
    #print(transformer(english,french).shape)


# config = TransformerConfig()
# encoder = TransformerEncoder(config)
# src = torch.rand(2, 35, 512)  # Batch size of
# print(encoder(src))
    
    
    
# config = TransformerConfig()
# ff = FeedForward(config)
# src = torch.rand(2,35,512)    
# print(ff(src).shape)  # -> torch.Size([2, 35, 512])

#pe = PositionalEncoding(512,512)

# config = TransformerConfig()
# attn = Attention(config)
# src = torch.rand(2,35,512)
# attn(src) -> output of self-attention - torch.Size([2, 35, 512])
# tgt = torch.rand(2,60,512)        
# attn(src, tgt) # -> output of cross-attention - torch.Size([2, 60, 512])
    
# torch.Size([256])
# torch.Size([512, 1])
# torch.Size([512, 512])
# tensor([0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
#         0., 1., 0., 1., 0., 1., 0., 1.])    



