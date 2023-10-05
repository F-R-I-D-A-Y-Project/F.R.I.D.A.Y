import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy,re
import warnings
import pandas as pd
import numpy as np
import torchtext
import csv, json, pickle
import pathlib
import subprocess
from typing import Self
warnings.simplefilter("ignore")


class Embedding(nn.Module):
    def __init__(self: Self, vocab_size, embed_dim):
        '''
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
        '''
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    def forward(self: Self, x):
        '''
        Args:
            x: input vector
        Returns:
            out: embedding vector
        '''
        return self.embed(x)



class PositionalEmbedding(nn.Module):
    def __init__(self: Self,max_seq_len,embed_model_dim):
        '''
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        '''
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len,self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self: Self, x):
        '''
        Args:
            x: input vector
        Returns:
            x: output
        '''
      
        x = x * math.sqrt(self.embed_dim)
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self: Self, embed_dim=512, n_heads=8):
        '''
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        '''
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim 
        self.n_heads = n_heads  
        self.single_head_dim = int(self.embed_dim / self.n_heads)  
       
        self.query_matrix = nn.Linear(self.single_head_dim , self.single_head_dim ,bias=False)  
        self.key_matrix = nn.Linear(self.single_head_dim  , self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim ,self.single_head_dim , bias=False)
        self.out = nn.Linear(self.n_heads*self.single_head_dim ,self.embed_dim) 

    def forward(self: Self,key,query,value,mask=None):     
        '''
        Args:
           key : key vector
           query : query vector
           value : value vector
           mask: mask for decoder
        
        Returns:
           output vector from multihead attention
        '''
        batch_size = key.size(0)
        seq_length = key.size(1)
        
        seq_length_query = query.size(1)
        
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim) 
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim) 
       
        k = self.key_matrix(key)
        q = self.query_matrix(query)   
        v = self.value_matrix(value)

        q = q.transpose(1,2)
        k = k.transpose(1,2)  
        v = v.transpose(1,2)  
       
        k_adjusted = k.transpose(-1,-2)
        product = torch.matmul(q, k_adjusted)
      
        
        if mask is not None:
             product = product.masked_fill(mask == 0, float("-1e20"))

        product = product / math.sqrt(self.single_head_dim) 

        scores = F.softmax(product, dim=-1)
 
        scores = torch.matmul(scores, v)   
        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)
        output = self.out(concat)
       
        return output
    

class EncoderBlock(nn.Module):
    def __init__(self: Self, embed_dim, expansion_factor=4, n_heads=8):
        super(EncoderBlock, self).__init__()
        
        '''
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads
        
        '''
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
                          nn.Linear(embed_dim, expansion_factor*embed_dim),
                          nn.ReLU(),
                          nn.Linear(expansion_factor*embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self: Self,key,query,value):
        
        '''
        Args:
           key: key vector
           query: query vector
           value: value vector
           norm2_out: output of transformer block
        
        '''
        
        attention_out = self.attention(key,query,value)  
        attention_residual_out = attention_out + value  
        norm1_out = self.dropout1(self.norm1(attention_residual_out)) 

        feed_fwd_out = self.feed_forward(norm1_out)  
        feed_fwd_residual_out = feed_fwd_out + norm1_out 
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out)) 

        return norm2_out



class TransformerEncoder(nn.Module):
    '''
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention
        
    Returns:
        out: output of the encoder
    '''
    def __init__(self: Self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerEncoder, self).__init__()
        
        self.embedding_layer = Embedding(vocab_size, embed_dim)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList([EncoderBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
    
    def forward(self: Self, x):
        embed_out = self.embedding_layer(x)
        out = self.positional_encoder(embed_out)
        for layer in self.layers:
            out = layer(out,out,out)

        return out  


class DecoderBlock(nn.Module):
    def __init__(self: Self, embed_dim, expansion_factor=4, n_heads=8):
        super(DecoderBlock, self).__init__()

        '''
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads
        
        '''
        self.attention = MultiHeadAttention(embed_dim, n_heads=8)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = EncoderBlock(embed_dim, expansion_factor, n_heads)
        
    
    def forward(self: Self, key, query, x,mask):
        
        '''
        Args:
           key: key vector
           query: query vector
           value: value vector
           mask: mask to be given for multi head attention 
        Returns:
           out: output of transformer block
    
        '''
        
        
        attention = self.attention(x,x,x,mask=mask) 
        value = self.dropout(self.norm(attention + x))
        
        out = self.transformer_block(key, query, value)

        
        return out


class TransformerDecoder(nn.Module):
    def __init__(self: Self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerDecoder, self).__init__()
        '''  
        Args:
           target_vocab_size: vocabulary size of taget
           embed_dim: dimension of embedding
           seq_len : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention
        
        '''
        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=4, n_heads=8) 
                for _ in range(num_layers)
            ]

        )
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self: Self, x, enc_out, mask):
        
        '''
        Args:
            x: input vector from target
            enc_out : output from encoder layer
            trg_mask: mask for decoder self attention
        Returns:
            out: output vector
        '''
            
        
        x = self.word_embedding(x)  
        x = self.position_embedding(x) 
        x = self.dropout(x)
     
        for layer in self.layers:
            x = layer(enc_out, x, enc_out, mask) 

        out = F.softmax(self.fc_out(x))

        return out


class Transformer(nn.Module):
    def __init__(self: Self, embed_dim, src_vocab_size, target_vocab_size, seq_length,num_layers=2, expansion_factor=4, n_heads=8):
        super(Transformer, self).__init__()
        
        '''  
        Args:
           embed_dim:  dimension of embedding 
           src_vocab_size: vocabulary size of source
           target_vocab_size: vocabulary size of target
           seq_length : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention
        
        '''
        
        self.target_vocab_size = target_vocab_size

        self.encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        self.decoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        
    
    def make_trg_mask(self: Self, trg):
        '''
        Args:
            trg: target sequence
        Returns:
            trg_mask: target mask
        '''
        batch_size, trg_len = trg.shape
        
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask    

    def decode(self: Self,src,trg):
        '''
        for inference
        Args:
            src: input to encoder 
            trg: input to decoder
        out:
            out_labels : returns final prediction of sequence
        '''
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)
        out_labels = []
        batch_size,seq_len = src.shape[0],src.shape[1]
        #outputs = torch.zeros(seq_len, batch_size, self.target_vocab_size)
        out = trg
        for i in range(seq_len): 
            out = self.decoder(out,enc_out,trg_mask) 
            out = out[:,-1,:]
     
            out = out.argmax(-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(out,axis=0)
          
        
        return out_labels
    
    def forward(self: Self, src, trg):
        '''
        Args:
            src: input to encoder 
            trg: input to decoder
        out:
            out: final vector which returns probabilities of each target word
        '''
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)
   
        outputs = self.decoder(trg, enc_out, trg_mask)
        return outputs


class Model:
    '''
        This class is responsible for the NLP model of the chatbot.
    '''
    def __init__(self: Self, path_to_dataset: str) -> None:
        '''
            Constructor of the class. It receives the path to the dataset, but does not train the model.
            
            Args:
                path_to_dataset (str): path to dataset used for training
        '''
        self.__dataset = path_to_dataset

    def fit(self: Self, path_to_dataset: str|None=None) -> None:
        '''
            This method is responsible for training the model.
            It reads the dataset, captures the amount of unique words existent in the dataset, 
            creates the Transformers Model and trains it

            If the dataset was unchanged, it loads the model from the pickle file. If not, the training algorithm will be executed.

            Args:
                path_to_dataset (str): path to dataset used for training
        '''
        if path_to_dataset:
            self.__dataset = path_to_dataset

        # self.__model = Transformer(...)
        # self.__tokenizer = Tokenizer(...)
        if (pathlib.Path(__file__).parent.parent.parent.parent / 'model.pkl').exists():
            with open('model.pkl', 'rb') as f:
                self.__model = pickle.load(f)
        else:
            self.__train()

    def predict(self: Self, message: str) -> str:
        '''
            This method is responsible for returning the answer of the model for the chatbot.
            It receives a message, tokenizes it, and passes it to the Transformers Model

            Args:
                message (str): message to be answered by the chatbot

        '''
        return '?'

    def __serialize_model(self: Self) -> None:
        '''
            Serializes the model into a pickle file to avoid retraining it every time the chatbot is executed.
        '''

        with (pathlib.Path(__file__).parent.parent.parent.parent / 'model.pkl').open('wb') as f:
            pickle.dump(self.__model, f)

    def __train(self: Self) -> None:
        '''
            Training algorithm of the Tranformers model
        '''
        with open(self.__dataset, 'r') as f:
            data = csv.reader(f)

        self.__serialize_model()

    def check_db_change(self: Self) -> None:
        '''
            Verifies changes in the dataset. If there are changes, it will delete the serialized model.
        '''
        out = subprocess.run(f'git diff {self.__dataset}', shell=True, cwd=pathlib.Path(__file__).parent.parent.parent.parent.absolute(),
                       capture_output=True).stdout.strip()
        if out:
            subprocess.run(f'rm -f model.pkl', shell=True, cwd=pathlib.Path(__file__).parent.parent.parent.parent.absolute())
            subprocess.run([f'git add {self.__dataset}', 'git commit -m "Update dataset"'], shell=True, cwd=pathlib.Path(__file__).parent.parent.parent.parent.absolute())
            