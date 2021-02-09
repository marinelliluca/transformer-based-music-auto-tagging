import math
import copy
import torch
import torch.nn as nn
import numpy as np

from .bert_modules import BertConfig, BertEncoder, BertPooler

class Backend(nn.Module):
    '''
    Modified from https://github.com/minzwon/sota-music-tagging-models/
    '''
    def __init__(self,backend_dict, 
                 bert_config = None):
        super(Backend, self).__init__()

        
        self.frontend_channels = backend_dict["frontend_channels"]
        
        self.recurrent = backend_dict["recurrent"]
        
        # seq2seq for position encoding
        if backend_dict["recurrent_units"] is not None:
            self.seq2seq = nn.GRU(backend_dict["frontend_channels"], 
                                  backend_dict["frontend_channels"], 
                                  backend_dict["recurrent_units"], 
                                  batch_first=True) # input and output = (batch, seq, feature)
            
        # Transformer encoder
        if bert_config is None:
            bert_config = BertConfig(hidden_size=backend_dict["frontend_channels"],
                                     num_hidden_layers=2,
                                     num_attention_heads=8,
                                     intermediate_size=1024,
                                     hidden_dropout_prob=0.4,
                                     attention_probs_dropout_prob=0.5)
        
        self.encoder = BertEncoder(bert_config)
        self.pooler = BertPooler(bert_config, activation=nn.ELU())
        self.single_cls = self.get_cls()
        
        # Dense
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(backend_dict["frontend_channels"], backend_dict["n_class"])
        
    def get_cls(self,):
        torch.manual_seed(42) # for reproducibility
        single_cls = torch.rand((1, self.frontend_channels))
        return single_cls

    def append_cls(self, x):
        # insert always the same token as a reference for classification
        vec_cls = self.single_cls.repeat(x.shape[0],1,1) # batch_size = x.shape[0]
        vec_cls = vec_cls.to(x.device)
        return torch.cat([vec_cls, x], dim=1)

    def forward(self, x):
        
        x = x.permute(0, 2, 1) # (Batch,Sequence,Features)
        
        # positional encoding
        if self.seq2seq is not None:
            x,_ = self.seq2seq(x)
        
        # Get [CLS] token
        x = self.append_cls(x)

        # Transformer encoder
        _,x = self.encoder(x)

        x = self.pooler(x)

        # Dense
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.Sigmoid()(x)

        return x