import math
import copy
import torch
import torch.nn as nn
import numpy as np

from bert_modules import BertConfig, BertEncoder, BertPooler

class Backend(nn.Module):
    '''
    Heavily modified from https://github.com/minzwon/sota-music-tagging-models/
    '''
    def __init__(self,main_dict, 
                 bert_config = None):
        super(Backend, self).__init__()

        backend_dict = main_dict["backend_dict"]
        self.frontend_out_channels = main_dict["frontend_dict"]["list_out_channels"][-1]
        
        self.seq2seq = None
        # seq2seq for position encoding
        if backend_dict["recurrent_units"] is not None:
            self.seq2seq = nn.GRU(self.frontend_out_channels, 
                                  self.frontend_out_channels, 
                                  backend_dict["recurrent_units"],
                                  batch_first=True) # input and output = (batch, seq, feature)
        
        self.dropout1 = nn.Dropout(0.1)
        
        # Transformer encoder
        if bert_config is None:
            bert_config = BertConfig(hidden_size=self.frontend_out_channels,
                                     num_hidden_layers=2,
                                     num_attention_heads=8,
                                     intermediate_size=1024,
                                     hidden_dropout_prob=0.4,
                                     attention_probs_dropout_prob=0.5)
        
        self.encoder = BertEncoder(bert_config)
        self.pooler = BertPooler(bert_config)
        self.single_cls = self.get_cls()
        
        # Dense
        self.dropout2 = nn.Dropout(0.5)
        self.dense = nn.Linear(self.frontend_out_channels, backend_dict["n_class"])
        
    def get_cls(self,):
        torch.manual_seed(42) # for reproducibility
        single_cls = torch.rand((1, self.frontend_out_channels))
        return single_cls

    def append_cls(self, x):
        # insert always the same token as a reference for classification
        vec_cls = self.single_cls.repeat(x.shape[0],1,1) # batch_size = x.shape[0]
        vec_cls = vec_cls.to(x.device)
        return torch.cat([vec_cls, x], dim=1)

    def forward(self, x):
        
        # positional encoding
        if self.seq2seq is not None:
            
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.seq2seq.flatten_parameters() 
            
            # frontend output shape = (batch, features, sequence)
            # change to (sequence, batch, features)
            x = x.permute(2, 0, 1) 
            x,_ = self.seq2seq(x)
            
            # change back to (batch, features, sequence)
            x = x.permute(1, 2, 0) 
            
        # dropout
        x = self.dropout1(x)
        
        # frontend output shape = (batch, features, sequence)
        # input to self attention (batch, sequence, features)
        x = x.permute(0, 2, 1)
        
        # Get [CLS] token
        x = self.append_cls(x)

        # Transformer encoder
        _,x = self.encoder(x)

        x = self.pooler(x)

        # Dense
        x = self.dropout2(x)
        x = self.dense(x)
        x = nn.Sigmoid()(x)

        return x