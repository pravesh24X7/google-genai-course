# this file contains the code segment for encoder only.
# I later plug this with Seq2Seq architecture model.

import torch
import torch.nn as nn

import numpy as np

class MyCustomEncoder(nn.Module):
    def __init__(self,
                input_dim,
                embed_dim,
                hidden_dim,
                no_of_layers,
                dropout=0.2,
                device='cpu'):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.no_of_layers = no_of_layers

        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, no_of_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)


    def forward(self, src):
        batch_size = src.shape[0]   # this will return the batch size
        embed = self.dropout(self.embedding(src)) # before shape: [B,S], after this: [B, S, E]
        hidden = torch.zeros( self.no_of_layers, batch_size, self.hidden_dim  ).to(device=self.device) # shape of hidden tensor is [L, B, H]
        cell = torch.zeros(self.no_of_layers, batch_size, self.hidden_dim).to(device=self.device)   # same shape as hidden tensor

        # output is of shape: [ B, S, H ]
        output, (hidden, cell) = self.rnn(embed, (hidden, cell))
        return output, hidden, cell
        
