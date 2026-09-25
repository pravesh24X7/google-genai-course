# this file contains the code segment of decoder class.
# I later plug this into Seq2Seq architecture as well, similar to encoder class.

import torch
import torch.nn as nn

import numpy as np


class MyCustomDecoder(nn.Module):
    def __init__(self,
                output_dim,
                emb_dim,
                hidden_dim,
                no_of_layers,
                dropout=0.2,
                ):
        super().__init__()
        self.no_of_layers = no_of_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM((emb_dim + hidden_dim), hidden_dim, no_of_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_, context, hidden, cell):
        input_ = input_.unsqueeze(1)    # before shape=[B], after shape=[B,1]
        embedding = self.dropout(self.embedding(input_))  # before shape=[B,1], after shape=[B, 1, E]

        context = context.unsqueeze(1)
        rnn_input = torch.cat(
            (embedding, context),
            dim=2
        )

        # shape of output = [ B, 1, H  ]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))    # shape of hidden and cell = [L, B, H]
        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden, cell     # shape of prediction willl be [B, outpu_dim]
