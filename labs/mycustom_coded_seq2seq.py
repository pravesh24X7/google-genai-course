# this file contains the code segment of Seq2Seq model.
# I'll use my previously coded MyCustomEncoder and MyCustomDecoder model instances in this class.

import random
import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self,
                encoder,
                decoder,
                device='cpu'):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert (encoder.hidden_dim == decoder.hidden_dim), "Hidden layers must match"
        assert (encoder.no_of_layers == decoder.no_of_layers), "Encoder and Decoder must have same no_of_layers value"

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]    # source of shape=[B,S], this is what I need to encode
        target_len = target.shape[1]

        target_vocab_size = self.decoder.output_dim
        
        # outputs is `torch.zeros()` because, it represents the `<SOS>` token.
        # shape of this outputs=[B, target_len, output_dim], because we want `<SOS>` token for each sentences, and we know LSTM output is [B, S, E]
        # in case of decoder B=batch_size, S=target_len, E=output_dimension of embeddings. because decoder is using an additional projection layer of shape (hidden_dim, output_dim)
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(device=self.device)
        encoder_output, hidden, cell = self.encoder(source)

        input_ = target[:, 0]   # <SOS> token for each example
        for t in range(1, target_len):

            query = hidden[-1].unsqueeze(2)
            energy = torch.bmm(
                encoder_output, query
            )
            energy = energy.squeeze(2)
            attn_wts = torch.softmax(energy, dim=1)
            context = torch.bmm(
                attn_wts.unsqueeze(1),
                encoder_output
            )
            context = context.squeeze(1)
        
            output, hidden, cell = self.decoder(input_, context, hidden, cell)
            outputs[:, t, :] = output   # save each output of shape [B, output_dim]

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(dim=1)

            input_ = target[:, t] if teacher_force else top1
        return outputs      # outputs shape=[B, Target_len, Vocab_target]
        
        
