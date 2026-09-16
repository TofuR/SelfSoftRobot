"""SPONGE GRU/LSTM model classes, copied verbatim from official source.

Source: https://github.com/tlhabich/sponge
Commit: fffb24a063475a3bca09f04968be0d165ad3f90a
File: rnn_mpc/software/rnn_training/NN_fcn.py (classes GRU and LSTM)
Only the two model classes are included; imports are restricted to torch.nn.
Copyright (c) 2023 Tim-Lukas Habich. MIT license: SPONGE_LICENSE.txt.
"""
from torch import nn

class GRU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer, dropout):
        super(GRU, self).__init__()
        self.ident = "GRU"
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gru = nn.GRU(input_size = input_dim, hidden_size = hidden_dim, num_layers = num_layer, batch_first = True, dropout = dropout )
        self.dense = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, ht):
        out, ht = self.gru(x, ht)
        out = out[:, -1, :]  # Select the output of the last time step
        out = self.dense(out)
        return out, ht

class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer, dropout):
        super(LSTM, self).__init__()
        self.ident = "LSTM"
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = num_layer, batch_first = True, dropout = dropout )
        self.dense = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, ht, ct):
        out, (ht, ct) = self.lstm(x, (ht, ct))
        out = out[:, -1, :]  # Select the output of the last time step
        out = self.dense(out)
        return out, ht, ct
