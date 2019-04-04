import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy


def copy_module(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        return x


class CharCNN(nn.Module):
    def __init__(self, char_emb_dim, char_vocab_size, channel_num, channel_width, dropout):
        super(CharCNN, self).__init__()
        self.char_emb_dim = char_emb_dim
        self.channel_num = channel_num
        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim)
        self.char_cnn = nn.Conv2d(1, channel_num, (channel_width, char_emb_dim))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_len = x.size(0)
        seq_len = x.size(1)
        x = self.dropout(self.char_embedding(x))                            # (batch_len, seq_len, word_len, char_dim)
        x = x.view(batch_len*seq_len, -1, self.char_emb_dim).unsqueeze(1)   # (batch * seq_len, 1, word_len, char_dim)
        x = self.char_cnn(x).squeeze()                                      # (batch * seq_len, channel_num, convolved)
        x = F.max_pool1d(x, x.size(-1)).squeeze()                           # (batch * seq_len, channel_num)
        x = x.view(batch_len, -1, self.channel_num)                         # (batch, seq_len, channel_num)
        return x


class HighwayMLP(nn.Module):
    def __init__(self, input_size, output_size, num_layer=2):
        super(HighwayMLP, self).__init__()
        self.num_layer = num_layer
        self.gate = copy_module(
            nn.Sequential(nn.Linear(input_size, output_size), nn.Sigmoid()),
            num_layer)
        self.transform = copy_module(
            nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU()),
            num_layer)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        for i in range(self.num_layer):
            t = self.transform[i](x)
            g = self.gate[i](x)
            x = t * g + (1-g) * x
        return x


class SingleLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional, dropout):
        super(SingleLayerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1,
                            batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_len):
        x = self.dropout(x)

        sorted_x_len, x_idx = torch.sort(x_len, descending=True)
        sorted_x = x.index_select(dim=0, index=x_idx)
        _, x_ori_idx = torch.sort(x_idx)

        x_packed = nn.utils.rnn.pack_padded_sequence(sorted_x, sorted_x_len, batch_first=True)
        x_packed, _ = self.lstm(x_packed, None)

        x, _ = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
        x = x.index_select(dim=0, index=x_ori_idx)
        return x
