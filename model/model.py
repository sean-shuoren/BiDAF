import torch
import torch.nn as nn
import torch.nn.functional as F

from model.util import Linear, CharCNN, HighwayMLP, SingleLayerLSTM


class BiDAF(nn.Module):
    def __init__(self, pretrain_embedding, char_vocab_size, consider_na=False, enable_c2q=True, enable_q2c=True,
                 hidden_size=100, char_emb_dim=8, char_channel_num=100, char_channel_width=5, dropout=0.2):
        super(BiDAF, self).__init__()
        assert enable_c2q or enable_q2c, "c2q attention and q2c attention cannot both be disabled"
        self.enable_c2q = enable_c2q
        self.enable_q2c = enable_q2c
        self.g_dim = 8 if enable_q2c else 6  # size of G matrix (x_len * g_dim)

        # Character Embedding Layer
        self.char_emb = CharCNN(char_emb_dim=char_emb_dim,
                                char_vocab_size=char_vocab_size,
                                channel_num=char_channel_num,
                                channel_width=char_channel_width,
                                dropout=dropout)
        # Word Embedding Layer
        self.word_emb = nn.Embedding.from_pretrained(pretrain_embedding, freeze=True)
        self.highway = HighwayMLP(input_size=hidden_size*2,
                                  output_size=hidden_size*2,
                                  num_layer=2)
        # Contextual Embedding Layer
        self.contextual_emb = SingleLayerLSTM(input_size=hidden_size*2,
                                              hidden_size=hidden_size,
                                              bidirectional=True, dropout=dropout)
        # Attention Flow Layer
        self.ws_h = Linear(hidden_size * 2, 1, dropout)
        self.ws_u = Linear(hidden_size * 2, 1, dropout)
        self.ws_hu = Linear(hidden_size * 2, 1, dropout)
        # Modeling Layer
        self.modeling_lstm_1 = SingleLayerLSTM(input_size=hidden_size * self.g_dim,
                                               hidden_size=hidden_size,
                                               bidirectional=True, dropout=dropout)
        self.modeling_lstm_2 = SingleLayerLSTM(input_size=hidden_size * 2,
                                               hidden_size=hidden_size,
                                               bidirectional=True, dropout=dropout)
        # Output Layer
        self.output_lstm = SingleLayerLSTM(input_size=hidden_size * 2,
                                           hidden_size=hidden_size,
                                           bidirectional=True, dropout=dropout)
        self.wp1_g = Linear(hidden_size * self.g_dim, 1, dropout=dropout)
        self.wp1_m = Linear(hidden_size * 2, 1, dropout=dropout)
        self.wp2_g = Linear(hidden_size * self.g_dim, 1, dropout=dropout)
        self.wp2_m = Linear(hidden_size * 2, 1, dropout=dropout)
        # Consider non-answerable cases
        self.consider_na = consider_na
        if consider_na:
            self.na_bias = torch.tensor([[1.0]])

    def bidaf(self, h, u):
        t = h.size(1)  # x_len, h: (batch, x_len, hidden*2)
        j = u.size(1)  # q_len, u: (batch, q_len, hidden*2)
        hh = h.unsqueeze(2).expand(-1, -1, j, -1)  # (batch, x_len, q_len, hidden*2)
        uu = u.unsqueeze(1).expand(-1, t, -1, -1)  # (batch, x_len, q_len, hidden*2)
        s = self.ws_h(hh) + self.ws_u(uu) + self.ws_hu(hh * uu)  # (batch, x_len, q_len)
        s = s.squeeze()

        if self.enable_c2q:
            a = F.softmax(s, dim=2)     # (batch, x_len, q_len)
            c2q_att = torch.bmm(a, u)   # (batch, x_len, hidden*2)
            del a
        else:
            c2q_att = u.mean(dim=1, keepdim=True).expand(-1, t, -1)

        if self.enable_q2c:
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)  # (batch, 1, x_len)
            q2c_att = torch.bmm(b, h).squeeze()                        # (batch, hidden*2)
            q2c_att = q2c_att.unsqueeze(1).repeat(1, t, 1)             # (batch, x_len, hidden*2)
            del b

        del s
        return torch.cat((h, c2q_att, h * c2q_att, h * q2c_att), dim=-1) if self.enable_q2c \
            else torch.cat((h, c2q_att, h * c2q_att), dim=-1)

    def forward(self, batch):
        # Character Embedding Layer
        x_char_emb = self.char_emb(batch.x_char)
        q_char_emb = self.char_emb(batch.q_char)
        # Word Embedding Layer
        x_word_emb = self.word_emb(batch.x_word[0])
        q_word_emb = self.word_emb(batch.q_word[0])
        x_lens = batch.x_word[1]
        q_lens = batch.q_word[1]
        x = self.highway(x_char_emb, x_word_emb)
        q = self.highway(q_char_emb, q_word_emb)
        del x_char_emb, q_char_emb, x_word_emb, q_word_emb
        # Contextual Embedding Layer
        h = self.contextual_emb(x, x_lens)
        u = self.contextual_emb(q, q_lens)
        del x, q
        # Attention Flow Layer
        g = self.bidaf(h, u)
        del h, u
        # Modeling Layer
        m = self.modeling_lstm_1(g, x_lens)
        m = self.modeling_lstm_2(m, x_lens)
        # Output Layer
        p1 = (self.wp1_g(g) + self.wp1_m(m)).squeeze()
        m2 = self.output_lstm(m, x_lens)
        p2 = (self.wp2_g(g) + self.wp2_m(m2)).squeeze()
        del g, m, m2
        # Add NA bias
        if self.consider_na:
            batch_size = len(batch)
            p1 = torch.cat([p1, self.na_bias.expand(batch_size, -1)], dim=-1)
            p2 = torch.cat([p2, self.na_bias.expand(batch_size, -1)], dim=-1)
        return p1, p2
