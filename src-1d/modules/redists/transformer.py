import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/CyberZHG/torch-multi-head-attention/blob/master/torch_multi_head_attention/multi_head_attention.py
class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None, train=False, dropout=0.0):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        attention = F.dropout(attention, dropout, train)
        return attention.matmul(value)

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 head_num,
                 bias=True,
                 activation=torch.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError(
                '`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, out_features, bias)
        self.linear_k = nn.Linear(in_features, out_features, bias)
        self.linear_v = nn.Linear(in_features, out_features, bias)
        self.linear_o = nn.Linear(out_features, out_features, bias)

    def forward(self, q, k, v, mask=None, train=False):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)

        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask, train)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=300):
        super(PositionalEncoding, self).__init__()
        self.dropout = dropout

        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)

    def forward(self, x, train=False):
        x = x + self.pe[:x.size(0)]
        x = F.dropout(x, self.dropout, train)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, args, in_features, out_features):
        super(TransformerLayer, self).__init__()
        self.args = args
        self.in_features = in_features
        self.out_features = out_features
        self.linear_1 = nn.Linear(in_features, out_features)
        self.pe = PositionalEncoding(d_model=out_features)
        self.lnorm_1 = nn.LayerNorm(out_features)
        self.lnorm_2 = nn.LayerNorm(out_features)
        self.attn = MultiHeadAttention(out_features, out_features, 1, activation=torch.tanh)
        self.linear_2 = nn.Linear(out_features, out_features)

    def forward(self, x, train=False):
        v = torch.tanh(self.linear_1(x))
        v = self.pe(v.permute(1, 0, 2), train=train).permute(1, 0, 2)
        h = self.attn(v, v, v, MultiHeadAttention.gen_history_mask(v), train=train)
        h = F.dropout(self.lnorm_1(h + v), 0.0, train)
        h_out = torch.tanh(self.linear_2(h))
        h_out = F.dropout(self.lnorm_2(h_out + h), 0.0, train)

        return h_out
