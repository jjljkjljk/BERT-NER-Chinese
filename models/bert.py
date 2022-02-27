import torch
import torch.nn as nn
from models.modules import PositionalEmbedding
from models.sublayers import MultiHeadAttention
from models.sublayers import FeedForwardLayer


class BertModel(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, n_heads,
                 max_seq_len, vocab_size, pad_id, dropout=0.1):
        super(BertModel, self).__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = PositionalEmbedding(d_model, max_seq_len)
        self.segment_emb = nn.Embedding(2, d_model, padding_idx=0)
        self.dropout_emb = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, input_ids, token_type_ids, mask, return_attn=False):
        # input_ids:[b_size, len]
        out = self.word_emb(input_ids) + self.pos_emb(input_ids) + self.segment_emb(token_type_ids)
        out = self.dropout_emb(out)
        attn_list = []
        for layer in self.layers:
            out, attn = layer(out, mask)
            if return_attn:
                attn_list.append(attn)

        out = (out, attn_list) if return_attn else out
        return out


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(n_heads, d_model, dropout)
        self.feed_forward = FeedForwardLayer(d_model, d_ff, dropout)

    def forward(self, inputs, mask):
        out, attn = self.attention(inputs, inputs, inputs, mask)
        out = self.feed_forward(out)

        return out, attn


if __name__ == '__main__':
    input_ids = torch.randint(0,100,(4,10))
    token_type_ids = torch.randint(0,1,(4,10))
    model = BertModel(n_layers=2, d_model=64, d_ff=256, n_heads=4,
                 max_seq_len=128, vocab_size=1000, pad_id=0, dropout=0.1)
    model(input_ids, token_type_ids)

    # b_size x len x dim
    # a = torch.LongTensor([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]])
    # print(a.size())
    # b_size, _,  dim = a.size()
    #
    # b = a.repeat(1, b_size, 1)
    # print(b)
    # b = b.view(b_size*b_size, -1, dim)
    # print(b.size())
    # print(b)
    #
    # c = a.repeat(b_size, 1, 1)
    # print(c)
    # print(c.size())

    # a = torch.LongTensor([1,2,3,4])
    # index = torch.LongTensor([3,1])
    # print(a.index_select(0, index))
    # # b = torch.max(a).tolist()
    # # print(b)
    # b = np.array([1,1,1,2,2,3])
    # res = np.argwhere(b == 2)
    # print(res)