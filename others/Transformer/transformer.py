import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Transformer(nn.Module):
    def __init__(self, inputs_vocab, outputs_vocab, d_model, N, heads, dropout):
        super(Transformer, self).__init__()
        self.encoder = Encoder(inputs_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(outputs_vocab, d_model, N, heads, dropout)
        self.outproject = nn.Linear(d_model, outputs_vocab)
    
    def forward(self, inputs, shift_outputs, inputs_mask, outputs_mask):
        e_outputs = self.encoder(inputs, inputs_mask)
        d_output = self.decoder(shift_outputs, e_outputs,inputs_mask, outputs_mask)
        output = self.outproject(d_output)
        return output

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos][i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos][i + 1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self, x):
        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)

        seq_len = x.size(1)
        x = (x + torch.tensor(self.pe[:,:seq_len], requires_grad=False)).cuda()
        return x




class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super(Encoder, self).__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = NormLayer(d_model)
        
    def forward(self, inputs, mask):
        x = self.embed(inputs)
        x = self.pos(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model, dropout)
        self.norm = NormLayer(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x = x + self.attn(x, x, x, mask)
        x = self.norm(x)
        x = x + self.ff(x)
        x = self.norm(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # 上下文单词所对应的权重得分，形状是 seq_len, d_model × d_model, seq_len = seq_len, seq_len
        # 掩盖掉那些为了填补长度增加的单元，使其通过 softmax 计算后为 0
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        
        # 进行线性操作划分为 h 个头， batch_size, seq_len, d_model -> batch_size, seq_len, h, d_k
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)  
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # 矩阵转置  batch_size, seq_len, h, d_k -> batch_size, h, seq_len, d_k
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # 计算 attention
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        
        # 连接多个头并输入到最后的线性层 (bs, h, seq_len, d_k) 转换为 (bs, seq_len, h, d_k)
        # .contiguous() 用于确保内存的连续性，方便后续的操作。
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__(FeedForward, self)
        d_ff = d_model * 4
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)




class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.norm_1 = NormLayer(d_model)
        self.norm_2 = NormLayer(d_model)
        self.norm_3 = NormLayer(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))  # q是x2，k是encoder的输出，v也是encoder的输出
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super(Decoder, self).__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = NormLayer(d_model)
    
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)   # 用的是trg
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


def get_clones(module, N):
    """ 
    生成 N 个相同的模块。
    
    Parameters:
    - module: nn.Module，想要克隆的模块
    - N: 克隆的数量

    ModuleList 是 PyTorch 中的一个容器，可以方便地管理多个子模块。
    """
    return nn.ModuleList([module for _ in range(N)])