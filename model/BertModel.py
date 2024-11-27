from torch.utils.data import Dataset
import torch
import torch.nn as nn

import random

import math
from BertModel.BertEncoder import BertEncoder


class BDataset(Dataset):
    def __init__(self, all_text1, all_text2, all_label, max_len, word_2_index):
        self.all_text1 = all_text1
        self.all_text2 = all_text2
        self.all_label = all_label
        self.max_len = max_len
        self.word_2_index = word_2_index

    def __getitem__(self, index):
        text1 = self.all_text1[index]
        text2 = self.all_text2[index]

        lable = self.all_label[index]
        unk_idx = self.word_2_index["[UNK]"]
        text1_idx = [self.word_2_index.get(i, unk_idx) for i in text1][:62]
        text2_idx = [self.word_2_index.get(i, unk_idx) for i in text2][:63]

        # 验证 text1_idx 中的索引是否越界，如果越界，替换为 [UNK] 索引
        for i, idx in enumerate(text1_idx):
            if idx >= len(self.word_2_index):
                print(f"Index out of range in text1 at position {i}: {idx}, replacing with [UNK]")
                text1_idx[i] = unk_idx  # 替换为 [UNK] 的索引

        # 验证 text2_idx 中的索引是否越界，如果越界，替换为 [UNK] 索引
        for i, idx in enumerate(text2_idx):
            if idx >= len(self.word_2_index):
                print(f"Index out of range in text2 at position {i}: {idx}, replacing with [UNK]")
                text2_idx[i] = unk_idx  # 替换为 [UNK] 的索引

        mask_val = [0] * self.max_len

        text_idx = [self.word_2_index["[CLS]"]] + text1_idx + [self.word_2_index["[SEP]"]] + text2_idx + [
            self.word_2_index["[SEP]"]]
        seg_idx = [0] + [0] * len(text1_idx) + [0] + [1] * len(text2_idx) + [1] + [2] * (self.max_len - len(text_idx))

        for i, v in enumerate(text_idx):
            if v in [self.word_2_index["[CLS]"], self.word_2_index["[SEP]"], self.word_2_index["[UNK]"]]:
                continue

            if random.random() < 0.15:
                r = random.random()
                if r < 0.8:
                    text_idx[i] = self.word_2_index["[MASK]"]

                    mask_val[i] = v

                elif r > 0.9:
                    other_idx = random.randint(6, len(self.word_2_index) - 1)
                    text_idx[i] = other_idx
                    mask_val[i] = v

        text_idx = text_idx + [self.word_2_index["[PAD]"]] * (self.max_len - len(text_idx))

        return torch.tensor(text_idx), torch.tensor(lable), torch.tensor(mask_val), torch.tensor(seg_idx)

    def __len__(self):
        return len(self.all_label)


class BertEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.word_embeddings.weight.requires_grad = True

        self.position_embeddings = nn.Embedding(config["max_position_embeddings"], config["hidden_size"])
        self.position_embeddings.weight.requires_grad = True

        self.token_type_embeddings = nn.Embedding(config["type_vocab_size"], config["hidden_size"])
        self.token_type_embeddings.weight.requires_grad = True

        self.LayerNorm = nn.LayerNorm(config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, batch_idx, batch_seg_idx):
        word_emb = self.word_embeddings(batch_idx)

        pos_idx = torch.arange(0, self.position_embeddings.weight.data.shape[0], device=batch_idx.device)
        pos_idx = pos_idx.repeat(batch_idx.shape[0], 1)
        pos_emb = self.position_embeddings(pos_idx)

        token_emb = self.token_type_embeddings(batch_seg_idx)

        emb = word_emb + pos_emb + token_emb

        emb = self.LayerNorm(emb)
        emb = self.dropout(emb)

        return emb


class Add_Norm(nn.Module):
    def __init__(self, embedding_num):
        super().__init__()
        self.Add = nn.Linear(embedding_num, embedding_num)
        self.Norm = nn.LayerNorm(embedding_num)

    def forward(self, x):  # B * Layer * emb
        add_x = self.Add(x)
        norm_x = self.Norm(add_x)
        return norm_x


class Feed_Forward(nn.Module):
    def __init__(self, embedding_num, feed_num):
        super(Feed_Forward, self).__init__()
        self.l1 = nn.Linear(embedding_num, feed_num)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(feed_num, embedding_num)

    def forward(self, x):
        l1_x = self.l1(x)
        r_x = self.relu(l1_x)
        l2_x = self.l2(r_x)
        return l2_x


class M_Self_Attention(nn.Module):
    def __init__(self, embedding_num, n_heads):
        super(M_Self_Attention, self).__init__()
        self.W_Q = nn.Linear(embedding_num, embedding_num, bias=False)
        self.W_K = nn.Linear(embedding_num, embedding_num, bias=False)
        self.W_V = nn.Linear(embedding_num, embedding_num, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.n_heads = n_heads

    def forward(self, x):
        b, l, n = x.shape
        x_ = x.reshape(b, self.n_heads, -1, n)
        Q = self.W_Q(x_)  # 查询
        K = self.W_K(x_)  # 关键

        V = self.W_V(x_)  # 值

        s = (Q @ (K.transpose(-1, -2))) / (math.sqrt(x.shape[-1] / 1.0))
        score = self.softmax(s)
        r = score @ V
        r = r.reshape(b, l, n)
        return r


class Block(nn.Module):
    def __init__(self, embeding_dim, n_heads, feed_num):
        super(Block, self).__init__()
        self.att_layer = M_Self_Attention(embeding_dim, n_heads)
        self.add_norm1 = Add_Norm(embeding_dim)

        self.feed_forward = Feed_Forward(embeding_dim, feed_num)
        self.add_norm2 = Add_Norm(embeding_dim)
        self.n = 100

    def forward(self, x):
        att_x = self.att_layer(x)
        adn_x1 = self.add_norm1(att_x)

        adn_x1 = x + adn_x1  # 残差网络

        ff_x = self.feed_forward(adn_x1)
        adn_x2 = self.add_norm2(ff_x)

        adn_x2 = adn_x1 + adn_x2  # 残差网络

        return adn_x2


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]

        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = BertEmbedding(config)
        # self.bert_layer = nn.Linear(config["hidden_size"],config["hidden_size"]) #

        self.bert_layer = nn.Sequential(
            *[BertEncoder(config["hidden_size"], config["feed_num"], config["head_num"]) for i in
              range(config["layer_num"])])

        self.pool = BertPooler(config)

    def forward(self, batch_idx, batch_seg_idx):
        x = self.embedding(batch_idx, batch_seg_idx)

        x = self.bert_layer(x)

        bertout2 = self.pool(x)

        return x, bertout2


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bert = BertModel(config)

        self.cls_mask = nn.Linear(config["hidden_size"], config["vocab_size"])
        self.cls_next = nn.Linear(config["hidden_size"], 2)

        self.loss_fun_mask = nn.CrossEntropyLoss(ignore_index=0)
        self.loss_fun_next = nn.CrossEntropyLoss()

    def forward(self, batch_idx, batch_seg_idx, batch_mask_val=None, batch_label=None):
        bertout1, bertout2 = self.bert(batch_idx, batch_seg_idx)

        pre_mask = self.cls_mask(bertout1)
        pre_next = self.cls_next(bertout2)

        if batch_mask_val is not None and batch_label is not None:
            loss_mask = self.loss_fun_mask(pre_mask.reshape(-1, pre_mask.shape[-1]), batch_mask_val.reshape(-1))
            loss_next = self.loss_fun_next(pre_next, batch_label)

            loss = loss_mask + loss_next

            return loss
        else:
            return torch.argmax(pre_mask, dim=-1), torch.argmax(pre_next, dim=-1)
