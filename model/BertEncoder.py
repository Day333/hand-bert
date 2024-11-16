import torch
import torch.nn as nn

class Feed_Forward(nn.Module):
    def __init__(self,hidden_size,feed_num):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size,feed_num)
        self.relu = nn.GELU()
        self.linear2 = nn.Linear(feed_num,hidden_size)

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x


class Add_Norm(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.ADD = nn.Linear(hidden_size,hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self,x):
        x = self.ADD(x)
        x = self.norm(x)
        return x


class Multi_Head_Att_old(nn.Module):
    def __init__(self,hidden_num,head_num):
        super().__init__()
        self.att = nn.Linear(hidden_num,hidden_num)
        self.head_num = head_num
        self.softmax = nn.Softmax(dim=-1)


    def forward(self,x):
        batch,len_,hidden = x.shape
        x = x.reshape(batch,self.head_num,-1,hidden)

        x_ = torch.mean(x,dim=-1)

        score = self.softmax(x_)

        x = score.reshape(batch,-1,1) * x.reshape(batch,len_,-1)
        return x


class Multi_Head_Att(nn.Module):
    def __init__(self,hidden_num,head_num):
        super().__init__()
        self.Q = nn.Linear(hidden_num,hidden_num)
        self.K = nn.Linear(hidden_num,hidden_num)
        self.V = nn.Linear(hidden_num,hidden_num)


        self.head_num = head_num
        self.softmax = nn.Softmax(dim=-1)


    def forward(self,x): # 10 * 128 * 768
        batch, len_, hidden = x.shape
        x = x.reshape(batch, self.head_num, -1, hidden)

        q = self.Q(x) # 10 * 128 * 768
        k = self.K(x) # 10 * 128 * 768
        v = self.V(x) # 10 * 128 * 768

        score = self.softmax(q @ k.transpose(-2,-1))

        x = score @ v

        x = x.reshape(batch, len_, hidden)
        return x


class BertEncoder(nn.Module):
    def __init__(self,hidden_size,feed_num,head_num):
        super().__init__()

        self.multi_head_att = Multi_Head_Att(hidden_size,head_num)  # 768 * 768
        self.add_norm1 = Add_Norm(hidden_size)
        self.feed_forward = Feed_Forward(hidden_size,feed_num)
        self.add_norm2 = Add_Norm(hidden_size)

    def forward(self,x): # x: batch * seq_len * embedding   100 * 128 * 768
        multi_head_out = self.multi_head_att(x) # multi_head_out : 100 * 128 * 768
        add_norm1_out = self.add_norm1(multi_head_out) # add_norm1_out: 100 * 128 * 768

        add_norm1_out = x + add_norm1_out # add_norm1_out :  100 * 128 * 768

        feed_forward_out = self.feed_forward(add_norm1_out)  # feed_forward_out :  100 * 128 * 768
        add_norm2_out = self.add_norm2(feed_forward_out)  # add_norm2_out: 100 * 128 * 768

        add_norm2_out = add_norm1_out + add_norm2_out

        return add_norm2_out



