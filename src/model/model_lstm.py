#实现核心的 BiLSTM，使用 pack_padded_sequence 忽略填充帧。

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class CSL_BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
        super(CSL_BiLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 1. 降维层 (FC): 225 -> 128
        # 减少参数量，提取紧凑特征
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. BiLSTM 层
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 3. 分类层
        # 输入维度: hidden_dim * 2 (双向)
        self.fc_out = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x, lengths):
        # x: (Batch, Max_Len, Feature_Dim)
        # lengths: (Batch,) 真实长度
        
        # 1. 降维
        x = self.fc_in(x)
        
        # 2. 打包 (Masking Padding)
        # enforce_sorted=False 允许数据不按长度排序
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # 3. LSTM 前向传播
        # self.lstm 输出: packed_output, (h_n, c_n)
        _, (h_n, _) = self.lstm(packed_x)
        
        # 4. 提取特征
        # h_n shape: (num_layers * 2, batch, hidden_dim)
        # 我们需要取最后一层 (last layer) 的 正向(forward) 和 反向(backward) 状态
        
        # view: [layers, directions, batch, hidden]
        h_n = h_n.view(self.num_layers, 2, x.size(0), self.hidden_dim)
        
        # 取最后一层: h_n[-1] -> shape (2, batch, hidden)
        # 拼接两个方向: forward (idx 0) + backward (idx 1)
        final_hidden = torch.cat((h_n[-1, 0], h_n[-1, 1]), dim=1) 
        # final_hidden shape: (Batch, hidden_dim * 2)
        
        # 5. 分类
        out = self.fc_out(final_hidden)
        
        return out