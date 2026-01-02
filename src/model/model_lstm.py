import torch
import torch.nn as nn
import sys
import os

# Add src to path if needed (though running as module is preferred)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.config as cfg

class BiLSTM(nn.Module):
    """
    双向 LSTM 网络模型，用于手语识别序列分类。
    """
    def __init__(self, input_size=cfg.INPUT_SIZE, hidden_size=cfg.HIDDEN_SIZE, num_layers=cfg.NUM_LAYERS, num_classes=cfg.NUM_CLASSES, dropout=cfg.DROPOUT):
        """
        初始化模型层。
        
        Args:
            input_size (int): 输入特征维度 (每帧的关键点坐标数)。
            hidden_size (int): LSTM 隐藏层状态的维度。
            num_layers (int): LSTM 的层数。
            num_classes (int): 分类任务的类别总数。
            dropout (float): Dropout 概率，用于防止过拟合。
        """
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = cfg.BIDIRECTIONAL
        
        # 定义 LSTM 层
        # batch_first=True 表示输入数据的维度顺序为 (batch, seq, feature)
        # bidirectional=True 使用双向 LSTM，能够同时利用过去和未来的上下文信息
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, 
                            bidirectional=self.bidirectional, 
                            dropout=dropout if num_layers > 1 else 0)
        
        # 全连接层输入维度：如果是双向 LSTM，则为 hidden_size * 2
        fc_input_dim = hidden_size * 2 if self.bidirectional else hidden_size
        
        # Dropout 层
        self.dropout_fc = nn.Dropout(dropout)
        
        # 输出层：将 LSTM 的输出映射到类别空间
        self.fc = nn.Linear(fc_input_dim, num_classes)
        
    def forward(self, x, lengths):
        """
        前向传播计算。
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, input_size)
            lengths (torch.Tensor): 每个样本的真实长度，形状为 (batch_size,)
            
        Returns:
            torch.Tensor: 输出得分，形状为 (batch_size, num_classes)
        """
        # x shape: (batch_size, seq_len, input_size)
        
        # 1. Pack the sequence
        # pack_padded_sequence 要求 lengths 必须在 CPU 上
        # enforce_sorted=False 允许 batch 中的长度是乱序的
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # 2. LSTM Forward
        # out 也是一个 PackedSequence 对象
        # (h_n, c_n) 是最后一个*有效*时间步的隐藏状态，这正是我们想要的！
        packed_out, (h_n, c_n) = self.lstm(packed_input)
        
        # 3. 提取特征
        # 如果是双向 LSTM，h_n 的形状是 (num_layers * 2, batch, hidden_size)
        # 我们需要把最后两个方向的状态拼接起来
        
        if self.bidirectional:
            # 取最后一层的正向和反向 hidden state
            # h_nview: (num_layers, num_directions, batch, hidden_size)
            h_n_view = h_n.view(self.num_layers, 2, x.size(0), self.hidden_size)
            
            # 获取最后一层 (index = -1)
            # forward_state: (batch, hidden_size)
            forward_state = h_n_view[-1, 0, :, :]
            # backward_state: (batch, hidden_size)
            backward_state = h_n_view[-1, 1, :, :]
            
            # 拼接: (batch, hidden_size * 2)
            out_feature = torch.cat((forward_state, backward_state), dim=1)
        else:
            # 单向: 直接取最后一层的 hidden state
            # (batch, hidden_size)
            out_feature = h_n[-1, :, :]
        
        # 4. Dropout & FC
        out = self.dropout_fc(out_feature)
        out = self.fc(out)
        
        return out

if __name__ == "__main__":
    # Test the model
    model = BiLSTM()
    print(model)
    dummy_input = torch.randn(32, 110, 270)
    output = model(dummy_input)
    print("Output shape:", output.shape)
