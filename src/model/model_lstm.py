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
        
        # 输出层：将 LSTM 的输出映射到类别空间
        self.fc = nn.Linear(fc_input_dim, num_classes)
        
    def forward(self, x):
        """
        前向传播计算。
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: 输出得分，形状为 (batch_size, num_classes)
        """
        # x shape: (batch_size, seq_len, input_size)
        
        # LSTM 前向传播
        # out 形状: (batch, seq_len, num_directions * hidden_size)
        # 我们忽略了最终的隐藏状态 _ (h_n, c_n)
        out, _ = self.lstm(x)
        
        # 特征聚合策略：平均池化 (Mean Pooling)
        # 对时间维度 (dim=1) 求平均，这有助于捕获整个序列的信息，
        # 并且相比于取最后一帧，对填充 (Padding) 的零向量更鲁棒。
        out = torch.mean(out, dim=1)
        
        # 通过全连接层得到分类预测
        out = self.fc(out)
        return out

if __name__ == "__main__":
    # Test the model
    model = BiLSTM()
    print(model)
    dummy_input = torch.randn(32, 110, 270)
    output = model(dummy_input)
    print("Output shape:", output.shape)
