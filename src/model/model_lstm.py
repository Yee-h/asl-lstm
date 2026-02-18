import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add src to path if needed (though running as module is preferred)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.config as cfg


class Attention(nn.Module):
    """
    加性注意力机制 (Additive Attention / Bahdanau Attention)。
    用于对 LSTM 各时间步的输出进行加权求和，聚焦于关键帧。
    """

    def __init__(self, hidden_dim, attention_dim=cfg.MODEL.attention_dim):
        """
        初始化注意力层。

        Args:
            hidden_dim (int): LSTM 输出的隐藏维度 (双向则为 hidden_size * 2)。
            attention_dim (int): 注意力计算的中间维度。
        """
        super(Attention, self).__init__()
        # 将 LSTM 输出映射到注意力空间
        self.attention_fc = nn.Linear(hidden_dim, attention_dim)
        # 注意力上下文向量，用于计算每个时间步的重要性分数
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, lstm_output, mask=None):
        """
        计算注意力加权的上下文向量。

        Args:
            lstm_output (torch.Tensor): LSTM 输出，形状为 (batch, seq_len, hidden_dim)
            mask (torch.Tensor, optional): 掩码张量，形状为 (batch, seq_len)，
                                           True 表示有效位置，False 表示填充位置

        Returns:
            context (torch.Tensor): 加权上下文向量，形状为 (batch, hidden_dim)
            attention_weights (torch.Tensor): 注意力权重，形状为 (batch, seq_len)
        """
        # 1. 计算注意力能量分数
        # (batch, seq_len, hidden_dim) -> (batch, seq_len, attention_dim)
        energy = torch.tanh(self.attention_fc(lstm_output))

        # 2. 计算注意力分数 (未归一化)
        # (batch, seq_len, attention_dim) -> (batch, seq_len, 1) -> (batch, seq_len)
        attention_scores = self.context_vector(energy).squeeze(-1)

        # 3. 应用掩码：将填充位置的分数设为负无穷，softmax 后接近 0
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

        # 4. Softmax 归一化得到注意力权重
        # (batch, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)

        # 5. 加权求和得到上下文向量
        # (batch, seq_len, 1) * (batch, seq_len, hidden_dim) -> (batch, hidden_dim)
        context = torch.sum(attention_weights.unsqueeze(-1) * lstm_output, dim=1)

        return context, attention_weights


class BiLSTMAttention(nn.Module):
    """
    双向 LSTM + Attention 网络模型，用于手语识别序列分类。
    通过注意力机制自动学习各帧的重要性权重，聚焦于关键动作帧。
    """

    def __init__(
        self,
        input_size=cfg.SEQUENCE.input_size,
        hidden_size=cfg.MODEL.hidden_size,
        num_layers=cfg.MODEL.num_layers,
        num_classes=cfg.SEQUENCE.num_classes,
        dropout=cfg.MODEL.dropout,
        attention_dim=cfg.MODEL.attention_dim,
    ):
        """
        初始化模型层。

        Args:
            input_size (int): 输入特征维度 (每帧的关键点坐标数)。
            hidden_size (int): LSTM 隐藏层状态的维度。
            num_layers (int): LSTM 的层数。
            num_classes (int): 分类任务的类别总数。
            dropout (float): Dropout 概率，用于防止过拟合。
            attention_dim (int): 注意力机制的中间维度。
        """
        super(BiLSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = cfg.MODEL.bidirectional

        # 定义 LSTM 层
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        # LSTM 输出维度：双向则为 hidden_size * 2
        lstm_output_dim = hidden_size * 2 if self.bidirectional else hidden_size

        # 注意力层
        self.attention = Attention(lstm_output_dim, attention_dim)

        self.dropout_fc = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x, lengths):
        """
        前向传播计算。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, input_size)
            lengths (torch.Tensor): 每个样本的真实长度，形状为 (batch_size,)

        Returns:
            torch.Tensor: 输出得分，形状为 (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.size()

        # 1. Pack the sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # 2. LSTM Forward
        packed_out, (h_n, c_n) = self.lstm(packed_input)

        # 3. Unpack the sequence
        # lstm_output: (batch, seq_len, hidden_dim)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=seq_len
        )

        # 4. 创建掩码：标记有效位置 (非填充)
        # mask: (batch, seq_len), True 表示有效位置
        mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1).to(
            x.device
        )

        # 5. Attention 计算
        # context: (batch, hidden_dim)
        context, attention_weights = self.attention(lstm_output, mask)

        # 6. 分类
        out = self.dropout_fc(context)
        out = self.fc(out)

        return out

    def forward_with_attention(self, x, lengths):
        """
        前向传播并返回注意力权重，用于可视化分析。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, input_size)
            lengths (torch.Tensor): 每个样本的真实长度，形状为 (batch_size,)

        Returns:
            output (torch.Tensor): 输出得分，形状为 (batch_size, num_classes)
            attention_weights (torch.Tensor): 注意力权重，形状为 (batch_size, seq_len)
        """
        batch_size, seq_len, _ = x.size()

        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_out, (h_n, c_n) = self.lstm(packed_input)

        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=seq_len
        )

        mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1).to(
            x.device
        )

        context, attention_weights = self.attention(lstm_output, mask)

        out = self.dropout_fc(context)
        out = self.fc(out)

        return out, attention_weights


# 保留原始 BiLSTM 类以便对比或回退
class BiLSTM(nn.Module):
    """
    原始双向 LSTM 网络模型 (无 Attention)，用于对比实验。
    """

    def __init__(
        self,
        input_size=cfg.SEQUENCE.input_size,
        hidden_size=cfg.MODEL.hidden_size,
        num_layers=cfg.MODEL.num_layers,
        num_classes=cfg.SEQUENCE.num_classes,
        dropout=cfg.MODEL.dropout,
    ):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = cfg.MODEL.bidirectional

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        fc_input_dim = hidden_size * 2 if self.bidirectional else hidden_size
        self.dropout_fc = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x, lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_out, (h_n, c_n) = self.lstm(packed_input)

        if self.bidirectional:
            h_n_view = h_n.view(self.num_layers, 2, x.size(0), self.hidden_size)
            forward_state = h_n_view[-1, 0, :, :]
            backward_state = h_n_view[-1, 1, :, :]
            out_feature = torch.cat((forward_state, backward_state), dim=1)
        else:
            out_feature = h_n[-1, :, :]

        out = self.dropout_fc(out_feature)
        out = self.fc(out)

        return out


def build_dummy_batch_for_smoke(
    batch_size: int = 32,
    seq_len: int = 110,
    input_size: int | None = None,
):
    """构造与当前配置一致的随机输入，用于模型前向自测。"""
    feature_dim = cfg.SEQUENCE.input_size if input_size is None else int(input_size)
    seq_len = max(1, int(seq_len))
    dummy_input = torch.randn(batch_size, seq_len, feature_dim)
    dummy_lengths = torch.randint(1, seq_len + 1, (batch_size,))
    return dummy_input, dummy_lengths


def get_model(use_attention=cfg.MODEL.use_attention):
    """
    根据配置返回相应的模型实例。

    Args:
        use_attention (bool): 是否使用 Attention 机制。

    Returns:
        nn.Module: BiLSTMAttention 或 BiLSTM 模型实例。
    """
    if use_attention:
        return BiLSTMAttention()
    else:
        return BiLSTM()


if __name__ == "__main__":
    # 测试 BiLSTM + Attention 模型
    print("=" * 50)
    print("测试 BiLSTMAttention 模型")
    print("=" * 50)

    model = BiLSTMAttention()
    print(model)
    print(f"\n模型参数总量: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟输入（维度严格对齐 cfg.SEQUENCE.input_size）
    dummy_input, dummy_lengths = build_dummy_batch_for_smoke()

    # 前向传播
    output = model(dummy_input, dummy_lengths)
    print(f"\n输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")

    # 测试带注意力权重的前向传播
    output, attn_weights = model.forward_with_attention(dummy_input, dummy_lengths)
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"注意力权重示例 (第一个样本前10帧): {attn_weights[0, :10].detach().numpy()}")
