import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
	"""Sine-cosine positional encoder (moved from project root model.py)."""

	def __init__(self, d_input: int, n_freqs: int, log_space: bool = False):
		super().__init__()
		self.d_input = d_input
		self.n_freqs = n_freqs
		self.log_space = log_space
		self.d_output = d_input * (1 + 2 * self.n_freqs)
		self.embed_fns = [lambda x: x]

		if self.log_space:
			freq_bands = 2.0 ** torch.linspace(0.0, self.n_freqs - 1, self.n_freqs)
		else:
			freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** (self.n_freqs - 1), self.n_freqs)

		for freq in freq_bands:
			self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
			self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

	def forward(self, x) -> torch.Tensor:
		return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)


class TemporalLSTMEncoder(nn.Module):
	"""通用的时序编码器：
	- 默认返回最后一层的最后时间步隐状态 (Batch, Hidden_Dim)
	- 如需完整输出，可设置 return_all=True，得到 (output, (h_n, c_n)) 与原生 LSTM 对齐。
	forward 输入形状: (Batch, Seq_Len, Input_Dim)
	forward 输出形状: (Batch, Hidden_Dim)
	"""
	def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.0, batch_first=True):
		super().__init__()
		self.lstm = nn.LSTM(
			input_size=input_dim,
			hidden_size=hidden_dim,
			num_layers=num_layers,
			batch_first=batch_first,
			dropout=dropout if num_layers > 1 else 0.0,
		)

	def forward(self, seq, return_all: bool = False):
		output, (h_n, c_n) = self.lstm(seq)
		if return_all:
			return output, (h_n, c_n)
		return h_n[-1]


class ActuatorMLPEncoder(nn.Module):
	"""将驱动/执行器信号编码为紧凑特征。"""
	def __init__(self, input_dim, feat_dim=32):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, feat_dim),
			nn.ReLU(),
			nn.Linear(feat_dim, feat_dim),
			nn.ReLU(),
		)

	def forward(self, x):
		return self.net(x)


class MLPDecoder(nn.Module):
	"""通用的 MLP 解码器，用于 (位置编码 + 状态 [+ 动作]) 到输出。"""
	def __init__(self, input_dim, d_filter=128, output_size=2):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, d_filter * 2),
			nn.ReLU(),
			nn.Linear(d_filter * 2, d_filter * 2),
			nn.ReLU(),
			nn.Linear(d_filter * 2, d_filter),
			nn.ReLU(),
			nn.Linear(d_filter, d_filter // 2),
			nn.ReLU(),
			nn.Linear(d_filter // 2, output_size),
		)

	def forward(self, x):
		return self.net(x)


class VisualConvDecoder64(nn.Module):
	"""视觉序列解码器：将线性隐向量映射为 (3, 64, 64) 图像。"""
	def __init__(self, linear_in_dim):
		super().__init__()
		self.fc = nn.Linear(linear_in_dim, 256 * 4 * 4)
		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
			nn.Sigmoid(),
		)

	def forward(self, z):
		z = self.fc(z)
		z = z.view(z.size(0), 256, 4, 4)
		return self.deconv(z)


__all__ = [
	"PositionalEncoder",
	"TemporalLSTMEncoder",
	"ActuatorMLPEncoder",
	"MLPDecoder",
	"VisualConvDecoder64",
]
