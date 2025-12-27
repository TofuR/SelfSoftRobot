import torch
from torch import nn
from .layers import PositionalEncoder


class FBV_SM(nn.Module):
	"""FBV_SM base model (moved from project root model.py)."""

	def __init__(self, encoder=None, d_input: int = 5, d_filter: int = 128, output_size: int = 2):
		super(FBV_SM, self).__init__()

		self.d_input = d_input
		self.act = nn.functional.relu
		self.encoder = encoder

		if self.encoder is None:
			pos_encoder_d = 3
			cmd_encoder_d = d_input - 3
			self.feed_forward = nn.Sequential(
				nn.Linear(d_filter * 2 + d_input, d_filter),
				nn.ReLU(),
				nn.Linear(d_filter, d_filter // 4),
			)
		else:
			n_freqs = self.encoder.n_freqs
			pos_encoder_d = (n_freqs * 2 + 1) * 3
			cmd_encoder_d = (n_freqs * 2 + 1) * (d_input - 3)
			self.feed_forward = nn.Sequential(
				nn.Linear(d_filter * 2, d_filter),
				nn.ReLU(),
				nn.Linear(d_filter, d_filter // 4),
			)

		self.pos_encoder = nn.Sequential(
			nn.Linear(pos_encoder_d, d_filter),
			nn.ReLU(),
			nn.Linear(d_filter, d_filter),
		)

		self.cmd_encoder = nn.Sequential(
			nn.Linear(cmd_encoder_d, d_filter),
			nn.ReLU(),
			nn.Linear(d_filter, d_filter),
		)

		self.output = nn.Linear(d_filter // 4, output_size)

		with torch.no_grad():
			self.output.bias[1] = -5.0

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if self.encoder is not None:
			x_pos = self.encoder(x[:, :3])
			x_cmd = self.encoder(x[:, 3:])
			x_pos = self.pos_encoder(x_pos)
			x_cmd = self.cmd_encoder(x_cmd)
			x = self.feed_forward(torch.cat((x_pos, x_cmd), dim=1))
		else:
			x_pos = self.pos_encoder(x[:, :3])
			x_cmd = self.cmd_encoder(x[:, 3:])
			x = self.feed_forward(torch.cat((x_pos, x_cmd, x), dim=1))

		return self.output(x)


__all__ = ["FBV_SM"]
