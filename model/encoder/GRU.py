import torch
import torch.nn as nn

class GRU(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers):
		super(GRU, self).__init__()
		self.gru = nn.GRU(input_dim + 1, hidden_dim, num_layers, batch_first=True)

	def forward(self, t, x):
		batch_size, seq_len, num_features = x.shape
		mask = ~torch.isnan(x)
		x = x[mask]
		x = x.view(batch_size, int(x.shape[0] / (batch_size * num_features)), num_features)
		embedding, _ = self.gru(x)
		return embedding[:, -1, :]