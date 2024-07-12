import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, t, x): # t could be used for delta-t LSTM
        embedding, _ = self.lstm(x)
        return embedding[:, -1, :]