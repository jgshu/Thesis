import torch
from torch import nn


###############
# SLSTM
###############
class SLSTM(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers, out_features, do=0.5, device='cpu'):
        super(SLSTM, self).__init__()
        self.input_dim = n_features
        self.output_dim = out_features
        self.hidden_dim = n_hidden
        self.seq_len = seq_len
        self.num_layers = n_layers
        self.drop_out = do
        self.device = device

        # x: (batch_dim, seq_dim, feature_dim) or (samples, timesteps, features) when considering batch_first=True
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.drop_out,
            batch_first=True,
        )

        self.linear = nn.Linear(in_features=self.hidden_dim * self.seq_len, out_features=self.output_dim)

    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)
        return (hidden_state, cell_state)

    def forward(self, x):
        batch_size, _, _ = x.shape
        self.hidden = self.init_hidden(batch_size)
        lstm_out, _ = self.lstm(
            x, self.hidden
        )
        x = lstm_out.contiguous().view(batch_size, -1)
        return self.linear(x)


###############
# BiLSTM
###############
class BiLSTM(nn.Module):

    def __init__(self, n_features, n_hidden, seq_len, n_layers, out_features, do=0.5, device='cpu'):
        super(BiLSTM, self).__init__()
        self.input_dim = n_features
        self.output_dim = out_features
        self.hidden_dim = n_hidden
        self.seq_len = seq_len
        self.num_layers = n_layers
        self.drop_out = do
        self.device = device

        # x: (batch_dim, seq_dim, feature_dim) or (samples, timesteps, features) when considering batch_first=True
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.drop_out,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Linear(in_features=self.hidden_dim * self.seq_len * 2, out_features=self.output_dim)

    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.num_layers*2, batch_size, self.hidden_dim).requires_grad_().to(self.device)
        cell_state = torch.zeros(self.num_layers*2, batch_size, self.hidden_dim).requires_grad_().to(self.device)
        return (hidden_state, cell_state)

    def forward(self, x):
        batch_size, _, _ = x.shape
        # self.hidden = self.init_hidden(batch_size)
        lstm_out, _ = self.lstm(
            x,
            # self.hidden
        )
        x = lstm_out.contiguous().view(batch_size, -1)
        return self.linear(x)