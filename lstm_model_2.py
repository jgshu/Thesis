import torch
from torch import nn


class LoadForecastser(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers, out_features, do=0.5, device='cpu'):
        super(LoadForecastser, self).__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.device = device
        # x: (batch_dim, seq_dim, feature_dim) or (samples, timesteps, features)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=do
        )
        # Readout layer
        self.reg = nn.Sequential(
            nn.Linear(in_features=n_hidden, out_features=n_hidden),
            nn.Tanh(),
            nn.Linear(in_features=n_hidden, out_features=out_features)
        )
        self.hidden_cell = (torch.zeros(self.n_layers, self.seq_len, self.n_hidden).requires_grad_().to(device),
                            torch.zeros(self.n_layers, self.seq_len, self.n_hidden).requires_grad_().to(device)
                            )

    def forward(self, x):
        # lstm_out, self.hidden_cell = self.lstm(
        #     x.view(len(x), self.seq_len, -1),
        #     self.hidden_cell
        # )
        # lstm_out, self.hidden_cell = self.lstm(
        #     x.view(len(x), self.seq_len, -1),
        #     self.hidden_cell
        # )
        lstm_out, _= self.lstm(
            x.view(len(x), self.seq_len, -1),
        )
        last_time_step = lstm_out.view(
            self.seq_len,
            len(x),
            self.n_hidden
        )[-1]
        y_pred = self.reg(last_time_step)
        return y_pred

        # y, (h0, c0) = self.lstm(x)
        # y = y.view(-1, self.n_hidden)
        # y = self.reg(y)
        # y = y.view(self.seq_len, len(x), -1)
        # return y







