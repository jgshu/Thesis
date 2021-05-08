import torch
from torch import nn


class RNN_LoadForecastser(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, do, device='cpu'):
        super(RNN_LoadForecastser, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = do
        self.device = device
        # batch_first=True causes input/output tensors to be of shape
        # x: (batch_dim, seq_dim, feature_dim)
        # x: (samples, timesteps, features)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Readout layer
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def init_hidden_state(self, bs):
        return torch.zeros(self.num_layers, bs, self.hidden_dim).requires_grad_()

    def forward(self, x):
        bs = x.shape[0]
        h0 = self.init_hidden_state(bs).to(self.device)
        c0 = self.init_hidden_state(bs).to(self.device)
        # out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out

# class RNN(nn.Module):
#     def __init__(self):
#         super(RNN, self).__init__()
#         self.rnn = nn.RNN(
#             input_size=INPUT_SIZE,
#             hidden_size=HIDDEN_SIZE,
#             num_layers=1,
#             batch_first=False
#         )
#         self.out = nn.Linear(32, 1)
#
#     def forward(self, x, h_state):
#         # x: (batch, time_step, input_size)
#         # h_state: (n_layers, batch, hidden_size)
#         # r_out: (batch, time_step, output_size)
#         r_out, h_state = self.rnn(x, h_state)
#         outs = []
#         for time_step in range(r_out.size(1)):
#             outs.append(self.out(r_out[:, time_step, :]))
#         return torch.stack(outs, dim=1), h_state


