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
        self.hidden = self.init_hidden(batch_size)
        lstm_out, _ = self.lstm(
            x,
            self.hidden
        )
        x = lstm_out.contiguous().view(batch_size, -1)
        return self.linear(x)


###############
# Conv LSTM with Attention
###############
class ConvLSTM_with_Attion(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers, out_features, do=0.5, device='cpu'):
        super(ConvLSTM_with_Attion, self).__init__()
        self.input_dim = n_features
        self.output_dim = out_features
        self.hidden_dim = n_hidden
        self.seq_len = seq_len
        self.num_layers = n_layers
        self.drop_out = do
        self.device = device

        self.conv1 = nn.Sequential(
            nn.Conv
        )

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
        self.hidden = self.init_hidden(batch_size)
        lstm_out, _ = self.lstm(
            x,
            self.hidden
        )
        x = lstm_out.contiguous().view(batch_size, -1)
        return self.linear(x)


###############
# simpleLSTM
###############
class simpleLSTM(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers, out_features, do=0.5, device='cpu'):
        super(simpleLSTM, self).__init__()
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