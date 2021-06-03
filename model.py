import torch
import torch.nn as nn
import torch.nn.functional as F


###############
# SLSTM
###############
class SLSTM(nn.Module):
    def __init__(self, args, device='cpu'):
        super(SLSTM, self).__init__()
        self.input_dim = args.n_features
        self.output_dim = args.out_features
        self.hidden_dim = args.n_hidden
        self.seq_len = args.seq_len
        self.num_layers = args.n_layers
        self.drop_out = args.do
        self.batch_size = args.batch_size
        self.device = device

        # x: (batch_dim, seq_dim, feature_dim) or (samples, timesteps, features) when considering batch_first=True
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.drop_out,
            batch_first=True,
        )

        # self.linear = nn.Linear(in_features=self.hidden_dim * self.seq_len, out_features=self.output_dim)
        self.reg = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)
        return (hidden_state, cell_state)

    def forward(self, x):
        batch_size = x.size(0)
        h0, c0 = self.init_hidden(batch_size)
        lstm_out, (hn, cn) = self.lstm(
            x,
            (h0.detach(), c0.detach())
        )
        last_time_step = lstm_out[:, -1].view(-1, self.hidden_dim * 2)
        out = self.reg(last_time_step)
        return out


###############
# BiLSTM
###############
class BiLSTM(nn.Module):
    def __init__(self, args, device='cpu'):
        super(BiLSTM, self).__init__()
        self.input_dim = args.n_features
        self.output_dim = args.out_features
        self.hidden_dim = args.n_hidden
        self.seq_len = args.seq_len
        self.num_layers = args.n_layers
        self.drop_out = args.do
        self.batch_size = args.batch_size
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
        # self.linear = nn.Linear(in_features=self.hidden_dim * self.seq_len * 2, out_features=self.output_dim)
        self.reg = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(self.hidden_dim * 2, self.output_dim)
        )

    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).requires_grad_().to(self.device)
        cell_state = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).requires_grad_().to(self.device)
        return (hidden_state, cell_state)

    def forward(self, x):
        batch_size = x.size(0)
        h0, c0 = self.init_hidden(batch_size)
        lstm_out, (hn, cn) = self.lstm(
            x,
            (h0.detach(), c0.detach())
        )
        # x = np.array([[[],[],[a, b]], [[],[],[c, d]]])
        # x.shape: (2, 3, 2)
        # x[:, -1] = x[:, -1, :] = [[a, b],[c, d]]
        # x.shape: (2, 2)
        # lstm_out.shape: (batch_size, seq_len, hidden_dim * num_directions)
        # lstm_out[:, -1].shape: (batch_size, hidden_dim * num_directions)
        # 或者lstm_out[:, -1, :]
        # view(-1, self.hidden_dim * 2)可省略
        last_time_step = lstm_out[:, -1].view(-1, self.hidden_dim * 2)
        out = self.reg(last_time_step)
        # print(lstm_out.shape)
        # print(lstm_out[:, -1].shape)
        # print(last_time_step.shape)
        # print(out.shape)
        return out


###############
# LSTNet
###############
class LSTNet(nn.Module):
    def __init__(self, args, device='cpu'):
        super(BiLSTM, self).__init__()
        self.P = args.seq_len
        self.m = args.n_features
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip
        self.pt = (self.P - self.Ck) / self.skip
        self.hw = args.day_range
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args.do)
        self.device = device

        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh


    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.P, self.m);
        c = F.relu(self.conv1(c));
        c = self.dropout(c);
        c = torch.squeeze(c, 3);

        # RNN
        r = c.permute(2, 0, 1).contiguous();
        _, r = self.GRU1(r);
        r = self.dropout(torch.squeeze(r, 0));

        # skip-rnn

        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous();
            s = s.view(batch_size, self.hidC, self.pt, self.skip);
            s = s.permute(2, 0, 3, 1).contiguous();
            s = s.view(self.pt, batch_size * self.skip, self.hidC);
            _, s = self.GRUskip(s);
            s = s.view(batch_size, self.skip * self.hidS);
            s = self.dropout(s);
            r = torch.cat((r, s), 1);

        res = self.linear1(r);

        # highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :];
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw);
            z = self.highway(z);
            z = z.view(-1, self.m);
            res = res + z;

        if (self.output):
            res = self.output(res);
        return res;