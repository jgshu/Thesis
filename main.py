import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from lstm_model import RNN_LoadForecastser
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    n_predictions = 672
    n_next = 96
    batch_size = 128
    input_dim = 30
    hidden_dim = 32
    num_layers = 2
    output_dim = 96
    do = 0.05
    lr = 0.01
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    path = './output/train_test'
    train_x = np.load(path + '/train_x.npy')
    train_y = np.load(path + '/train_y.npy')
    test_x = np.load(path + '/test_x.npy')
    test_y = np.load(path + '/test_y.npy')

    train_x = torch.from_numpy(train_x).type(torch.Tensor)
    train_y = torch.from_numpy(train_y).type(torch.Tensor)
    train_x
    test_x = torch.from_numpy(test_x).type(torch.Tensor)
    test_y = torch.from_numpy(test_y).type(torch.Tensor)

    n_steps = n_predictions - 1
    num_epochs = 100

    train = torch.utils.data.TensorDataset(train_x, train_y)
    test = torch.utils.data.TensorDataset(test_x, test_y)

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=batch_size,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=batch_size,
                                              shuffle=False)

    model = RNN_LoadForecastser(input_dim=input_dim, hidden_dim=hidden_dim,
                                num_layers=num_layers, output_dim=output_dim,
                                do=do, device=device).to(device)

    loss_fn = nn.MSELoss().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    print(model)
    print(len(list(model.parameters())))

    for i in range(len(list(model.parameters()))):
        print(list(model.parameters())[i].size())

    hist = np.zeros(num_epochs)
    # Number of steps to unroll
    seq_dim = n_predictions - 1
    for t in range(num_epochs):
        for i, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            print(i, data.shape, label.shape)
            # Initialise hidden state
            # Don't do this if you want your LSTM to be stateful
            # model.hidden = model.init_hidden()

            # Forward pass
            train_y_pred = model(data).to(device)

            loss = loss_fn(train_y_pred, label)
            if t % 10 == 0 and t != 0:
                print("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()

            # Zero out gradient, else they will accumulate between epochs
            optimiser.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimiser.step()

    torch.save(model, 'model.pkl')
    # model = torch.load('./model.pkl')
    # make predictions
    test_x = test_x.to(device)
    test_y_pred = model(test_x).to(device)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    # invert predictions
    # train_y_pred = scaler.inverse_transform(train_y_pred.detach().numpy())
    # test_y = scaler.inverse_transform(train_y.detach().numpy())
    # test_y_pred = scaler.inverse_transform(test_y_pred.detach().cpu().numpy())
    # print(test_y_pred.shape)
    # print(reshape_y_hat(test_y_pred.cpu().detach().numpy(), 1)[1, -1, :])
    # print(reshape_y_hat(test_y.cpu().numpy(), 1).shape)
    # print(reshape_y_hat(test_y.cpu().numpy(), 1))
    # print(test_y.shape)
    # test_y = scaler.inverse_transform(test_y.detach().numpy())

    # calculate root mean squared error
    # trainScore = math.sqrt(mean_squared_error(train_y[:, 0], train_y_pred[:, 0]))
    # print('Train Score: %.2f RMSE' % (trainScore))
    # testScore = math.sqrt(mean_squared_error(test_y[:, 0], test_y_pred[:, 0]))
    # print('Test Score: %.2f RMSE' % (testScore))
    train_y_pred = reshape_y_hat(train_y_pred.cpu().detach().numpy(), 1)[-1,:,:]
    train_y = reshape_y_hat(train_y.cpu().numpy(), 1)[-1, :, :]
    test_y_pred = reshape_y_hat(test_y_pred.cpu().detach().numpy(), 1)[-1,:,:]
    test_y = reshape_y_hat(test_y.cpu().numpy(), 1)[-1, :, :]
    # testScore = math.sqrt(mean_squared_error(test_y, test_y_pred))
    testScore = math.sqrt(mean_squared_error(train_y, train_y_pred))
    print('Test Score: %.2f RMSE' % (testScore))
    testScore = math.sqrt(mean_squared_error(test_y, test_y_pred))
    print('Test Score: %.2f RMSE' % (testScore))