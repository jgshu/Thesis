import pandas as pd
import numpy as np
import torch
import torch.nn as nn
# from lstm_model import RNN_LoadForecastser
from lstm_model_2 import LoadForecastser
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import wandb
import argparse
import copy
import logging
import sys
from datetime import datetime


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='lf', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='type10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--n_features', type=int, default=27, metavar='N',
                        help='number of features')

    parser.add_argument('--n_hidden', type=int, default=512, metavar='N',
                        help='number of hidden nodes')

    parser.add_argument('--seq_len', type=int, default=672, metavar='N',
                        help='sequence length')

    parser.add_argument('--n_layers', type=int, default=2, metavar='N',
                        help='number of layers')

    parser.add_argument('--out_features', type=int, default=96, metavar='N',
                        help='number of out features')

    parser.add_argument('--do', type=float, default=0.2, metavar='N',
                        help='drop out rate')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=150, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--frequency_of_the_test', type=int, default=10,
                        help='the frequency of the test')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    return parser


def inverse_y(csv_path, user_id):
    file_list = os.listdir(csv_path)
    user_id_df_dict = {}
    user_id_co_name_dict = {}
    for i in range(len(file_list)):
        file = file_list[i]
        coName, userID = file[:-4].split('_')
        print(coName)
        df = pd.read_csv(csv_path + file)
        user_id_df_dict[userID] = df
        user_id_co_name_dict[userID] = coName

    co_name = user_id_co_name_dict[user_id]
    print(co_name)
    user_df = user_id_df_dict[user_id].copy()
    col_load = user_df.loc[:, 'load'].values.reshape(-1, 1)
    # 提取load的最大最小值
    scaler = MinMaxScaler()
    scaler = scaler.fit(col_load)
    del user_df
    del col_load
    del user_id_df_dict
    return scaler


def load_partition_data_industry_load(npy_path, batch_size, user_id):
    dir_list = os.listdir(npy_path)
    user_id_co_name_dict = {}
    for i in range(len(dir_list)):
        directory = dir_list[i]
        c, u = directory[:].split('_')
        # print(c)
        user_id_co_name_dict[u] = c
    co_name = user_id_co_name_dict[user_id]
    logging.info("co_name = %s" % co_name)

    path = npy_path + '%s_%s/' % (co_name, user_id)
    train_x = np.load(path + 'train_x_days_365.npy')
    train_y = np.load(path + 'train_y_days_365.npy')
    test_x = np.load(path + 'test_x_days_365.npy')
    test_y = np.load(path + 'test_y_days_365.npy')

    # 判断是否有Nan
    logging.info("Is there any nan:")
    logging.info(np.any(np.isnan(train_x)))
    logging.info(np.any(np.isnan(train_y)))
    logging.info(np.any(np.isnan(test_x)))
    logging.info(np.any(np.isnan(test_y)))

    train_x = torch.from_numpy(train_x).type(torch.Tensor)
    train_y = torch.from_numpy(train_y).type(torch.Tensor)
    test_x = torch.from_numpy(test_x).type(torch.Tensor)
    test_y = torch.from_numpy(test_y).type(torch.Tensor)

    train = torch.utils.data.TensorDataset(train_x, train_y)
    test = torch.utils.data.TensorDataset(test_x, test_y)

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=batch_size,
                                               shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_loader, test_loader


def load_data(args, dataset_name, user_id):
    args_batch_size = args.batch_size
    npy_path = "/home/gavin/PycharmProjects/Thesis/output/train_test/type10/"
    logging.info("load_data. dataset_name = %s, user_id = %s" % (dataset_name, user_id))
    train_loader, test_loader = load_partition_data_industry_load(npy_path, args_batch_size, user_id)

    dataset = [train_loader, test_loader]
    return dataset


def create_model(args, device, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "lf" and args.dataset == "type10":
        logging.info("lf + type10")
        model = LoadForecastser(n_features=args.n_features, n_hidden=args.n_hidden, seq_len=args.seq_len,
                                n_layers=args.n_layers, out_features=args.out_features,
                                do=args.do, device=device).to(device)
    return model


def test(dataset, b_use_test_dataset, device, model):
    if b_use_test_dataset:
        test_data = dataset[1]
    else:
        test_data = dataset[0]

    model.to(device)
    model.eval()

    metrics = {
        'test_loss': 0,
        'test_total': 0
    }
    criterion = nn.MSELoss().to(device)

    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(test_data):
            x, labels = x.to(device), labels.to(device)
            pred = model(x)
            loss = criterion(pred, labels)

            metrics['test_loss'] += loss.item() * labels.size(0)
            metrics['test_total'] += labels.size(0)

    return metrics


def train(dataset, args, device, model):
    train_data = dataset[0]
    model.to(device)

    # train and update
    criterion = nn.MSELoss().to(device)

    # 未正则化
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 正则化
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for i in range(len(list(model.parameters()))):
        logging.info(list(model.parameters())[i].size())

    epoch_loss = []

    logging.info("################train")
    for epoch in range(args.epochs):
        model.train()
        batch_loss = []
        for batch_idx, (x, labels) in enumerate(train_data):
            x, labels = x.to(device), labels.to(device)
            model.zero_grad()
            log_probs = model(x)
            loss = criterion(log_probs, labels)
            loss.backward()

            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        logging.info('Epoch: {}\tLoss: {:.6f}'.format(
            epoch, sum(epoch_loss) / len(epoch_loss)))
        if (epoch % args.frequency_of_the_test == 0) | (epoch == args.epochs - 1):
            logging.info("################test : {}".format(epoch))
            train_metrics = {
                'num_samples': [],
                'losses': []
            }

            test_metrics = {
                'num_samples': [],
                'losses': []
            }

            # train data
            train_local_metrics = test(dataset, False, device, model)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = test(dataset, True, device, model)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            # test on training dataset
            train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

            # test on test dataset
            test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

            stats = {'training_loss': train_loss}
            wandb.log({"Train/Loss": train_loss, "epoch": epoch})
            logging.info(stats)

            stats = {'test_loss': test_loss}
            wandb.log({"Test/Loss": test_loss, "epoch": epoch})
            logging.info(stats)

            now = datetime.now()
            dt_string = now.strftime("%Y%m%d-%H%M%S")
            torch.save(model.state_dict(), './temp-%s.pth' % dt_string)

def training():
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description='Thesis-type10'))
    args = parser.parse_args()
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    run = wandb.init(
        project="thesis",
        name="StackLSTM" + "-e" + str(args.epochs) + "-lr" + str(args.lr),
        config=args
    )

    server = ['10']
    clients = ['638024734', '662615482']

    user_id = clients[0]

    dataset = load_data(args, args.dataset, user_id)

    model = create_model(args, device=device, model_name=args.model, output_dim=args.out_features)
    logging.info(model)

    train(dataset, args, device, model)


