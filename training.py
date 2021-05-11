import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from model import SLSTM
from model import BiLSTM
import os
import wandb
import argparse
import copy
import logging
from datetime import datetime
from data_preprocessing.utils import find_files
from testing import testing


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--day_range', type=int, default=48, metavar='N',
                        help='day_range')

    parser.add_argument('--train_range', type=int, default=426, metavar='N',
                        help='day_range')

    parser.add_argument('--norm', type=str, default='standard', metavar='N',
                        help='normalization')

    parser.add_argument('--model', type=str, default='BiLSTM', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--type_num', type=int, default=10, metavar='N',
                        help='dataset used for training')

    parser.add_argument('--n_features', type=int, default=27, metavar='N',
                        help='number of features')

    parser.add_argument('--n_hidden', type=int, default=128, metavar='N',
                        help='number of hidden nodes')

    parser.add_argument('--seq_len', type=int, default=336, metavar='N',
                        help='sequence length')

    parser.add_argument('--n_layers', type=int, default=2, metavar='N',
                        help='number of layers')

    parser.add_argument('--out_features', type=int, default=48, metavar='N',
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


def load_partition_data_industry_load(normalization_tvt_path, args, specific_user_id):
    file_names_list = find_files(normalization_tvt_path, suffix='.')

    for file_name in file_names_list:
        co_name, user_id = file_name.split('_')
        if user_id == specific_user_id:
            logging.info('-------' + co_name + '--------')
            logging.info('-------' + user_id + '--------')
            train_x = np.load(normalization_tvt_path + file_name + '/train_x_range_%s.npy' % args.train_range)
            train_y = np.load(normalization_tvt_path + file_name + '/train_y_range_%s.npy' % args.train_range)
            validation_x = np.load(normalization_tvt_path + file_name + '/validation_x_range_%s.npy' % args.train_range)
            validation_y = np.load(normalization_tvt_path + file_name + '/validation_y_range_%s.npy' % args.train_range)
            break

    # 判断是否有Nan
    logging.info("Is there any nan:")
    logging.info(np.any(np.isnan(train_x)))
    logging.info(np.any(np.isnan(train_y)))
    logging.info(np.any(np.isnan(validation_x)))
    logging.info(np.any(np.isnan(validation_y)))

    train_x = torch.from_numpy(train_x).type(torch.Tensor)
    train_y = torch.from_numpy(train_y).type(torch.Tensor)
    validation_x = torch.from_numpy(validation_x).type(torch.Tensor)
    validation_y = torch.from_numpy(validation_y).type(torch.Tensor)

    train = torch.utils.data.TensorDataset(train_x, train_y)
    validation = torch.utils.data.TensorDataset(validation_x, validation_y)

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=args.batch_size,
                                               shuffle=False)

    validation_loader = torch.utils.data.DataLoader(dataset=validation,
                                                    batch_size=args.batch_size,
                                                    shuffle=False)

    return train_loader, validation_loader


def load_data(base_path, args, user_id):

    normalization_tvt_path = base_path + 'data/type_%s/day_%s/%s_normalization_tvt/' % (
    args.type_num, args.day_range, args.norm)
    logging.info("load_data. dataset_name = type_%s, user_id = %s" % (args.type_num, user_id))
    train_loader, validation_loader = load_partition_data_industry_load(normalization_tvt_path, args, user_id)

    dataset = [train_loader, validation_loader]

    return dataset


def create_model(args, device, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))

    model = BiLSTM(n_features=args.n_features, n_hidden=args.n_hidden, seq_len=args.seq_len,
                  n_layers=args.n_layers, out_features=args.out_features, do=args.do,
                  device=device).to(device)

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


def train(base_path, dataset, args, device, model):
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

            model_path = base_path + 'output/model/'
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(model.state_dict(), model_path + 'temp-%s.pth' % dt_string)

def training(base_path):
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description='Thesis-type10'))
    args = parser.parse_args()
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    # run = wandb.init(
    #     project="thesis",
    #     name="StackLSTM" + "-e" + str(args.epochs) + "-lr" + str(args.lr),
    #     config=args
    # )

    server = ['10']
    clients = ['638024734', '662615482']

    user_id = clients[0]

    # dataset = load_data(base_path, args, user_id)
    #
    # model = create_model(args, device=device, model_name=args.model, output_dim=args.out_features)
    # logging.info(model)
    #
    # train(base_path, dataset, args, device, model)

    testing(base_path, args, user_id)

