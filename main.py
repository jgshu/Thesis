import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import logging
import wandb
from datetime import datetime
from model import *

from data_preprocessing.txt_to_csv import txt_to_csv
from data_preprocessing.split_data_by_trade_type import split_data_by_trade_type
from data_preprocessing.feature_engineering import feature_engineering
from data_preprocessing.anomaly_detection import anomaly_detection
from data_preprocessing.data_normalization import data_normalization
from data_preprocessing.train_validation_test_split import train_validation_test_split
from data_preprocessing.utils import find_files
from daily_load_plotting import daily_load_plotting
from training import training
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

    parser.add_argument('--model', type=str, default='simpleLSTM', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--type_num', type=int, default=10, metavar='N',
                        help='dataset used for training')

    parser.add_argument('--n_features', type=int, default=27, metavar='N',
                        help='number of features')

    parser.add_argument('--n_hidden', type=int, default=512, metavar='N',
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

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=200, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--frequency_of_the_test', type=int, default=10,
                        help='the frequency of the test')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    return parser


def create_model(args, device, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))

    model = None
    if args.model == 'BiLSTM':
        model = BiLSTM(n_features=args.n_features, n_hidden=args.n_hidden, seq_len=args.seq_len,
                  n_layers=args.n_layers, out_features=args.out_features, do=args.do,
                  device=device).to(device)
    elif args.model == 'simpleLSTM':
        model = simpleLSTM(n_features=args.n_features, n_hidden=args.n_hidden, seq_len=args.seq_len,
                  n_layers=args.n_layers, out_features=args.out_features, do=args.do,
                  device=device).to(device)

    return model


def before_normalization(base_path, type_num):
    # txt_to_csv(base_path)
    # split_data_by_trade_type(base_path)
    # feature_engineering(base_path, type_num)
    # sum_user_id_list = ['638024734', '662615482']
    feature_engineering(base_path, type_num, sum_flag=True)
    # anomaly_detection(base_path, type_num)
    anomaly_detection(base_path, type_num, sum_flag=True)


def normalization_and_split(base_path, type_num, day_range=96, norm='minmax'):
    # data_normalization(base_path, type_num, day_range=day_range, norm=norm)
    data_normalization(base_path, type_num, day_range=day_range, norm=norm, sum_flag=True)
    n_predictions = day_range * 7
    n_next = day_range
    # train_validation_test_split(base_path, type_num, n_predictions, n_next, 426, 122, 61, day_range=day_range, norm=norm, sum_flag=False)
    # train_validation_test_split(base_path, type_num, n_predictions, n_next, 426, 122, 61, day_range=day_range, norm=norm, sum_flag=True)
    train_validation_test_split(base_path, type_num, n_predictions, n_next, 512, 146, 73, day_range=day_range, norm=norm, sum_flag=False)
    train_validation_test_split(base_path, type_num, n_predictions, n_next, 512, 146, 73, day_range=day_range, norm=norm, sum_flag=True)


def data_preprocessing(base_path, type_num):
    # before_normalization(base_path, type_num)
    # normalization_and_split(base_path, type_num, day_range=24, norm='minmax')
    # normalization_and_split(base_path, type_num, day_range=24, norm='standard')
    # normalization_and_split(base_path, type_num, day_range=48, norm='minmax')
    normalization_and_split(base_path, type_num, day_range=48, norm='standard')
    # normalization_and_split(base_path, type_num, day_range=96, norm='minmax')
    # normalization_and_split(base_path, type_num, day_range=96, norm='standard')

if __name__ == '__main__':
    base_path = '../../../Downloads/Thesis-temp/'
    type_num = 10
    server = ['10']
    clients = ['638024734', '662615482']
    user_id = server[0]
    parser = add_args(argparse.ArgumentParser(description='Thesis'))
    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    model = create_model(args, device=device, model_name=args.model, output_dim=args.out_features)

    # data_preprocessing(base_path, type_num)
    # daily_load_plotting(base_path, type_num, day_range=96, norm='standard')
    # daily_load_plotting(base_path, type_num, day_range=48, norm='standard')
    # daily_load_plotting(base_path, type_num, start=426+122+8, end=426+122+61, week_range=1, day_range=48, norm='standard')
    # daily_load_plotting(base_path, type_num, start=512+146+8, end=512+146+73, week_range=1, day_range=48, norm='standard')
    # daily_load_plotting(base_path, type_num, start=1, end=40, week_range=1, day_range=48, norm='standard')
    # daily_load_plotting(base_path, type_num, day_range=24, norm='standard')
    #
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.info(args)
    logger.info(device)
    logging.info(model)

    run = wandb.init(
        project="thesis",
        name=args.model + "-e" + str(args.epochs) + "-lr" + str(args.lr),
        config=args,
    )

    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    logger.info(dt_string)

    # dt_string = '20210511_195848'

    model_path = base_path + 'output/model/%s/' % dt_string
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    training(base_path, model_path, args, device, model, user_id)
    testing(base_path, model_path, args, dt_string, model, user_id)
