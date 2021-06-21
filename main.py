import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import wandb
from datetime import datetime
import os
import argparse
import logging
import shutil

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
    parser.add_argument('--type_num', type=int, default=12, metavar='N',
                        help='dataset used for training')

    parser.add_argument('--day_range', type=int, default=48, metavar='N',
                        help='day_range')

    parser.add_argument('--seq_len', type=int, default=48*7, metavar='N',
                        help='sequence length')

    parser.add_argument('--train_range', type=int, default=584, metavar='N',
                        help='day_range')

    parser.add_argument('--epochs', type=int, default=200, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--norm', type=str, default='standard', metavar='N',
                        help='normalization')

    parser.add_argument('--model', type=str, default='SLSTM', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--n_features', type=int, default=27, metavar='N',
                        help='number of features')

    parser.add_argument('--out_features', type=int, default=1, metavar='N',
                        help='number of out features')

    parser.add_argument('--n_layers', type=int, default=3, metavar='N',
                        help='number of layers')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')

    parser.add_argument('--do', type=float, default=0.2, metavar='N',
                        help='drop out rate')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--hidCNN', type=int, default=128,
                        help='number of CNN hidden units')

    parser.add_argument('--hidRNN', type=int, default=64,
                        help='number of RNN hidden units')

    parser.add_argument('--CNN_kernel', type=int, default=6,
                        help='the kernel size of the CNN layers')

    parser.add_argument('--highway_window', type=int, default=48,
                        help='The window size of the highway component')

    parser.add_argument('--clip', type=float, default=10.,
                        help='gradient clipping')

    parser.add_argument('--skip', type=float, default=48)

    parser.add_argument('--hidSkip', type=int, default=5)

    parser.add_argument('--output_fun', type=str, default='Linear')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--frequency_of_the_test', type=int, default=10,
                        help='the frequency of the test')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    return parser


def create_model(args, device):
    logging.info("create_model. model_name = %s, in_dim = %s, output_dim = %s"
                 % (args.model, args.seq_len, args.out_features))

    model = None
    if args.model == 'BiLSTM':
        model = BiLSTM(args, device=device).to(device)
    elif args.model == 'SLSTM':
        model = SLSTM(args, device=device).to(device)
    elif args.model == 'LSTNet':
        model = LSTNet(args, device=device).to(device)
    return model


def before_normalization(base_path, args, need_filter=False):
    # txt_to_csv(base_path)
    # split_data_by_trade_type(base_path)
    type_num_path = base_path + 'data/type_%s/' % args.type_num
    if os.path.exists(type_num_path):
        shutil.rmtree(type_num_path)
    os.makedirs(type_num_path)

    anomaly_detection_path = base_path + 'output/type_%s/day_%s_range_%s_%s/img/anomaly_detection/' \
                 % (args.type_num, args.day_range, args.train_range, args.norm)
    if os.path.exists(anomaly_detection_path):
        shutil.rmtree(anomaly_detection_path)
    os.makedirs(anomaly_detection_path)

    feature_engineering(base_path, args.type_num)
    # anomaly_detection(base_path, args.type_num, anomaly_detection_path, need_ad=False)
    anomaly_detection(base_path, args.type_num, anomaly_detection_path)

    sum_user_id_list = []
    feature_engineering(base_path, args.type_num, sum_flag=True, sum_user_id_list=sum_user_id_list)
    # anomaly_detection(base_path, args.type_num, anomaly_detection_path, sum_flag=True, need_ad=False)
    anomaly_detection(base_path, args.type_num, anomaly_detection_path, sum_flag=True)

    if need_filter:
        daily_load_plotting(base_path, args, start=1, end=300, per_day=7, need_filter=True)


def normalization(base_path, args):
    data_normalization(base_path, args)
    data_normalization(base_path, args, sum_flag=True)


def plot_norm(base_path, args):
    per_day = 7
    daily_load_plotting(base_path, args, start=1, end=150, per_day=per_day)

    if args.train_range == 426:
        start = 426 + 122 + 8
        end = 426 + 122 + 61
    elif args.train_range == 512:
        start = 512 + 146 + 8
        end = 512 + 146 + 73
    else:
        start = 438 + 146 + 8
        end = 438 + 146 + 146

    daily_load_plotting(base_path, args, start=start, end=end, per_day=per_day)


def split(base_path, args):
    train_validation_test_split(base_path, args, sum_flag=False)
    train_validation_test_split(base_path, args, sum_flag=True)


def data_preprocessing(base_path, args, need_filter=False):
    if need_filter:
        before_normalization(base_path, args, need_filter=need_filter)
    else:
        before_normalization(base_path, args)
        normalization(base_path, args)
        plot_norm(base_path, args)
        split(base_path, args)


if __name__ == '__main__':
    base_path = '../../../Downloads/Thesis-temp/'
    parser = add_args(argparse.ArgumentParser(description='Thesis'))
    parser.add_argument('--fed_alg')
    args = parser.parse_args()

    type_clients_dict = {
        0: ['930131545', '332212524', '150991350',
            '638164411', '930146713', '430174717',
            '332233792', '150131331', '630007616',
            '0'],
        7: ['930131545', '332212524', '150991350', '7'],
        10: ['638164411', '930146713', '430174717', '10'],
        12: ['332233792', '150131331', '630007616', '12'],
        14: ['154287848', '400005515', '650318004', '14'],
        11: ['11']
    }

    type_clients_list = type_clients_dict[args.type_num]
    user_id = type_clients_list[2]

    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    model = create_model(args, device=device)

    # 用于挑选用户
    # data_preprocessing(base_path, args, need_filter=True)
    # data_preprocessing(base_path, args, need_filter=False)

    logging.basicConfig()
    # 禁止font manager的debug信息
    logger = logging.getLogger()
    logging.getLogger('matplotlib.font_manager').disabled = True
    logger.setLevel(logging.DEBUG)

    logger.info(args)
    logger.info(device)
    logging.info(model)

    run = wandb.init(
        project="thesis-without-fedml-type%s" % args.type_num,
        name=args.model + "-e" + str(args.epochs) + "-lr" + str(args.lr),
        config=args,
    )

    now = datetime.now()
    dt_string = 'UID%s_%s_B%s_D%s_E%s_L%s_L%s_H%s_' \
                % (user_id, args.model, args.batch_size, args.do, args.epochs, args.lr, args.n_layers, args.hidRNN) \
                + now.strftime("%Y%m%d%H%M%S")

    # dt_string = 'BiLSTM_B64_D0.2_E200_L0.0001_L1_H128_20210608002143'

    model_path = base_path + 'output/type_%s/day_%s_range_%s_%s/model/%s/save/' \
                 % (args.type_num, args.day_range, args.train_range, args.norm, dt_string)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    training(base_path, model_path, args, device, model, user_id)
    testing(base_path, model_path, args, dt_string, model, user_id)
    testing(base_path, model_path, args, dt_string, model, user_id, 'ms')

    logger.info(dt_string)
    print('----------------------')
    print('dt_string:', dt_string)
