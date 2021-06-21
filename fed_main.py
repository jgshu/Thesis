import argparse
import logging
import os
import sys

from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import wandb

# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from model import *
from fedml_api.data_preprocessing.industry_load.data_loader import load_partition_data_industry_load
from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI
from fedml_api.standalone.fedavg.my_model_trainer_forecasting import MyModelTrainer as MyModelTrainerFR
from testing import testing

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--type_num', type=int, default=0, metavar='N',
                        help='dataset used for training')

    parser.add_argument('--fed_alg', type=str, default='fedavg', metavar='N',
                        help='algorithm used for federated learning')

    parser.add_argument('--day_range', type=int, default=48, metavar='N',
                        help='day_range')

    parser.add_argument('--seq_len', type=int, default=48*7, metavar='N',
                        help='sequence length')

    parser.add_argument('--train_range', type=int, default=584, metavar='N',
                        help='day_range')

    parser.add_argument('--norm', type=str, default='standard', metavar='N',
                        help='normalization')

    parser.add_argument('--model', type=str, default='LSTNet', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--n_features', type=int, default=27, metavar='N',
                        help='number of features')

    parser.add_argument('--out_features', type=int, default=1, metavar='N',
                        help='number of out features')

    parser.add_argument('--n_layers', type=int, default=1, metavar='N',
                        help='number of layers')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')

    parser.add_argument('--do', type=float, default=0.2, metavar='N',
                        help='drop out rate')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--hidCNN', type=int, default=128,
                        help='number of CNN hidden units')

    parser.add_argument('--hidRNN', type=int, default=128,
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

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument('--frequency_of_the_test', type=int, default=10,
                        help='the frequency of the test')

    parser.add_argument('--client_num_in_total', type=int, default=9, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=9, metavar='NN',
                        help='number of workers')

    parser.add_argument('--epochs', type=int, default=10, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=15,
                        help='how many round of communications we should use')

    return parser


def load_data(base_path, args, type_clients_list):
    # check if the centralized training is enabled
    centralized = True if args.client_num_in_total == 1 else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False

    logging.info("load_data. type_num = %s" % args.type_num)
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = load_partition_data_industry_load(base_path, args, type_clients_list)

    if centralized:
        train_data_local_num_dict = {
            0: sum(user_train_data_num for user_train_data_num in train_data_local_num_dict.values())}
        train_data_local_dict = {
            0: [batch for cid in sorted(train_data_local_dict.keys()) for batch in train_data_local_dict[cid]]}
        test_data_local_dict = {
            0: [batch for cid in sorted(test_data_local_dict.keys()) for batch in test_data_local_dict[cid]]}
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {cid: combine_batches(train_data_local_dict[cid]) for cid in
                                 train_data_local_dict.keys()}
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]


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


base_path = '../../../Downloads/Thesis-temp/'
parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
args = parser.parse_args()

device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
model = create_model(args, device=device)
model_trainer = MyModelTrainerFR(model, args)

logging.basicConfig()
logger = logging.getLogger()
logging.getLogger('matplotlib.font_manager').disabled = True
logger.setLevel(logging.DEBUG)

logger.info(args)
logger.info(model)
logger.info(device)

run = wandb.init(
    project="thesis-with-fedml-type%s" % args.type_num,
    name="FedAVG-" + args.model + "-r" + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(args.lr),
    config=args,
)

# Set the random seed. The np.random seed determines the dataset partition.
# The torch_manual_seed determines the initial weight.
# We fix these two, so that we can reproduce the result.
# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)

now = datetime.now()
dt_string = '%s_B%s_C%s_D%s_E%s_L%s_L%s_H%s_' \
            % (args.model, args.batch_size, args.comm_round,
               args.do, args.epochs, args.lr, args.n_layers, args.hidRNN) \
            + now.strftime("%Y%m%d%H%M%S")

# dt_string = 'LSTNet_B64_D0.2_E15_L0.0001_L1_H128_20210608145623'

model_path = base_path + 'output/type_%s/day_%s_range_%s_%s_%s/model/%s/save/' \
             % (args.type_num, args.day_range, args.train_range, args.norm, args.fed_alg, dt_string)
if not os.path.exists(model_path):
    os.makedirs(model_path)

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

# load data
dataset = load_data(base_path, args, type_clients_list)

fedavgAPI = FedAvgAPI(model_path, dataset, device, args, model_trainer)
fedavgAPI.train()

model = create_model(args, device=device)

for user_id in type_clients_list:
    testing(base_path, model_path, args, dt_string, model, user_id)
    testing(base_path, model_path, args, dt_string, model, user_id, type='ms')

logger.info(dt_string)
print('----------------------')
print('dt_string:', dt_string)

