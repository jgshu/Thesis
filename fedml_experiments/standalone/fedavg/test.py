import argparse
import logging
import os
import sys

from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.industry_load.data_loader import load_partition_data_industry_load

from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI
from fedml_api.standalone.fedavg.my_model_trainer_forecasting import MyModelTrainer as MyModelTrainerFR


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

    parser.add_argument('--type_num', type=int, default=10, metavar='N',
                        help='type number')

    parser.add_argument('--n_features', type=int, default=27, metavar='N',
                        help='number of features')

    parser.add_argument('--n_hidden', type=int, default=500, metavar='N',
                        help='number of hidden nodes')

    parser.add_argument('--seq_len', type=int, default=672, metavar='N',
                        help='sequence length')

    parser.add_argument('--n_layers', type=int, default=2, metavar='N',
                        help='number of layers')

    parser.add_argument('--out_features', type=int, default=96, metavar='N',
                        help='number of out features')

    parser.add_argument('--do', type=float, default=0.2, metavar='N',
                        help='drop out rate')

    parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=3, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=2, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=2, metavar='NN',
                        help='number of workers')

    parser.add_argument('--comm_round', type=int, default=30,
                        help='how many round of communications we should use')

    parser.add_argument('--frequency_of_the_test', type=int, default=3,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    return parser


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


def load_data(args, dataset_name):
    # check if the centralized training is enabled
    centralized = True if args.client_num_in_total == 1 else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False

    logging.info("load_data. dataset_name = %s" % dataset_name)
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = load_partition_data_industry_load(args.dataset, args.type_num, args.client_num_in_total, args.batch_size)

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


def create_model(args, device, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "lf":
        logging.info("lf")
        model = LoadForecastser(n_features=args.n_features, n_hidden=args.n_hidden, seq_len=args.seq_len,
                                n_layers=args.n_layers, out_features=args.out_features,
                                do=args.do, device=device).to(device)
    return model


def custom_model_trainer(args, model):
    if args.dataset == "type10":
        return MyModelTrainerFR(model, args)
    else:
        return None


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    args = parser.parse_args()
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    run = wandb.init(
        project="fedml",
        name="FedAVG-r" + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(args.lr),
        config=args
    )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)

    # load data
    dataset = load_data(args, args.dataset)

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, device=device, model_name=args.model, output_dim=args.out_features)
    model_trainer = custom_model_trainer(args, model)
    logging.info(model)

    fedavgAPI = FedAvgAPI(dataset, device, args, model_trainer)
    fedavgAPI.train()

    now = datetime.now()
    dt_string = now.strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(), './lastest-%s.pth' % dt_string)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file('./lastest-%s.pth' % dt_string)
    run.log_artifact(artifact)

