import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import wandb
import argparse
import copy
import logging
from data_preprocessing.utils import find_files


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


def train(model_path, dataset, args, device, model):
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

            torch.save(model.state_dict(), model_path + 'temp-%s.pth' % (epoch / args.frequency_of_the_test))


def training(base_path, model_path, args, device, model, user_id):
    dataset = load_data(base_path, args, user_id)

    train(model_path, dataset, args, device, model)
