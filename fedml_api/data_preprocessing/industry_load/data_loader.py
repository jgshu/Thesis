import logging
import os
import numpy as np
import torch

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def find_files(path, suffix='.csv'):
    file_names = os.listdir(path)
    if suffix == '.':
        return [file_name for file_name in file_names]
    else:
        return [file_name[:-len(suffix)] for file_name in file_names if file_name.endswith(suffix)]


def get_tensor_data(base_path, args, specific_user_id):
    normalization_tvt_path = base_path + 'data/type_%s/day_%s/%s_normalization_tvt/' % (
    args.type_num, args.day_range, args.norm)

    file_names_list = find_files(normalization_tvt_path, suffix='.')
    for file_name in file_names_list:
        co_name, user_id = file_name.split('_')
        if user_id == specific_user_id:
            logging.info('-------' + co_name + '--------')
            logging.info('-------' + user_id + '--------')
            train_x = np.load(normalization_tvt_path + file_name + '/train_x_in_%s_out_%s_range_%s.npy' % (args.seq_len, args.out_features, args.train_range))
            train_y = np.load(normalization_tvt_path + file_name + '/train_y_in_%s_out_%s_range_%s.npy' % (args.seq_len, args.out_features, args.train_range))
            validation_x = np.load(normalization_tvt_path + file_name + '/validation_x_in_%s_out_%s_range_%s.npy' % (args.seq_len, args.out_features, args.train_range))
            validation_y = np.load(normalization_tvt_path + file_name + '/validation_y_in_%s_out_%s_range_%s.npy' % (args.seq_len, args.out_features, args.train_range))
            break

    train_x = torch.from_numpy(train_x).type(torch.Tensor)
    train_y = torch.from_numpy(train_y).type(torch.Tensor)
    validation_x = torch.from_numpy(validation_x).type(torch.Tensor)
    validation_y = torch.from_numpy(validation_y).type(torch.Tensor)

    return train_x, train_y, validation_x, validation_y


# for centralized training
def get_dataloader(base_path, args, user_id):

    return get_dataloader_industry_load(base_path, args, user_id)


def get_dataloader_industry_load(base_path, args, user_id):
    train_x, train_y, validation_x, validation_y = get_tensor_data(base_path, args, user_id)

    train = torch.utils.data.TensorDataset(train_x, train_y)
    validation = torch.utils.data.TensorDataset(validation_x, validation_y)

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               drop_last=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    drop_last=True)

    return train_loader, validation_loader


def load_partition_data_industry_load(base_path, args, type_clients_list):
    class_num = 0

    server = type_clients_list[-1]
    clients_list = type_clients_list[:-1]

    train_data_global, validation_data_global = get_dataloader(base_path, args, server)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(train_data_global)))
    train_data_num = len(train_data_global)
    validation_data_num = len(validation_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    validation_data_local_dict = dict()

    for client_idx in range(args.client_num_in_total):

        logging.info("client_idx = %d, local_sample_number = %s" % (client_idx, clients_list[client_idx]))

        train_data_local, validation_data_local = get_dataloader(base_path, args, clients_list[client_idx])
        user_train_data_num = len(train_data_local)
        user_validation_data_num = len(validation_data_local)

        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(validation_data_local)))

        train_data_local_dict[client_idx] = train_data_local
        validation_data_local_dict[client_idx] = validation_data_local

        # TODO:  client[client_idx]的sample num，目前不确定是多少，但是由于权重是(sample1)/(sample1+sample2)=0.5，故不影响
        data_local_num_dict[client_idx] = user_train_data_num

    return train_data_num, validation_data_num, train_data_global, validation_data_global, \
           data_local_num_dict, train_data_local_dict, validation_data_local_dict, class_num