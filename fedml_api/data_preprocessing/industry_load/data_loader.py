import logging
import os
import numpy as np
import torch

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_tensor_data(user_id, type_num):
    npy_path = "../../../../../../Downloads/Thesis-temp/output/train_test/type%s/" % type_num
    dir_list = os.listdir(npy_path)
    user_id_co_name_dict = {}
    for i in range(len(dir_list)):
        directory = dir_list[i]
        c, u = directory[:].split('_')
        # print(c)
        user_id_co_name_dict[u] = c
    co_name = user_id_co_name_dict[user_id]
    print(co_name)

    # TODO: 数据集：train_x_days_365.npy or train_x_days_90.npy
    path = npy_path + '%s_%s/' % (co_name, user_id)

    date_range = 365

    if date_range == 365:
        train_x = np.load(path + 'train_x_days_365.npy')
        train_y = np.load(path + 'train_y_days_365.npy')
        test_x = np.load(path + 'test_x_days_365.npy')
        test_y = np.load(path + 'test_y_days_365.npy')
    elif date_range == 90:
        train_x = np.load(path + 'train_x_days_90.npy')
        train_y = np.load(path + 'train_y_days_90.npy')
        test_x = np.load(path + 'test_x_days_90.npy')
        test_y = np.load(path + 'test_y_days_90.npy')

    train_x = torch.from_numpy(train_x).type(torch.Tensor)
    train_y = torch.from_numpy(train_y).type(torch.Tensor)
    test_x = torch.from_numpy(test_x).type(torch.Tensor)
    test_y = torch.from_numpy(test_y).type(torch.Tensor)

    return train_x, train_y, test_x, test_y


# for centralized training
def get_dataloader(dataset, user_id, type_num, train_bs, test_bs):

    return get_dataloader_industry_load(user_id, type_num, train_bs, test_bs)


def get_dataloader_industry_load(user_id, type_num, train_bs, test_bs):
    train_x, train_y, test_x, test_y = get_tensor_data(user_id, type_num)

    train = torch.utils.data.TensorDataset(train_x, train_y)
    test = torch.utils.data.TensorDataset(test_x, test_y)

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=train_bs,
                                               shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=test_bs,
                                              shuffle=False)

    return train_loader, test_loader


def load_partition_data_industry_load(dataset, type_num, client_number, batch_size):
    class_num = 0
    type_server_dict = {
        3: ['3'],
        10: ['10']
    }
    type_clients_dict = {
        3: ['809035870', '809033085'],
        # 10: ['638024734', '662615482', '153406449', '930146713', '150032002']
        10: ['153406449', '930146713']
    }
    server = type_server_dict[type_num]
    clients = type_clients_dict[type_num]

    train_data_global, test_data_global = get_dataloader(dataset,
                                                         server[0],
                                                         type_num,
                                                         batch_size,
                                                         batch_size
                                                         )
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(train_data_global)))
    train_data_num = len(train_data_global)
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):

        logging.info("client_idx = %d, local_sample_number = %s" % (client_idx, clients[client_idx]))

        train_data_local, test_data_local = get_dataloader(dataset,
                                                           clients[client_idx],
                                                           type_num,
                                                           batch_size,
                                                           batch_size,
                                                           )
        user_train_data_num = len(train_data_local)
        user_test_data_num = len(test_data_local)

        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))

        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

        # TODO:  client[client_idx]的sample num，目前不确定是多少，但是由于权重是(sample1)/(sample1+sample2)=0.5，故不影响
        data_local_num_dict[client_idx] = user_train_data_num

    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num