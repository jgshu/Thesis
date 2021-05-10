import pandas as pd
import numpy as np
import os


def find_files(path, suffix='.csv'):
    file_names = os.listdir(path)

    return [file_name[:-4] for file_name in file_names if file_name.endswith(suffix)]


def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
    data.drop([col], axis=1, inplace=True)

    return data


def last_hour_interpolation(last_hour_value, this_hour_value, current_time_step):
    this_hour_weight = current_time_step / 4.0  # 1h = 4 * 15min
    last_hour_weight = 1.0 - this_hour_weight

    return last_hour_weight * last_hour_value + this_hour_weight * this_hour_value


def create_dataset(data, n_predictions, n_next):
    dim = data.shape[1]
    train_x, train_y  = [], []
    for i in range(data.shape[0] - n_predictions - n_next + 1):
        a = data[i: (i + n_predictions), :]
        train_x.append(a)
        tempb = data[(i + n_predictions): (i + n_predictions + n_next), :]
        b = []
        for j in range(len(tempb)):
            for k in range(dim):
                b.append(tempb[j, k])
        train_y.append(b)
    train_x = np.array(train_x, dtype='float64')
    train_y = np.array(train_y, dtype='float64')
    return train_x, train_y
