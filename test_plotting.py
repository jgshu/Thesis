import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_by_per_day(test_path, args, file_name, test_y, test_y_pred, color_list, font, specific_user_id, per_day=7):
    fig = plt.figure()
    fig.set_size_inches(40, 60, forward=True)
    week_len = test_y.shape[0] // (per_day * args.day_range)

    for i in range(week_len - 1):
        ax = plt.subplot(week_len, 1, i + 1)

        y1 = test_y[i * per_day * args.day_range: (i+1) * per_day * args.day_range]
        y2 = test_y_pred[i * per_day * args.day_range: (i+1) * per_day * args.day_range]

        x = range(0, y1.shape[0])
        ax.plot(x, y1, color_list[0])
        ax.plot(x, y2, color_list[-1])
        plt.xticks(font=font)
        plt.yticks(font=font)
        ax.set_ylabel('values', font=font)
        ax.set_xticks(np.arange(0, y1.shape[0], args.day_range))
        ax.set_xticklabels(np.arange(0, per_day))

    plt.savefig(test_path + 'stacked_per_day_%02d_uid_%s_%s.jpg' % (per_day, specific_user_id, file_name))
    plt.close(fig)


def plot_single(test_path, args, file_name, test_y, test_y_pred, color_list, font, specific_user_id, per_day=7):
    fig, ax = plt.subplots(figsize=(100, 20))
    y1 = test_y
    y2 = test_y_pred
    x = range(0, y1.shape[0])
    ax.plot(x, y1, color_list[0])
    ax.plot(x, y2, color_list[-1])
    plt.xticks(font=font)
    plt.yticks(font=font)
    ax.set_ylabel('values', font=font)
    ax.set_xticks(np.arange(0, y1.shape[0], per_day * args.day_range))
    ax.set_xticklabels(np.arange(0, int(y1.shape[0] / (per_day * args.day_range)) + 1))

    plt.savefig(test_path + 'single_%s_%s.jpg' % (specific_user_id, file_name))
    plt.close(fig)

def test_plotting(test_path, args, file_name, dt_string, test_y, test_y_pred, specific_user_id):
    font = {
        'family': 'SimHei',
        'weight': 'normal',
        # 'size': 10,
    }
    CB91_Blue = '#2CBDFE'
    CB91_Green = '#47DBCD'
    CB91_Pink = '#F3A0F2'
    CB91_Purple = '#9D2EC5'
    CB91_Violet = '#661D98'
    CB91_Amber = '#F5B14C'
    color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
                  CB91_Purple, CB91_Violet]

    if not os.path.exists(test_path):
        os.makedirs(test_path)
    plot_by_per_day(test_path, args, file_name, test_y, test_y_pred, color_list, font, specific_user_id)
    plot_single(test_path, args, file_name, test_y, test_y_pred, color_list, font, specific_user_id)


