import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_by_week_range(test_path, args, file_name, test_y, test_y_pred, color_list, font, specific_user_id, i=0, week_range=7):
    fig = plt.figure()
    fig.set_size_inches(40, 60, forward=True)

    test_y_i = test_y[i::args.day_range, :]
    test_y_pred_i = test_y_pred[i::args.day_range, :]

    day_len = test_y_i.shape[0]
    week_len = int(day_len / week_range)

    for k in range(week_len):
        ax = plt.subplot(week_len, 1, k + 1)

        y1 = test_y_i[k * week_range]
        y2 = test_y_pred_i[k * week_range]
        for j in range(1, week_range):
            y1 = np.concatenate((y1, test_y_i[k * week_range + j]))
            y2 = np.concatenate((y2, test_y_pred_i[k * week_range + j]))

        x = range(0, week_range * args.day_range)
        ax.plot(x, y1, color_list[0])
        ax.plot(x, y2, color_list[-1])
        plt.xticks(font=font)
        plt.yticks(font=font)
        ax.set_ylabel('values', font=font)
        ax.set_xticks(np.arange(0, len(y1), args.day_range))
        ax.set_xticklabels(np.arange(0, int(len(y1) / args.day_range)))

    plt.savefig(test_path + 'stacked_%02d_week_%s_%s.jpg' % (week_range, specific_user_id, file_name))
    plt.close(fig)


def plot_single(test_path, args, file_name, test_y, test_y_pred, color_list, font, specific_user_id, i=0):
    fig, ax = plt.subplots(figsize=(100, 20))
    y1 = test_y[i::args.day_range, :]
    y1 = y1.reshape(y1.shape[0] * y1.shape[1])

    y2 = test_y_pred[i::args.day_range, :]
    y2 = y2.reshape(y2.shape[0] * y2.shape[1])

    x = range(0, y1.shape[0])
    ax.plot(x, y1, color_list[0])
    ax.plot(x, y2, color_list[-1])
    plt.xticks(font=font)
    plt.yticks(font=font)
    ax.set_ylabel('values', font=font)
    ax.set_xticks(np.arange(0, y1.shape[0], args.day_range))
    ax.set_xticklabels(np.arange(0, int(y1.shape[0] / args.day_range)))

    plt.savefig(test_path + 'single_%s_%s.jpg' % (specific_user_id, file_name))
    plt.close(fig)

def test_plotting(base_path, args, file_name, dt_string, test_y, test_y_pred, specific_user_id):
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

    test_path = base_path + 'output/img/type_%s/test/day_%s/%s_normalization_%s/%s/' % (args.type_num, args.day_range, args.norm, args.train_range, dt_string)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    plot_by_week_range(test_path, args, file_name, test_y, test_y_pred, color_list, font, specific_user_id, i=0, week_range=7)
    plot_single(test_path, args, file_name, test_y, test_y_pred, color_list, font, specific_user_id, i=0)


