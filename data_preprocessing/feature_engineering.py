import pandas as pd
import numpy as np
import os
from .utils import find_files
from .utils import encode
from .utils import last_hour_interpolation


def weather_data_preprocessing(base_path):
    # 导入天气数据
    weather_original_path = base_path + 'data/weather_original/'
    weather_file = weather_original_path + 'nanjing_weather.csv'
    weather_df = pd.read_csv(weather_file)
    weather_df.drop(['站点', '经度', '纬度', '省份', '市', '区（县）'], axis=1, inplace=True)

    # 使用Last Hour Interpolation对weatherDF插值，时间间隔从1h转变为15min
    weather_features = weather_df.columns.values
    post_interpolation = []

    for feature in weather_features[4:]:
        feature_value = weather_df[feature].values
        temp_list = [float(feature_value[0])]
        for i in range(1, len(feature_value)):
            last_hour_value = temp_list[-1]
            for j in range(1, 5):
                temp_list.append(last_hour_interpolation(last_hour_value, feature_value[i], j))
        post_interpolation.append(temp_list)

    for i in range(0, len(post_interpolation)):
        post_interpolation[i].pop()

    weather_df = pd.DataFrame()
    i = 0
    for feature in weather_features[4:]:
        weather_df[feature] = post_interpolation[i][:]
        i = i + 1

    # 由于缺少2017-01-01 00:00:00 的数据，用newWeatherDF尾行填充
    for i in range(4):
        weather_df = weather_df.append(weather_df.iloc[-1:, :], ignore_index=True)

    # 修改中文特征为英文特征
    col_name_dict = {
        '气压': 'barometer',
        '风速': 'windspeed',
        '平均温度': 'temp',
        '相对湿度': 'humidity',
        '降水': 'precipitation',
    }
    weather_df.rename(columns=col_name_dict, inplace=True)

    # 保存newWeatherDF
    print('Saving weather data...')
    weather_path = base_path + 'data/weather/'
    if not os.path.exists(weather_path):
        os.makedirs(weather_path)
    weather_df.to_csv(weather_path + 'new_nanjing_weather.csv', index=False)

    return weather_df


def extract_missing_date(df):
    # 提取user缺失的日期
    cnt = 0
    missing_date_list = []
    date = np.datetime64('2015-01-01T00:00:00.00')
    for i in range(731):
        if (len(df[(df['data_date'] - date < np.timedelta64(1, 'D')) &
                       (df['data_date'] - date >= np.timedelta64(0, 'D'))]) != 96):
            # print(date)
            missing_date_list.append(date)
            cnt = cnt + 1
        date = date + np.timedelta64(1, 'D')
    # TODO: 只提取缺少x天的
    if cnt <= 5:
        pass

    return missing_date_list


def fill_missing_date(df, missing_date_list):
    # 缺失的数据用下周的数据填充
    index = df.index
    print('Number of missing date:', len(missing_date_list))
    for missing_date in missing_date_list:
        date = missing_date + np.timedelta64(7, 'D')
        next_week_day_df = df[(df['data_date']- date < np.timedelta64(1, 'D')) &
                      (df['data_date'] - date >= np.timedelta64(0, 'D'))].copy()
        # print(len(next_week_day_df))
        data_date_list = []
        for m in range(96):
            data_date_list.append(missing_date + np.timedelta64(m * 15, 'm'))
        next_week_day_df['data_date'] = data_date_list
        next_week_day_df['day_of_week'] = next_week_day_df['data_date'].dt.dayofweek
        above_date = missing_date - np.timedelta64(15, 'm')
        # print(above_date)
        above_idx = index[df['data_date'] == above_date][-1]
        above = df.iloc[:above_idx + 1, :]
        below = df.iloc[above_idx + 1:, :]
        # print(len(above), len(below), len(above) + len(below))
        df = pd.concat([above, next_week_day_df, below], ignore_index=True)
        index = df.index

    return df


def get_date_list():
    # 获取2015和2016节假日
    base_date_list = [np.datetime64('2015-01-01T00:00:00.000'),
                      np.datetime64('2015-02-18T00:00:00.000'),
                      np.datetime64('2015-04-04T00:00:00.000'),
                      np.datetime64('2015-05-01T00:00:00.000'),
                      np.datetime64('2015-06-20T00:00:00.000'),
                      np.datetime64('2015-09-27T00:00:00.000'),
                      np.datetime64('2015-10-01T00:00:00.000'),
                      np.datetime64('2016-01-01T00:00:00.000'),
                      np.datetime64('2016-02-07T00:00:00.000'),
                      np.datetime64('2016-04-02T00:00:00.000'),
                      np.datetime64('2016-04-30T00:00:00.000'),
                      np.datetime64('2016-06-09T00:00:00.000'),
                      np.datetime64('2016-09-15T00:00:00.000'),
                      np.datetime64('2016-10-01T00:00:00.000'),
                      ]
    holidays_range_list = [3, 7, 3, 3, 3, 2, 7,
                           3, 7, 3, 3, 3, 3, 7,
                           ]

    holidays_date_list = []
    for i in range(len(base_date_list)):
        temp_list = [base_date_list[i] +  np.timedelta64(j,'D') for j in range(holidays_range_list[i])]
        holidays_date_list.extend(temp_list)

    # 获取2015和2016节假日前后的工作日
    special_workday_date_list = [np.datetime64('2015-01-04T00:00:00.000'),
                                 np.datetime64('2015-02-15T00:00:00.000'),
                                 np.datetime64('2015-02-28T00:00:00.000'),
                                 np.datetime64('2015-10-10T00:00:00.000'),
                                 np.datetime64('2016-02-06T00:00:00.000'),
                                 np.datetime64('2016-02-14T00:00:00.000'),
                                 np.datetime64('2016-06-12T00:00:00.000'),
                                 np.datetime64('2016-09-18T00:00:00.000'),
                                 np.datetime64('2016-10-08T00:00:00.000'),
                                 np.datetime64('2016-10-09T00:00:00.000'),
                                 ]

    return holidays_date_list, special_workday_date_list


def generate_features(load_df, weather_df, holidays_date_list, special_workday_date_list):
    # 添加workday和holiday特征
    cols = ['data_date', 'year_2015', 'year_2016', 'workday_0', 'workday_1', 'holiday_0', 'holiday_1',
            'day_of_week_0', 'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 'day_of_week_4',
            'day_of_week_5', 'day_of_week_6', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'temp', 'precipitation',
            'humidity', 'barometer', 'windspeed', 'load',
            ]
    day_of_week_list = load_df['day_of_week'].values.tolist()
    workday_list = [0 if (x == 5) | (x == 6) else 1 for x in day_of_week_list]
    index = load_df.index

    for date in special_workday_date_list:
        idx = index[load_df['data_date'] == date][0]
        for i in range(96):
            workday_list[idx + i] = 1
    load_df['workday'] = workday_list
    holiday_list = [0] * 70176

    for date in holidays_date_list:
        idx = index[load_df['data_date'] == date][0]
        for i in range(96):
            holiday_list[idx + i] = 1

    load_df['holiday'] = holiday_list
    load_df['year'] = load_df['data_date'].dt.year
    load_df['month'] = load_df['data_date'].dt.month
    load_df['day'] = load_df['data_date'].dt.day
    load_df['hour'] = load_df['data_date'].dt.hour
    load_df['minute'] = load_df['data_date'].dt.minute

    # 将month，day, hour, minute特征转变为sin，cos分量
    load_df = encode(load_df, 'month', 12)
    load_df = encode(load_df, 'day', 365)
    load_df = encode(load_df, 'hour', 24)
    load_df = encode(load_df, 'minute', 60)

    # 合并load和weather
    df = pd.concat([load_df, weather_df], axis=1, join='inner')

    # One-hot
    df = pd.get_dummies(df, prefix=['day_of_week'], columns=['day_of_week'])
    df = pd.get_dummies(df, prefix=['year'], columns=['year'])
    df = pd.get_dummies(df, prefix=['workday'], columns=['workday'])
    df = pd.get_dummies(df, prefix=['holiday'], columns=['holiday'])
    df = df[cols]

    return df


def load_data_proprecessing(type_path, weather_df):
    # 获取工作日和节假日信息
    holidays_date_list, special_workday_date_list = get_date_list()

    user_id_df_dict = {}
    user_id_co_name_dict = {}
    file_names_list = find_files(type_path)

    for file_name in file_names_list:
        # 提取文件名中的coName和userID
        co_name, user_id = file_name.split('_')
        print('-------' + co_name + '--------')
        print('-------' + user_id + '--------')
        user_id_co_name_dict[user_id] = co_name

        # 导入行业负荷数据
        df = pd.read_csv(type_path + file_name + '.csv')

        # 修改数据的格式
        df['DATA_DATE'] = pd.to_datetime(df['DATA_DATE'])
        df['day_of_week'] = df['DATA_DATE'].dt.dayofweek
        df.drop(['CONS_NO'], axis=1, inplace=True)
        len_df = len(df)
        load_list = list(df.iloc[:, 1: 97].values.reshape((96 * len_df, )))
        data_date_list = list(df['DATA_DATE'].values.repeat(96))
        day_of_week_list = list(df['day_of_week'].values.repeat(96))
        for k in range(len(data_date_list)):
            data_date_list[k] = data_date_list[k] + np.timedelta64(k % 96 * 15, 'm')
        new_df_dict = {'data_date': data_date_list, 'day_of_week': day_of_week_list, 'load': load_list}
        df = pd.DataFrame(new_df_dict)

        # 提取user缺失的日期
        missing_date_list = extract_missing_date(df)

        # 缺失的数据用下周的数据填充
        df = fill_missing_date(df, missing_date_list)
        df = generate_features(df, weather_df, holidays_date_list, special_workday_date_list)
        user_id_df_dict[user_id] = df.copy()

    return user_id_df_dict, user_id_co_name_dict


def feature_engineering(base_path, type_num, sum_flag=False, sum_user_id_list=[]):
    type_path = base_path + 'data/industry_load_by_type/%s/' % type_num
    type_num_original_path = base_path + 'data/type_%s/original/' % type_num

    if sum_flag:
        print('Sum of load files...')
        sum_load_df = pd.DataFrame()
        file_names_list = find_files(type_path)

        for file_name in file_names_list:
            co_name, user_id = file_name.split('_')
            if len(sum_user_id_list) > 0:
                if user_id in sum_user_id_list:
                    df = pd.read_csv(type_num_original_path + file_name + '.csv')
                    if sum_load_df.empty:
                        sum_load_df = df['load']
                    else:
                        sum_load_df = sum_load_df + df['load']
            else:
                df = pd.read_csv(type_num_original_path + file_name + '.csv')
                if sum_load_df.empty:
                    sum_load_df = df['load']
                else:
                    sum_load_df = sum_load_df + df['load']
        sum_df = df.copy()
        sum_df['load'] = sum_load_df
        print('Saving original sum csv file...')
        sum_df.to_csv(type_num_original_path + 'type%s_%s.csv' % (type_num, type_num), index=False)

    else:
        # weather预处理
        weather_df = weather_data_preprocessing(base_path)

        # load预处理
        user_id_df_dict, user_id_co_name_dict = load_data_proprecessing(type_path, weather_df)

        # 保存csv文件
        print('Saving original csv files...')

        if not os.path.exists(type_num_original_path):
            os.makedirs(type_num_original_path)

        for user_id in user_id_df_dict.keys():
            df = user_id_df_dict[user_id]
            co_name = user_id_co_name_dict[user_id]
            print('Saving %s - %s csv file...' % (co_name, user_id))
            df.to_csv(type_num_original_path + '%s_%s.csv' % (co_name, user_id), index=False)



