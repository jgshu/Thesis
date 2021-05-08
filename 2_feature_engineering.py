import pandas as pd
import numpy as np
import os


def encode(data, col, max_val):
    """
    :param data:
    :param col:
    :param max_val:
    :return:
    """
    data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
    data.drop([col], axis=1, inplace=True)
    return data


def last_hour_interpolation(last_hour_value, this_hour_value, current_time_step):
    """
    :param last_hour_value:
    :param this_hour_value:
    :param current_time_step:
    :return: lastHourInterpolation
    """
    this_hour_weight = current_time_step / 4.0  # 1h = 4 * 15min
    last_hour_weight = 1.0 - this_hour_weight
    return last_hour_weight * last_hour_value + this_hour_weight * this_hour_value

# 行业选择
type_num = 11

print('Loading weather data...')
# 导入天气数据
weather_path = './data/weather_original/'
weather_file = weather_path + 'nanjing_weather.csv'
weather_df = pd.read_csv(weather_file)
weather_df.drop(['站点', '经度', '纬度', '省份', '市', '区（县）'], axis=1, inplace=True)

print('Post interpolation...')
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

new_weather_df = pd.DataFrame()
i = 0
for feature in weather_features[4:]:
    new_weather_df[feature] = post_interpolation[i][:]
    i = i + 1

# 由于缺少2017-01-01 00:00:00 的数据，用newWeatherDF尾行填充
for i in range(4):
    new_weather_df = new_weather_df.append(new_weather_df.iloc[-1:, :], ignore_index=True)

# 修改中文特征为英文特征
colNameDict = {
    '气压': 'barometer',
    '风速': 'windspeed',
    '平均温度': 'temp',
    '相对湿度': 'humidity',
    '降水': 'precipitation',
}
new_weather_df.rename(columns=colNameDict, inplace=True)

print('Saving weather data...')
# 保存newWeatherDF
weather_path = './data/weather/'
new_weather_df.to_csv(weather_path + 'new_nanjing_weather.csv', index=False)

print('Loading industry load...')
# 导入行业负荷数据
type_path = './data/industry_load_by_type/%s/' % type_num
file_list = os.listdir(type_path)

print('1 hour to 15 minutes...')
# 修改数据的格式
user_id_df_dict = {}
user_id_co_name_dict = {}
for i in range(len(file_list)):
    file = file_list[i]
    coName, userID = file[:-4].split('_')
    print(coName)
    user_id_co_name_dict[userID] = coName
    df = pd.read_csv(type_path + file)
    df['DATA_DATE'] = pd.to_datetime(df['DATA_DATE'])
#     df['day'] = df['DATA_DATE'].dt.day
#     df['month'] = df['DATA_DATE'].dt.month
#     df['year'] = df['DATA_DATE'].dt.year
#     df['day_of_week'] = df['DATA_DATE'].dt.day_name()
    df['day_of_week'] = df['DATA_DATE'].dt.dayofweek
    df.drop(['CONS_NO'], axis=1, inplace=True)
    len_df = len(df)
    load_list = list(df.iloc[:, 1: 97].values.reshape((96 * len_df, )))
    data_date_list = list(df['DATA_DATE'].values.repeat(96))
#     day_list = list(df['day'].values.repeat(96))
#     month_list = list(df['month'].values.repeat(96))
#     year_list = list(df['year'].values.repeat(96))
    day_of_week_list = list(df['day_of_week'].values.repeat(96))
    for k in range(len(data_date_list)):
        data_date_list[k] = data_date_list[k] + np.timedelta64(k % 96 * 15, 'm')
#     new_df_dict = {'data_date': data_date_list, 'year': year_list, 'month': month_list,
#                    'day': day_list, 'day_of_week': day_of_week_list, 'load': load_list}
    new_df_dict = {'data_date': data_date_list, 'day_of_week': day_of_week_list, 'load': load_list}
    new_df = pd.DataFrame(new_df_dict)
    user_id_df_dict[userID] = new_df

print('Extracting missing date...')
# 提取每个user缺失的日期
user_id_missing_date_dict = {}
for userID in user_id_df_dict.keys():
    new_df = user_id_df_dict[userID]
    # print('------------------')
    # print(userID)
    # print('------------------')
    cnt = 0
    missing_date_list = []
    date = np.datetime64('2015-01-01T00:00:00.00')
    for i in range(731):
        if (len(new_df[(new_df['data_date'] - date < np.timedelta64(1, 'D')) &
                      (new_df['data_date'] - date >= np.timedelta64(0, 'D'))]) != 96):
            # print(date)
            missing_date_list.append(date)
            cnt = cnt + 1
        date = date + np.timedelta64(1, 'D')
    if cnt <= 5:
        user_id_missing_date_dict[userID] = missing_date_list

print('Filling missing data...')
# 缺失的数据用下周的数据填充
new_user_id_df_dict = {}
for userID in user_id_missing_date_dict.keys():
    print('------------------')
    print(userID)
    print('------------------')
    df = user_id_df_dict[userID].copy()
    index = df.index
    missing_date_list = user_id_missing_date_dict[userID]
    for missing_date in missing_date_list:
        date = missing_date + np.timedelta64(7, 'D')
        next_week_day_df = df[(df['data_date']- date < np.timedelta64(1, 'D')) &
                      (df['data_date'] - date >= np.timedelta64(0, 'D'))].copy()
        print(len(next_week_day_df))
        data_date_list = []
        for m in range(96):
            data_date_list.append(missing_date + np.timedelta64(m * 15, 'm'))
        next_week_day_df['data_date'] = data_date_list
#         next_week_day_df['day'] = next_week_day_df['data_date'].dt.day
#         next_week_day_df['month'] = next_week_day_df['data_date'].dt.month
#         next_week_day_df['year'] = next_week_day_df['data_date'].dt.year
#         next_week_day_df['day_of_week'] = next_week_day_df['data_date'].dt.day_name()
        next_week_day_df['day_of_week'] = next_week_day_df['data_date'].dt.dayofweek
        above_date = missing_date - np.timedelta64(15, 'm')
        print(above_date)
        above_idx = index[df['data_date'] == above_date][-1]
        above = df.iloc[:above_idx + 1, :]
        below = df.iloc[above_idx + 1:, :]
        print(len(above), len(below), len(above) + len(below))
        df = pd.concat([above, next_week_day_df, below], ignore_index=True)
        index = df.index
    new_user_id_df_dict[userID] = df

# 删除之前的字典
del user_id_df_dict

print('Generating features: workday, holiday...')
print('sin, cos...')
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

# 添加workday和holiday特征，并将month，day, hour, minute特征转变为sin，cos分量
# cols = ['data_date', 'year', 'month_sin', 'month_cos', 'day_sin',
#         'day_cos', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
#         'day_of_week', 'workday', 'holiday', 'temp', 'precipitation',
#         'humidity', 'barometer', 'windspeed', 'load',
#        ]
cols = ['data_date', 'year_2015', 'year_2016', 'workday_0', 'workday_1', 'holiday_0', 'holiday_1',
        'day_of_week_0', 'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 'day_of_week_4',
        'day_of_week_5', 'day_of_week_6', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'temp', 'precipitation',
        'humidity', 'barometer', 'windspeed', 'load',
        ]
for userID in new_user_id_df_dict.keys():
    df = new_user_id_df_dict[userID].copy()
    day_of_week_list = df['day_of_week'].values.tolist()
    #     workday_list = [0 if (x == 'Saturday') | (x == 'Sunday') else 1 for x in day_of_week_list]
    workday_list = [0 if (x == 5) | (x == 6) else 1 for x in day_of_week_list]
    index = df.index
    for date in special_workday_date_list:
        idx = index[df['data_date'] == date][0]
        for i in range(96):
            workday_list[idx + i] = 1
    df['workday'] = workday_list
    holiday_list = [0] * 70176
    for date in holidays_date_list:
        idx = index[df['data_date'] == date][0]
        for i in range(96):
            holiday_list[idx + i] = 1
    df['holiday'] = holiday_list

    df['year'] = df['data_date'].dt.year
    df['month'] = df['data_date'].dt.month
    df['day'] = df['data_date'].dt.day
    df['hour'] = df['data_date'].dt.hour
    df['minute'] = df['data_date'].dt.minute

    df = encode(df, 'month', 12)
    df = encode(df, 'day', 365)
    df = encode(df, 'hour', 24)
    df = encode(df, 'minute', 60)
    df = pd.concat([df, new_weather_df], axis=1, join='inner')

    # Random Forest之后
    df = pd.get_dummies(df, prefix=['day_of_week'], columns=['day_of_week'])
    df = pd.get_dummies(df, prefix=['year'], columns=['year'])
    df = pd.get_dummies(df, prefix=['workday'], columns=['workday'])
    df = pd.get_dummies(df, prefix=['holiday'], columns=['holiday'])

    df = df[cols]
    new_user_id_df_dict[userID] = df.copy()

print('Saving original csv files...')
# 保存csv文件
path = './data/type_%s_csv_original/' % type_num
if not os.path.exists(path):
    os.makedirs(path)
for userID in new_user_id_df_dict.keys():
    df = new_user_id_df_dict[userID]
    coName = user_id_co_name_dict[userID]
    df.to_csv(path + '%s_%s.csv' % (coName, userID), index=False)


# sum_user_id_list = ['809035870', '809033085']
#
# sum_load_df = pd.DataFrame()
# for userID in sum_user_id_list:
#      coName = user_id_co_name_dict[userID]
#      print(coName)
#      df = new_user_id_df_dict[userID]
#      if sum_load_df.empty:
#          sum_load_df = df['load']
#      else:
#          sum_load_df = sum_load_df + df['load']
# sum_df = df.copy()
# sum_df['load'] = sum_load_df
# sum_df.to_csv(path + 'type%s_%s.csv' % (type_num, type_num), index=False)

# userID1 = '638024734'
# coName1 = user_id_co_name_dict[userID1]
# print(coName1)
# first_df = new_user_id_df_dict[userID1]
# load_first_df = first_df['load']
#
# userID2 = '662615482'
# coName2 = user_id_co_name_dict[userID2]
# print(coName2)
# second_df = new_user_id_df_dict[userID2]
# load_second_df = second_df['load']
#
# sum_df = first_df.copy()
# load_df = load_first_df + load_second_df
# sum_df['load'] = load_df
# sum_df.to_csv(path + 'sum.csv', index=False)


