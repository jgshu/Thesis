import pandas as pd
import numpy as np
import os

def split_data_by_trade_type(base_path):
    # 导入数据
    print('Loading infoDF...')
    industry_load_original_path = base_path + 'data/industry_load_original/'
    info_file = industry_load_original_path + 'industry_user_info.csv'
    info_df = pd.read_csv(info_file)

    # 获取所有用户的ID
    print('Loading userDF...')
    userIDs = []
    load_num_user_id_dict = {}
    for i in range(5):
        load_file = industry_load_original_path + 'load%s.csv' % i
        df = pd.read_csv(load_file).reset_index(drop=True)
        temp = df['CONS_NO'].drop_duplicates().values.tolist()
        userIDs.extend(temp)
        load_num_user_id_dict[i] = temp
        del df

    # 建立coName和userID的键值对，coName对应多个userID
    co_name_user_id_list = info_df[['CONS_NO', 'CONS_NAME']].drop_duplicates().reset_index(drop=True).values
    co_name_user_id_list = co_name_user_id_list.tolist()
    co_name_user_id_dict = {}
    user_id_co_name_dict = {}
    for e in co_name_user_id_list:
        user_id = e[0]
        co_name = e[1]
        user_id_co_name_dict[user_id] = co_name
        if co_name in co_name_user_id_dict:
            co_name_user_id_dict[co_name].append(user_id)
        else:
            co_name_user_id_dict[co_name] = [user_id]

    # 根据地点提取公司
    # 有多行相同的coName，即coName会对应多个userID，需要删除重复行
    print('Getting location...')
    co_name = info_df['CONS_NAME'].drop_duplicates().values
    co_name = co_name.tolist()
    nanjing_list = []; yangzhou_list = []; changzhou_list = []; lianyungang_list = []; wuxi_list = [];
    suzhou_list = []; nantong_list = []; xuzhou_list = []; jiangsu_list = []; china_list = [];
    person_list = []; location = [];
    for co in co_name:
        if '南京' in co:
            nanjing_list.append(co)
        if '扬州' in co:
            yangzhou_list.append(co)
        if '常州' in co:
            changzhou_list.append(co)
        if '连云港' in co:
            lianyungang_list.append(co)
        if '无锡' in co:
            wuxi_list.append(co)
        if '苏州' in co:
            suzhou_list.append(co)
        if '南通' in co:
            nantong_list.append(co)
        if '徐州' in co:
            xuzhou_list.append(co)
        if '江苏' in co:
            jiangsu_list.append(co)
        if '中国' in co:
            china_list.append(co)
        if len(co) < 4:
            person_list.append(co)
    location.extend([nanjing_list, yangzhou_list, changzhou_list, lianyungang_list,
                     wuxi_list, suzhou_list, nantong_list, xuzhou_list, jiangsu_list,
                     china_list, person_list])

    # 根据userDF提取出的userIDs，在infoDF中找到对应的行，建立userID和tradeName的键值对
    # 由于没有重复的userID，故不需要删除重复行
    print('Getting user_id_trade_name_dict...')
    user_id_trade_name_list = info_df[info_df['CONS_NO'].isin(userIDs)][['CONS_NO', 'TRADE_NAME']].reset_index(drop=True).values
    user_id_trade_name_list = user_id_trade_name_list.tolist()
    user_id_trade_name_dict = {}
    for e in user_id_trade_name_list:
        user_id = e[0]
        trade_name = e[1].strip()
        user_id_trade_name_dict[user_id] = trade_name

    # 建立南京的tradeName小类和coName的键值对
    print('Getting trade_name_user_id_in_nangjing_dict...')
    trade_name_user_id_in_nangjing_dict = {}
    trade_name_user_id_cnt_in_nangjing_dict = {}
    user_id_not_found_list = []
    for co_name in nanjing_list:
        for user_id in co_name_user_id_dict[co_name]:
            try:
                trade_name = user_id_trade_name_dict[user_id]
                if trade_name in trade_name_user_id_in_nangjing_dict.keys():
                    trade_name_user_id_in_nangjing_dict[trade_name].append(user_id)
                    trade_name_user_id_cnt_in_nangjing_dict[trade_name] = trade_name_user_id_cnt_in_nangjing_dict[trade_name] + 1
                else:
                    trade_name_user_id_in_nangjing_dict[trade_name] = [user_id]
                    trade_name_user_id_cnt_in_nangjing_dict[trade_name] = 1
            except KeyError:
                # print(userID, coName)
                user_id_not_found_list.append(user_id)

    # 大类定义
    trade_type_trade_name_dict = {
        '1': ['农业服务业', '林业服务业', '畜牧服务业', '渔业服务业'],
        '2': ['烟煤和无烟煤的开采洗选', '褐煤的开采洗选', '其他煤炭开采洗选',
              '天然原油和天然气开采业', '与天然原油和天然气开采有关的服务活动',
              '铁矿采选', '其他黑色金属矿采选', '常用有色金属矿采选', '贵金属矿采选',
              '稀有稀土金属矿采选', '土砂石开采', '化学矿采选', '石棉及其他非金属矿采选'],
        '3': ['棉、化纤纺织及印染精加工', '毛纺织和染整精加工', '麻纺织', '丝绢纺织及精加工',
              '纺织制成品制造', '针织品、编织品及其制品制造', '纺织服装制造', '纺织面料鞋的制造',
              '塑料丝、绳及编织品的制造'],
        '4': ['石墨及其他非金属矿物制品制造', '水泥、石灰和石膏的制造', '水泥制造', '水泥及石膏制品制造',
              '砖瓦、石材及其他建筑材料制造', '玻璃及玻璃制品制造', '平板玻璃制造', '技术玻璃制品制造',
              '光学玻璃制造', '玻璃仪器制造', '日用玻璃制品及玻璃包装容器制造（轻）', '玻璃保温容器制造（轻）',
              '玻璃纤维及制品制造', '玻璃纤维增强塑料制品制造', '其他玻璃制品制造', '陶瓷制品制造',
              '卫生陶瓷制品制造', '特种陶瓷制品制造', '日用陶瓷制品制造（轻）', '园林、陈设艺术及其他陶瓷制品制造（轻）',
              '耐火材料制品制造'],
        '5': ['纺织服装制造', '皮革、毛皮鞣制及制品加工制造', '羽毛(绒)加工及制品制造'],
        '6': ['钢铁冶炼及钢压延加工', '铁合金冶炼', '常用有色金属冶炼及压延加工业',
              '贵金属冶炼及压延加工业', '稀有稀土金属冶炼及压延加工业'],
        '7': ['基础化学原料制造', '化学农药制造', '生物化学农药及微生物农药制造', '专用化学产品制造',
              '日用化学产品制造（轻）', '化学药品原药制造', '化学药品制剂制造'],
        '8': ['金属家具制造', '其他常用有色金属制造和压延加工', '有色金属合金制造和压延加工',
              '结构性金属制品制造', '金属工具制造', '农用及园林用金属工具制造', '刀剪及类似日用金属工具制造（轻）',
              '其他金属工具制造', '集装箱及金属包装容器制造', '金属丝绳及其制品的制造', '建筑、安全用金属制品制造',
              '不锈钢及类似日用金属制品制造（轻）', '其他金属制品制造', '铸币及贵金属制实验室用品制造',
              '其他未列明的金属制品制造', '金属加工机械制造'],
        '9': ['精炼石油产品的制造', '炼焦', '核燃料加工', '核辐射加工'],
        '10': ['其他农副食品加工  ', '焙烤食品制造', '方便食品制造', '其他食品制造',
               '软饮料制造', '其他烟草制品加工'],
        '11': ['起重运输设备制造', '风机、衡器、包装设备等通用设备制造', '矿山、冶金、建筑专用设备制造',
               '化工、木材、非金属加工专用设备制造', '食品、饮料、烟草及饲料生产专用设备制造',
               '印刷、制药、日化生产专用设备制造', '纺织、服装和皮革工业专用设备制造',
               '电子和电工机械专用设备制造', '医疗仪器设备及器械制造（轻）', '环保、社会公共安全及其他专用设备制造',
               '铁路运输设备制造', '交通器材及其他交通运输设备制造', '输配电及控制设备制造', '通信设备制造',
               '雷达及配套设备制造', '广播电视设备制造', '家用视听设备制造（轻）', '其他电子设备制造'],
        '12': ['橡胶板、管、带的制造', '橡胶零件制造', '再生橡胶制造', '日用及医用橡胶制品制造', '橡胶靴鞋制造',
               '其他橡胶制品制造', '塑料家具制造', '塑料薄膜制造', '塑料板、管、型材的制造',
               '塑料丝、绳及编织品的制造', '泡沫塑料制造', '塑料人造革、合成革制造', '塑料包装箱及容器制造（轻）',
               '塑料零件制造', '日用塑料制造（轻）', '其他塑料制品制造（轻）', '玻璃纤维增强塑料制品制造'],
        '13': ['纸浆制造', '造纸', '纸制品制造'],
        '14': ['烘炉、熔炉及电炉制造', '电机制造', '电线、电缆、光缆及电工器材制造', '电池制造（轻）',
               '家用电力器具制造（轻）', '非电力家用器具制造（轻）', '其他电气机械及器材制造', '电子计算机制造',
               '电子器件制造', '电子元件制造', '其中：电厂生产全部耗用电量', '线路损失电量', '抽水蓄能抽水耗用电量',
               '电力供应企业用电'],
        '15': ['抽水蓄能抽水耗用电量', '谷物、棉花等农产品仓储', '其他仓储'],
        '16': ['学前教育', '初等教育', '中等教育', '初中教育', '高中教育', '中等专业教育', '职业中学教育', '技工学校教育',
               '其他中等教育', '高等教育', '普通高等教育', '成人高等教育', '其他教育', '特殊教育', '其他未列明的教育',
               '文物及文化保护', '群众文化活动', '文化艺术经纪代理', '其他文化艺术', '体育组织', '体育场馆', '其他体育',
               '室内娱乐活动', '休闲健身娱乐活动', '其他娱乐活动'],
        '17': ['房地产开发经营', '房地产中介服务', '其他房地产活动', '其他商务服务', '其他居民服务'],
        '18': ['其他住宿服务', '其他餐饮服务'],
        '19': ['农畜产品批发', '食品、饮料及烟草制品批发', '纺织、服装及日用品批发', '文化、体育用品及器材批发',
               '医药及医疗器材批发', '矿产品、建材及化工产品批发', '机械设备、五金交电及电子产品批发', '其他批发',
               '综合零售', '食品、饮料及烟草制品专门零售', '纺织、服装及日用品专门零售', '文化、体育用品及器材专门零售',
               '医药及医疗器材专门零售', '汽车、摩托车、燃料及零配件专门零售', '家用电器及电子产品专门零售',
               '五金、家具及室内装修材料专门零售', '无店铺及其他零售'],
        '20': ['市政公共设施管理', '证券市场管理', '物业管理', '企业管理服务', '市场管理', '工程技术与规划管理',
               '防洪管理', '水资源管理', '其他水利管理', '市政公共设施管理', '城市绿化管理', '游览景区管理'],
        '21': ['电信', '固定电信服务', '移动电信服务', '其他电信服务', '互联网信息服务'],
        '22': ['卫生院及社区医疗活动', '其他卫生活动', '提供住宿的社会福利', '不提供住宿的社会福利']
    }

    # 建立南京的tradeType大类和userID键值对
    print('Getting trade_type_user_id_in_nanjing_dict...')
    trade_type_user_id_in_nanjing_dict = {}
    # 先根据小类找到对应的南京公司
    for trade_name in trade_name_user_id_in_nangjing_dict.keys():
        # 判断小类在哪个大类，把大类的key作为trade_name_user_id_in_nangjing_dict的key
        for tradeType in trade_type_trade_name_dict.keys():
            if trade_name in trade_type_trade_name_dict[tradeType]:
                # 存储大类对应的南京公司
                if tradeType in trade_type_user_id_in_nanjing_dict.keys():
                    trade_type_user_id_in_nanjing_dict[tradeType].extend(trade_name_user_id_in_nangjing_dict[trade_name])
                else:
                    trade_type_user_id_in_nanjing_dict[tradeType] = trade_name_user_id_in_nangjing_dict[trade_name]

    # 根据tradeType输出各userID的csv原始文件
    print('Outputting csv...')

    industry_load_by_type_path = base_path + 'data/industry_load_by_type/'
    for i in range(5):
        print('Outputting %s...' % i)
        load_file = industry_load_original_path + 'load%s.csv' % i
        user_df = pd.read_csv(load_file).reset_index(drop=True)
        for tradeType in trade_type_user_id_in_nanjing_dict.keys():
            trade_type_path = industry_load_by_type_path + tradeType + '/'
            if not os.path.exists(trade_type_path):
                os.makedirs(trade_type_path)
            for user_id in trade_type_user_id_in_nanjing_dict[tradeType]:
                co_name = user_id_co_name_dict[user_id]
                if user_id in load_num_user_id_dict[i]:
                    df = user_df[user_df['CONS_NO'] == user_id]
                    if (df.iloc[0, 1] == '2015-01-01') & (len(df) > 700):
                        csv_file = trade_type_path + '%s_%s.csv' % (co_name, user_id)
                        df.to_csv(csv_file, index=False)
                    del df
        del user_df

    print('Outputting npy...')
    dict_path = base_path + 'output/dict/'
    if not os.path.exists(dict_path):
        os.makedirs(dict_path)
    np.save(dict_path + 'user_id_not_found_list.npy', user_id_not_found_list)
    np.save(dict_path + 'co_name_user_id_dict.npy', co_name_user_id_dict)
    np.save(dict_path + 'user_id_co_name_dict.npy', user_id_co_name_dict)
    np.save(dict_path + 'trade_type_user_id_in_nanjing_dict.npy', trade_type_user_id_in_nanjing_dict)
