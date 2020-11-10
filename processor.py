# -*- coding: utf-8 -*-
# @Time    : 2020/10/29
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : processor.py
from tqdm import tqdm
from utils import dump_pkl_data, preprocess

def read_hotel_data(f_name):
    hotel_data = []  # list里的数据存储格式：{标签：评价}
    with open(f_name, 'r', encoding='utf-8') as f_hotel:
        hotel = f_hotel.readlines()
        hotel.insert(0, '\n')
        hotel_text, hotel_label = hotel[4::6], hotel[5::6]
        for idx, text in tqdm(enumerate(hotel_text)):
            hotel_data.append({int(hotel_label[idx].replace('\n', '')): preprocess(text)})
    return hotel_data

def read_phone_data(f_name):
    with open(f_name, 'r', encoding='utf-8') as f_phone:
        phone_data = []
        phone = f_phone.readlines()
        phone.insert(0, '\n')
        for idx, line in tqdm(enumerate(phone)):
            if idx + 1 < len(phone) and phone[idx + 1] == '\n': continue
            if line == '\n':
                i, text = 0, ''
            i += 1
            if i > 3:
                text += line
                if idx + 2 < len(phone) and phone[idx + 2] == '\n':
                    label = int(phone[idx + 1].replace('\n', ''))
                    phone_data.append({label: preprocess(text)})
    return phone_data

def statistics(data):
    # 这里统计词数，词频，标签数和OOV的情况，并建立词--ID，标签--ID
    word_cnt = {}  # word---词频
    word_id = {}  # word---id
    id_word={}    # id---word
    label_cnt = {}  # label---标签数
    label_id = {}  # label---id
    id_label = {}  # id---label
    wordID = 0
    labelID = 0
    for one_data in data:
        label = list(one_data.keys())[0]
        text_list = one_data.get(label)
        if label not in label_id.keys():
            label_id[label] = labelID
            id_label[labelID]=label
            labelID += 1
        if label not in label_cnt.keys():
            label_cnt[label] = 1
        else:
            label_cnt[label] += 1
        for word in text_list:
            if word not in word_id.keys():
                word_id[word] = wordID
                id_word[wordID]=word
                wordID += 1
            if word not in word_cnt.keys():
                word_cnt[word] = 1
            else:
                word_cnt[word] += 1
    wordscnt = len(word_id)
    print("词数：", wordscnt)
    #处理OOV，加入unknow
    word_id['unknow']=wordID
    id_word[wordID]='unknow'
    return word_id,id_word, word_cnt, label_id, id_label,label_cnt

def processor(dataType):
    # 读取原始数据, 转换成[{label:review},{label:review},....]的格式
    if dataType == 'hotel':
        train_data = read_hotel_data('data/' + dataType + '_train.txt')
        dev_data = read_hotel_data('data/' + dataType + '_dev.txt')
        test_data = read_hotel_data('data/' + dataType + '_test.txt')
    if dataType == 'phone':
        train_data = read_phone_data('data/' + dataType + '_train.txt')
        dev_data = read_phone_data('data/' + dataType + '_dev.txt')
        test_data = read_phone_data('data/' + dataType + '_test.txt')

    dump_pkl_data(train_data, 'data/' + dataType + '_train.pkl')
    dump_pkl_data(dev_data, 'data/' + dataType + '_dev.pkl')
    dump_pkl_data(test_data, 'data/' + dataType + '_test.pkl')
    # train_data = load_pkl_data('data/' + dataType + '_train.pkl')
    # dev_data = load_pkl_data('data/' + dataType + '_dev.pkl')
    # test_data = load_pkl_data('data/' + dataType + '_test.pkl')
    train_word_id,train_id_word, train_word_cnt,label_id, id_label, label_cnt= statistics(train_data)

    dump_pkl_data(train_word_id, 'data/'+dataType+'_word_id(train).pkl')
    dump_pkl_data(train_id_word, 'data/'+dataType+'_id_word(train).pkl')
    dump_pkl_data(label_id, 'data/'+dataType+'_label_id(train).pkl')
    dump_pkl_data(id_label, 'data/'+dataType+'_id_label(train).pkl')

# processor('hotel')
# processor('phone')
    # dev_oov_word = []
    # test_oov_word = []
    # for word in dev_data:
    #     if word not in train_word_id:
    #         dev_oov_word.append(word)
    # for word in test_data:
    #     if word not in train_word_id:
    #         test_oov_word.append(word)
    #
    # print(dataType+'验证集中OOV情况：', len(dev_oov_word), dev_oov_word)
    # print(dataType+'测试集中OOV情况：', len(test_oov_word), test_oov_word)
    # print(dataType+'共有标签数：', len(label_id), label_cnt)

#使用hanlp分词
# def processor1(str):
#     # 读取原始数据, 转换成[{label:review},{label:review},....]的格式
#     # if str=='hotel':
#     #     train_data = read_hotel_data('data/'+str+'_train.txt')
#     #     dev_data=read_hotel_data('data/'+str+'_dev.txt')
#     #     test_data=read_hotel_data('data/'+str+'_test.txt')
#
#     # dump_data(train_data,'data/'+str+'_train.pkl')
#     # dump_data(dev_data,'data/'+str+'_dev.pkl')
#     # dump_data(test_data,'data/'+str+'_test.pkl')
#     # if str == 'phone':
#     #     train_data = read_phone_data('data/' + str + '_train.txt')
#     #     dev_data = read_phone_data('data/' + str + '_dev.txt')
#     #     test_data = read_phone_data('data/' + str + '_test.txt')
#
#     # dump_data(train_data, 'data/' + str + '_train.pkl')
#     # dump_data(dev_data, 'data/' + str + '_dev.pkl')
#     # dump_data(test_data, 'data/' + str + '_test.pkl')
#     train_data = load_pkl_data('backup/' + str + '_train.pkl')
#     dev_data = load_pkl_data('backup/' + str + '_dev.pkl')
#     test_data = load_pkl_data('backup/' + str + '_test.pkl')
#     train_word_id, train_word_cnt, train_label_id, train_label_cnt = statistics(train_data)
#     dev_word_id, dev_word_cnt, dev_label_id, dev_label_cnt = statistics(dev_data)
#     test_word_id, test_word_cnt, test_label_id, test_label_cnt = statistics(test_data)
#     dev_oov_word = []
#     test_oov_word = []
#     for word in dev_word_id:
#         if word not in train_word_id:
#             dev_oov_word.append(word)
#     for word in test_word_id:
#         if word not in train_word_id:
#             test_oov_word.append(word)
#     print('验证集中OOV情况：', len(dev_oov_word), dev_oov_word)
#     print('测试集中OOV情况：', len(test_oov_word), test_oov_word)
#     print('训练集共有标签数：', len(train_label_id), train_label_cnt)
#     print('验证集共有标签数：', len(dev_label_id), dev_label_cnt)
#     print('测试集共有标签数：', len(test_label_id), test_label_cnt)
#
#
# processor1('hotel')
# processor1('phone')