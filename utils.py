# -*- coding: utf-8 -*-
# @Time    : 2020/11/9
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : utils.py
import re
import time
import numpy as np
import pickle
import torch
from tqdm import tqdm
import pandas
import torch.nn as nn

def dump_pkl_data(data, f_name):
    with open(f_name, 'wb') as f:
        pickle.dump(data, f)

def load_pkl_data(f_name):
    with open(f_name, 'rb') as f:
        return pickle.load(f)
# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass
def load_predtrained_emb_avg(words_dic, path='pre_embed_vec/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.txt',padding=False):
    """
    读取预训练词向量
    :param words_dic: 传入的词：dict(word,id) ,id一定要从0开始，否则下边构建np.array的时候会出问题
    :param path: 预训练模型下载地址：https://github.com/Embedding/Chinese-Word-Vectors
    :param padding:是否进行padding处理
    :param save:是否保存结果到文件
    :return:提取的预训练词向量
    """

    print("start to load predtrained embedding...")
    if padding:
        padID = words_dic['unknow']
    embeding_dim = -1
    with open(path, encoding='utf-8') as f:
        line=f.readline()# 第一行是相关信息，所以要接着读取一行
        line=f.readline()
        line = line.strip().split(" ")
        if len(line) <= 1:
            print("load_predtrained_embedding text is wrong!  -> len(line) <= 1")
        else:
            embeding_dim = len(line) - 1
    word_size = len(words_dic)
    print("The word size is ", word_size)
    print("The dim of predtrained embedding is ", embeding_dim, "\n")

    lines = []
    embedding = np.zeros((word_size, embeding_dim))
    in_word_list = []

    with open(path, encoding='utf-8') as f:
        i=0
        for line in tqdm(f.readlines()):
            if i==0: #跳过第一行
                i+=1
                continue
            rawline = line
            line = line.strip().split(' ')
            index = words_dic.get(line[0])
            if index:
                lines.append(rawline)
                vector = np.array(line[1:], dtype='float32')
                embedding[index] = vector
                in_word_list.append(index)

    avg_col = np.sum(embedding, axis=0) / len(in_word_list) #按列求平均值
    for i in range(word_size):
        if i not in in_word_list:
            if not padding:
                embedding[i] = avg_col
            elif padID in in_word_list:
                embedding[i] = embedding[padID]


    print("load done")
    print("{} words, {} in_words    {} OOV!".format(len(words_dic), len(in_word_list), len(words_dic) - len(in_word_list)))

    return torch.from_numpy(embedding).float()

def cut_str_by_len(str_, length):
    """
    将字符串str_按长度len拆分: ('abcdefg',3)-->['abc','def','g']
    :param str_: 待拆分的字符串
    :param length: 拆分长度
    :return: 拆分后的字符串列表
    """
    return [str_[i:i + length] for i in range(0, len(str_), length)]

def cut_sent(sent):
    import jieba
    return list(jieba.cut(sent))

def preprocess(sen):
    """
    用来清洗评价数据，包括统一为小写，删除文本中的空格,换行，句号，问号，感叹号以及标签信息，将繁体转换为简体，最后利用jieba库进行tokenization操作
    :param sen: 待处理的字符串
    :return: list,处理并分词后的列表
    """

    import zhconv
    import jieba
    # import hanlp
    # tokenizer = hanlp.load('LARGE_ALBERT_BASE')
    sen.lower()
    sen = sen.replace(' ', '')
    sen = sen.replace('\n', '')
    pattern = re.compile(r'(?<=<).+?(?=>)')  # https://blog.csdn.net/z1102252970/article/details/70739804
    str1 = pattern.sub('', sen)
    str1 = str1.replace('<>', '')
    cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")  # 匹配汉字，英文，数字
    str1 = zhconv.convert(str1, 'zh-cn')
    str1 = cop.sub('', str1)
    # tokens=tokenizer(str1)# hanlp问题：str1的长度不能超过126，多余的字符会被截断，所以需要对长字符串进行拆分
    # tokens=[]
    # if len(str1)>100:
    #     str1s=cut_str_by_len(str1,100)
    #     for split_text in str1s:
    #         tokens.extend(tokenizer(split_text))
    # else:tokens.extend(tokenizer(str1))
    tokens = jieba.cut(str1)
    return list(tokens)


def get_time():
    # tm_year=2018, tm_mon=10, tm_mday=28, tm_hour=10, tm_min=32, tm_sec=14, tm_wday=6, tm_yday=301, tm_isdst=0
    cur_time = time.localtime(time.time())

    dic = dict()
    dic['year'] = cur_time.tm_year
    dic['month'] = cur_time.tm_mon
    dic['day'] = cur_time.tm_mday
    dic['hour'] = cur_time.tm_hour
    dic['min'] = cur_time.tm_min
    dic['sec'] = cur_time.tm_sec

    return dic


def get_batches_by_padding(data_x, data_y, batch_size=16, shuffle=True,max_seqlen=64, padding_value=0):
    """
    批数据生成器
    :param data_x: 数据x
    :param data_y: 标签y
    :param batch_size:
    :param shuffle: 是否打乱顺序
    :return:
    """
    length = len(data_y)
    if shuffle:
        index = np.random.randint(0, length, length)
    else:
        index = list(range(length))
    start_idx = 0
    while start_idx < length:
        end_idx = min(length, start_idx + batch_size)
        excerpt = index[start_idx:end_idx]
        X, y = [], []
        for i in excerpt:
            X.append(data_x[i])
            y.append(data_y[i])
        X, true_len = padding_data(X, max_seqlen=max_seqlen, padding_value=padding_value)
        yield X, y,true_len
        start_idx += batch_size


def get_batches_by_len(data_x, data_y, batch_size=32):
    """
    批数据生成器
    :param data_x: 数据x
    :param data_y: 标签y
    :param batch_size:设置最大的batch_size
    :param shuffle: 是否打乱顺序
    :return:
    """
    seq_len={} #len_x---idxs
    for idx,x in enumerate(data_x):
        len_x=len(x)
        if len_x not in seq_len.keys():
            seq_len[len_x]=[idx]
        else:
            seq_len[len_x].append(idx)
    for len_x,idxs in seq_len.items():
        total_x=len(idxs)
        sum_x=0
        if total_x <= batch_size:
            X, y = [], []
            for idx in idxs:
                X.append(data_x[idx])
                y.append(data_y[idx])
            true_len = [len(X[0]) for _ in range(len(y))]
            yield X, y,true_len
        else:
            X, y = [], []
            for idx in idxs:
                X.append(data_x[idx])
                y.append(data_y[idx])
                sum_x+=1
                if sum_x==batch_size:
                    true_len = [len(X[0]) for _ in range(len(y))]
                    yield X, y,true_len
                    X, y = [], []
                    sum_x=0
            if len(y)!=0: # 如果batch_size=8,而idxs有18项，那么最后就会返回一个空的batch
                true_len = [len(X[0]) for _ in range(len(y))]
                yield X, y, true_len

def get_test_batches_by_len(data_x, data_idx, batch_size=32):
    """
    批数据生成器
    :param data_x: 数据x
    :param data_y: 标签y
    :param batch_size:设置最大的batch_size
    :param shuffle: 是否打乱顺序
    :return:
    """
    length = len(data_idx)
    print('count:',length)
    seq_len={} #len_x---idx
    for idx,x in enumerate(data_x):
        len_x=len(x)
        if len_x not in seq_len.keys():
            seq_len[len_x]=[idx]
        else:
            seq_len[len_x].append(idx)
    for len_x,idxs in seq_len.items():
        total_x=len(idxs)
        sum_x=0
        if total_x <= batch_size:
            X, true_idxs = [], []
            for idx in idxs:
                X.append(data_x[idx])
                true_idxs.append(data_idx[idx])
            true_len = [len(X[0]) for _ in range(len(true_idxs))]
            yield X, true_idxs,true_len
        else:
            X, true_idxs = [], []
            for idx in idxs:
                X.append(data_x[idx])
                true_idxs.append(data_idx[idx])
                sum_x+=1
                if sum_x==batch_size:
                    true_len = [len(X[0]) for _ in range(len(true_idxs))]
                    yield X, true_idxs,true_len
                    X, true_idxs = [], []
                    sum_x=0
            if len(true_idxs)!=0: # 如果batch_size=8,而idxs有18项，那么最后就会返回一个空的batch
                true_len = [len(X[0]) for _ in range(len(true_idxs))]
                yield X, true_idxs, true_len


def build_dataset(opts,use_word=True):
    train=pandas.read_csv(opts.train_data_path,sep='\t',names=['content','label'])
    dev=pandas.read_csv(opts.dev_data_path,sep='\t',names=['content','label'])
    test=pandas.read_csv(opts.test_data_path,sep='\t',names=['content','label'])
    print('空值统计--train_data:\n{}\n 空值统计--dev_data:\n{}\n空值统计--test_data:\n{}\n,有空值的行将被去掉！'.format(train.isnull().sum(),dev.isnull().sum(),test.isnull().sum()))
    # df.dropna(axis=0, how='any', inplace=True)
    # axis：0: 行操作（默认）1: 列操作how：any: 只要有空值就删除（默认）all:全部为空值才删除inplace：False: 返回新的数据集（默认）True: 在愿数据集上操作
    train.dropna(axis=0, how='any', inplace=True)
    dev.dropna(axis=0, how='any', inplace=True)
    test.dropna(axis=0, how='any', inplace=True)
    label2id={}
    id2label={}
    with open(opts.class_path,'r',encoding='utf-8') as f:
        for idx,cls in enumerate(f):
            label2id.update({cls:idx})
            id2label.update({idx:cls})
    dump_pkl_data(label2id,opts.label_id_path)
    dump_pkl_data(id2label,opts.id_label_path)
    if use_word:
        tokenizer = lambda x: cut_sent(x)  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    vocab={}
    for idx in train.index:
        content=train.loc[idx,'content']
        tokens=tokenizer(content)
        for token in tokens:
            if token not in vocab.keys():
                vocab.update({token:len(vocab)})
    vocab['unknow']=len(vocab)
    dump_pkl_data(vocab,opts.vocab_path)
    def gen_data(data,tokenizer):
        res=[]
        for idx in data.index:
            content = data.loc[idx, 'content']
            tokens = tokenizer(content)
            label=data.loc[idx,'label']
            res.append({label:tokens})
        return res
    train_data=gen_data(train,tokenizer)
    dev_data=gen_data(dev,tokenizer)
    test_data=gen_data(test,tokenizer)
    return train_data,dev_data,test_data


def padding_data(data_x, max_seqlen=0, padding_value=0, methord=''):
    """
    用 padding_value 按照某种方式 padding ，返回padding后的list
    :param data_x: padding前的list
    :param max_seqlen: 最大长度。如果不为0就以该值为最大长度进行padding
    :param padding_value: padding时填充的值
    :param methord: 如果max_seqlen=0,选择['max','avg'],以样本中最大长度或者平均长度padding
    :return: padding后的数据list; padding前各样本的真实长度
    """
    len_list = []
    true_len_list = []
    for data in data_x:
        len_list.append(len(data))
    max_len = max(len_list)
    avg_len = np.average(len_list)
    # min_len=min(true_len_list)
    new_data_x = []
    for data in data_x:
        if max_seqlen != 0:
            if len(data) < max_seqlen:
                true_len_list.append(len(data))
                data.extend([padding_value for i in range(max_seqlen - len(data))])
            else:
                data = data[:max_seqlen]
                true_len_list.append(len(data))

        elif methord == 'max':
            if len(data) < max_len:
                true_len_list.append(len(data))
                data.extend([padding_value for i in range(max_len - len(data))])
            else:
                data = data[:max_len]
                true_len_list.append(len(data))
        elif methord == 'avg':
            if len(data) < avg_len:
                true_len_list.append(len(data))
                data.extend([padding_value for i in range(avg_len - len(data))])
            else:
                data = data[:avg_len]
                true_len_list.append(len(data))

        new_data_x.append(data)

    return new_data_x, true_len_list


def autoDevice(obj, type='tensor', GPU_first=True):
    """
    根据是否有GPU来自动选择是否使用cuda来转换
    :param obj: 待转换的对象
    :param type: 'net' 或者 'tensor'，前者将模型转到GPU且成功时会给出提示，后者将tensor转到GPU无提示信息
    :param GPU_first: False时，不管是否有GPU都不使用GPU
    :return: 转换后的对象
    """
    gpus = [0]  # 使用哪几个GPU进行训练，这里选择0号GPU
    cuda_gpu = torch.cuda.is_available()  # 判断GPU是否存在可用
    if (cuda_gpu and GPU_first):
        if type == 'net':
            obj = torch.nn.DataParallel(obj, device_ids=gpus).cuda()  # 将模型转为cuda类型
            print('模型成功转到GPU上！----当前GPU：',torch.cuda.get_device_name(0))
        elif type == 'tensor':
            obj = obj.cuda()
        else:
            print('既不是net也不是tensor,直接返回！')
    return obj

def returnDevice(GPU_first=True,GPU_No=0):
    if GPU_first:
        device = torch.device("cuda:{}".format(GPU_No) if torch.cuda.is_available() else "cpu")
    else:
        device='cpu'
    return device