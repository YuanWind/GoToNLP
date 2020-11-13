# -*- coding: utf-8 -*-
# @Time    : 2020/11/5
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : embedding.py
import numpy as np
import torch
from tqdm import tqdm


def load_predtrained_emb_avg(words_dic, path='pre_embed_vec/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.txt',padding=False, save=''):
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

    # embedding = np.zeros((word_size, embeding_dim)) #这里为何又重新初始化了呢
    avg_col = np.sum(embedding, axis=0) / len(in_word_list) #按列求平均值
    for i in range(word_size):
        if i not in in_word_list:
            if not padding:
                embedding[i] = avg_col
            elif padID in in_word_list:
                embedding[i] = embedding[padID]


    print("load done")
    print("{} words, {} in_words    {} OOV!".format(len(words_dic), len(in_word_list), len(words_dic) - len(in_word_list)))
    '''
        save
    '''
    if save != '':
        with open(save, 'a') as f:
            for line in lines:
                line = line.strip()
                f.write(line+'\n')
            print("save successful! path=", save)
    return torch.from_numpy(embedding).float()
