# -*- coding: utf-8 -*-
# @Time    : 2020/11/13
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : main.py
from classify import trainer
from opts import opts
# export CUDA_VISIBLE_DEVICES=5

# processor('hotel')
# processor('phone')
from utils import build_dataset, dump_pkl_data, load_pkl_data

opts=opts()
train_data,dev_data,test_data=build_dataset(opts,use_word=False)
dump_pkl_data((train_data,dev_data,test_data),opts.data_path+'/data/data_train_dev_test.pkl')
# train_data,dev_data,test_data=load_pkl_data(opts.data_path+'/data/data_train_dev_test.pkl')
trainer=trainer(opts)
trainer.train(train_data,dev_data)
trainer.model_result(test_data)