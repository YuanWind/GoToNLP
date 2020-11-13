# -*- coding: utf-8 -*-
# @Time    : 2020/11/13
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : opts.py

class opts():
    def __init__(self):
        self.cuda = True
        self.seed=88
        self.lr=0.0001
        self.lr_decay_rate=0.1
        self.lr_decay_every=3
        self.weight_decay=1e-8
        self.batch_size=8
        self.max_len=64
        self.epochs=100
        self.print_every_step=100
        self.early_stop=10 #为0时不提前结束
        self.shuffle=False

        self.optims='Adam'
        self.embedding_dim=300
        self.input_size = 300
        self.hidden_size = 300
        self.num_layers = 2
        self.dropout = 0.1
        self.batch_first=True
        self.bidirectional=True

        self.model_dir='saved_models' # 模型保存和加载地址
        self.vocab_path='data/hotel_word_id(train).pkl' #
        self.label_id_path='data/hotel_label_id(train).pkl'

        self.train_data_pkl='data/hotel_train.pkl'
        self.dev_data_pkl='data/hotel_dev.pkl'
        self.test_data_pkl = 'data/hotel_test.pkl'