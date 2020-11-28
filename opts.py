# -*- coding: utf-8 -*-
# @Time    : 2020/11/13
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : opts.py

class opts():
    def __init__(self):
        self.pre_embed_path = 'pre_embed_vec/sgns.sogou.char'
        self.log_dir = 'log'
        self.log_fname = 'log.txt'
        self.model_dir = 'saved_models'  # 模型保存和加载地址
        self.data_path='THUCNews'
        self.vocab_path= self.data_path+'/data/vocab.pkl'
        self.class_path = self.data_path+'/data/class.txt'
        self.train_data_path = self.data_path+'/data/train.txt'
        self.dev_data_path = self.data_path+'/data/dev.txt'
        self.test_data_path = self.data_path+'/data/test.txt'
        self.label_id_path=self.data_path+'/data/label2id.pkl'
        self.id_label_path=self.data_path+'/data/id2label.pkl'
        self.cuda = True
        self.gpu_number=0
        self.seed=88
        self.lr= 5e-5
        self.lr_decay_rate=0.1
        self.lr_decay_every=3
        self.weight_decay=1e-8
        self.batch_size=32
        self.max_len=64
        self.epochs=100
        self.print_every_step=1000
        self.early_stop=10 #为0时不提前结束
        self.shuffle=False

        self.optims='Adam'
        self.embedding_dim=300
        self.input_size = 300
        self.hidden_size = 300
        self.num_layers = 2
        self.dropout = 0.3
        self.batch_first=True
        self.bidirectional=True

