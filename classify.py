# -*- coding: utf-8 -*-
# @Time    : 2020/10/31
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : classify.py
import os
import random
import torch
import time
from torch import optim
from BiLSTM import BiLSTM
from evaluate import calc_micro_f1, calc_acc, calc_macro_f1
from utils import load_pkl_data, get_time, returnDevice, get_batches_by_len,get_batches_by_padding
import torch.nn.functional as F

class trainer():
    def __init__(self,opts):

        random.seed(opts.seed)
        torch.manual_seed(opts.seed)

        self.opts=opts
        self.device = returnDevice(opts.cuda)
        self.lr = opts.lr
        self.lr_decay_rate = opts.lr_decay_rate
        self.batch_size = opts.batch_size
        self.max_len = opts.max_len
        self.epochs = opts.epochs
        self.print_every_step = opts.print_every_step
        self.early_stop = opts.early_stop  # 为0时不提前结束
        self.lr_decay_every = opts.lr_decay_every
        self.weight_decay = opts.weight_decay
        self.shuffle=opts.shuffle
        self.best_model_name=''
        self.vocab = load_pkl_data(opts.vocab_path)
        self.label_id = load_pkl_data(opts.label_id_path)

        self.best_score = 0
        self.best_score_epoch = 0

        self.model = self.get_model()
        self.optimizer = self.get_optim(opts.optims)

    def get_X_y(self,vocab,data,label_id):
        X=[]
        y=[]
        for one_data in data:
            label = list(one_data.keys())[0]
            y.append(label_id.get(label))
            word_list = one_data.get(label)
            one_x=[]
            for i in word_list:
                if i in vocab.keys():
                    one_x.append(vocab[i])
                else:
                    one_x.append(vocab['unknow'])
            X.append(one_x)
        return X,y

    def get_optim(self,optims):
        if optims=='Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optims=='SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def get_model(self):
        model=BiLSTM(self.opts,self.vocab, self.label_id).to(self.device)
        return model

    def save_model(self, model_name):
        """
        保存训练好的模型参数。
        :param model_path:
        :return:
        """
        if not os.path.isdir(self.opts.model_dir):
            os.mkdir(self.opts.model_dir)
        torch.save(self.model.state_dict(), os.path.join(self.opts.model_dir,model_name))
        print('model saved succeed in: ' + self.opts.model_dir+model_name)
    def eval_model(self,type,data_X,data_y,model_name=''):
        """
        评测结果
        :param type: 'dev'--验证集，'test'--测试集，值为'test'时会加载保存在 dev 上最好的模型参数。
        :param data_X:
        :param data_y:
        :param model_path: 'dev'上最好的模型的保存地址
        :return: micro_f1,macro_f1,acc
        """
        total_loss = torch.FloatTensor([0])
        init_num = 0
        label = []
        pred_label = []
        if type=='dev':
            self.model.eval()
            prefix='dev-->'
        elif type=='test':
            prefix='model_name:{}\n '.format(self.best_model_name)
            self.model.load_state_dict(torch.load(os.path.join(self.opts.model_dir,model_name)))  #model.load_state_dict()函数把加载的权重复制到模型的权重中去
            self.model.eval()
        else:
            raise RuntimeError('type wrong!')
        for X, y,true_len in get_batches_by_len(data_X,data_y, batch_size=self.batch_size, shuffle=self.shuffle):
            init_num+=1
            label.extend(y)
            # X, true_len = padding_data(X, max_seqlen=max_len, padding_value=0)
            X = torch.LongTensor(X).to(self.device)
            y = torch.LongTensor(y).to(self.device)
            true_len = torch.LongTensor(true_len).to(self.device)
            probs = self.model(X, true_len,true_len[0])
            _, pred = torch.max(probs, dim=1)
            log_probs = torch.log(probs)  # 由于我的lstm的输出已经进行了softmax，所以此处只用进行 log 和
            loss = F.nll_loss(log_probs, y)  #
            loss = loss.cpu()
            total_loss += loss
            pred = pred.view(-1).cpu().data.numpy()
            pred_label.extend(list(pred))

        mi_p, mi_r, micro_f1 = calc_micro_f1(label, pred_label)
        ma_p, ma_r, macro_f1 = calc_macro_f1(label, pred_label)
        acc = calc_acc(label, pred_label)
        log=prefix+'micro_f1={:.4f},macro_f1={:.4f},acc={:.4f},loss={:.4f}'.format(micro_f1, macro_f1, acc*100,float(total_loss/init_num))
        print(log)
        return micro_f1,macro_f1,acc


    def adjust_learning_rate(self,optim, lr_decay_rate):
        """
        调整学习率
        :param optim: 优化器对象
        :param lr_decay_rate: lr=lr*(1-lr_decay_rate)
        """
        for param_group in optim.param_groups:
            param_group['lr'] = param_group['lr'] * (1 - lr_decay_rate)
            self.lr=param_group['lr']

    def train(self):
        start_ = time.time()
        train_data = load_pkl_data(self.opts.train_data_pkl)  # 13913
        dev_data = load_pkl_data(self.opts.dev_data_pkl)  # 4372
        train_X, train_y = self.get_X_y(self.vocab, train_data, self.label_id)
        dev_X, dev_y = self.get_X_y(self.vocab, dev_data, self.label_id)
        for epoch in range(self.epochs):
            step = 0
            total_loss=torch.FloatTensor([0])
            init_num=0
            true_y,pred_y=[],[]
            for X, y,true_len in get_batches_by_len(train_X,train_y,batch_size=self.batch_size,shuffle=self.shuffle):
                init_num+=1
                true_y.extend(y)
                X = torch.LongTensor(X).to(self.device)
                y = torch.LongTensor(y).to(self.device)
                true_len = torch.LongTensor(true_len).to(self.device)
                self.model.train()
                self.model.zero_grad()
                probs = self.model(X,true_len,true_len[0])
                _, pred = torch.max(probs, dim=1)
                pred = pred.view(-1).cpu().data.numpy()
                pred_y.extend(list(pred))

                log_probs = torch.log(probs)  # 由于我的lstm的输出已经进行了softmax，所以此处只用进行 log 和
                loss = F.nll_loss(log_probs, y)  #
                loss=loss.cpu()
                total_loss+=loss
                step+=1

                if step % self.print_every_step == 0:
                    avg_loss=total_loss/init_num
                    acc=calc_acc(true_y,pred_y)
                    time_dic = get_time()
                    time_str = "[{}-{:0>2d}-{:0>2d} {:0>2d}:{:0>2d}:{:0>2d}]".format(time_dic['year'], time_dic['month'],time_dic['day'],time_dic['hour'], time_dic['min'],time_dic['sec'])
                    log = time_str + " Epoch {} step [{}] acc: {:.2f} loss: {:.6f}".format(epoch, step, acc,float(avg_loss))
                    print(log)
                    total_loss = torch.FloatTensor([0])
                    init_num = 0
                    true_y, pred_y = [], []
                loss.backward()
                self.optimizer.step()
            micro_f1,macro_f1,acc=self.eval_model('dev',dev_X,dev_y)
            if acc>self.best_score:
                self.early_stop_count = 0
                self.lr_decay_count = 0
                self.best_score = acc
                time_dic = get_time()
                time_str = "[{}-{:0>2d}-{:0>2d} {:0>2d}:{:0>2d}:{:0>2d}]".format(time_dic['year'], time_dic['month'],time_dic['day'],time_dic['hour'], time_dic['min'],time_dic['sec'])
                model_name =time_str+'_epoch_'+str(epoch)+'.pt'
                self.save_model(model_name)
                self.best_model_name=model_name
                log = "Update! best dev acc: {:.2f}%".format(self.best_score*100)
                print(log)
            else:
                self.early_stop_count += 1
                self.lr_decay_count += 1
            if self.early_stop!=0 and self.early_stop_count == self.early_stop:
                log = "{} epochs passed, has not improved, so early stop the train!".format(self.early_stop_count)
                print(log)
                break
            if self.lr_decay_count == self.lr_decay_every:
                self.lr_decay_count = 0
                self.adjust_learning_rate(self.optimizer, self.lr_decay_rate)
                log = "{} epochs passed, has not improved, so adjust lr to {}".format(self.early_stop_count, self.lr)
                print(log)
        end_ = time.time()
        print('训练模型共用时：{:.3f}s'.format(end_ - start_))
    def model_result(self):
        dev_data = load_pkl_data(self.opts.dev_data_pkl)  # 4372
        test_data = load_pkl_data(self.opts.test_data_pkl)  # 2150
        dev_X, dev_y = self.get_X_y(self.vocab, dev_data, self.label_id)
        test_X, test_y = self.get_X_y(self.vocab, test_data, self.label_id)
        print('加载dev上效果最好的模型评测test数据集：')
        self.eval_model('test',dev_X,dev_y,self.best_model_name)
        self.eval_model('test',test_X,test_y,self.best_model_name)
