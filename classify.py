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
from tqdm import tqdm
from log import Log
from BiLSTM import BiLSTM
from evaluate import calc_micro_f1, calc_acc, calc_macro_f1
from utils import load_pkl_data, get_time, returnDevice, get_batches_by_len, init_network
import torch.nn.functional as F

# -*- coding: utf-8 -*-
# @Time    : 2020/10/31
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : classify.py
class trainer():
    def __init__(self,opts):

        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        self.opts=opts
        self.log=Log(opts)
        self.device = returnDevice(opts.cuda,opts.gpu_number)
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
        """
        生成 x 和 y。
        :param vocab: 构建的词表
        :param data:
        :param label_id: 用于得到label对应的id
        :return:
        """
        X=[]
        y=[]
        for one_data in data:
            label = list(one_data.keys())[0]
            y.append(int(label))
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
        init_network(model) # init_netword 用于初始化模型内部参数，一般用xavier方法来初始化效果较好。
        return model

    def save_model(self, model_name):
        """
        保存训练好的模型参数。
        :param model_path:
        :return:
        """
        if not os.path.isdir(self.opts.model_dir):
            os.mkdir(self.opts.model_dir)
        path=os.path.join(self.opts.model_dir,model_name)
        torch.save(self.model.state_dict(),path)
        log='model saved succeed in: {}/{}'.format( self.opts.model_dir,model_name)
        self.log.fprint_log(log)
        print(log)
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
        for X, y,true_len in tqdm(get_batches_by_len(data_X,data_y, batch_size=self.batch_size)):
            init_num+=1
            label.extend(y)
            # X, true_len = padding_data(X, max_seqlen=max_len, padding_value=0)
            X = torch.LongTensor(X).to(self.device)
            y = torch.LongTensor(y).to(self.device)
            true_len = torch.LongTensor(true_len).to(self.device)
            probs = self.model(X, true_len,true_len[0])
            _, pred = torch.max(probs, dim=1)
            # log_probs = torch.log(probs)  # 由于我的lstm的输出已经进行了softmax，所以此处只用进行 log 和
            loss = F.nll_loss(probs, y)  #
            loss = loss.cpu()
            total_loss += loss
            pred = pred.view(-1).cpu().data.numpy()
            pred_label.extend(list(pred))

        mi_p, mi_r, micro_f1 = calc_micro_f1(label, pred_label)
        ma_p, ma_r, macro_f1 = calc_macro_f1(label, pred_label)
        acc = calc_acc(label, pred_label)
        log=prefix+'micro_f1={:.4f},macro_f1={:.4f},acc={:.4f},loss={:.4f}\n'.format(micro_f1, macro_f1, acc*100,float(total_loss/init_num))
        self.log.fprint_log(log)
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

    def train(self,train_data,dev_data):
        start_ = time.time()
        train_X, train_y = self.get_X_y(self.vocab, train_data, self.label_id)
        dev_X, dev_y = self.get_X_y(self.vocab, dev_data, self.label_id)
        for epoch in range(self.epochs):
            # 一个epoch会把整个训练集的数据都用一遍来更新模型参数，一般需要跑多个epoch，以使模型更准确。
            step = 0 # 每一个step就是训练一个batch的数据
            total_loss=torch.FloatTensor([0]) # 用来记录总的loss值
            init_num=0 # 用来计算平均loss
            true_y,pred_y=[],[]
            for X, y,true_len in get_batches_by_len(train_X,train_y,batch_size=self.batch_size):
                # 每一个循环得到一个batch的数据，用来训练模型
                init_num+=1
                true_y.extend(y)
                X = torch.LongTensor(X).to(self.device)
                y = torch.LongTensor(y).to(self.device)
                true_len = torch.LongTensor(true_len).to(self.device)
                self.model.train() # 开始训练模型
                self.model.zero_grad() # 当网络参量进行反馈时，梯度是被积累的而不是被替换掉；但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了
                probs = self.model(X,true_len,true_len[0]) # 这里其实就是调用了forword函数来前向传播计算结果，得到log_softmax
                _, pred = torch.max(probs, dim=1)# 取每一行最大值所在的位置作为该行句子预测的类别。由于我们输入的是一个batch的数据，所以会有batch个结果。
                pred = pred.view(-1).cpu().data.numpy() # 将其变成numpy后转成list最为预测的结果
                pred_y.extend(list(pred))
                loss = F.nll_loss(probs, y)  # 计算该批数据的平均损失值。
                loss=loss.cpu()
                total_loss+=loss
                step+=1

                if step % self.print_every_step == 0: # 每隔多少步打印一次步平均loss值，并在当前这么多批的数据上测试模型在训练集数据上的效果。
                    avg_loss=total_loss/init_num # 计算平均loss（步）
                    acc=calc_acc(true_y,pred_y) # 计算训练集上的效果（acc)
                    time_dic = get_time()
                    time_str = "[{}-{:0>2d}-{:0>2d} {:0>2d}-{:0>2d}-{:0>2d}]".format(time_dic['year'], time_dic['month'],time_dic['day'],time_dic['hour'], time_dic['min'],time_dic['sec'])
                    log = time_str + " Epoch {} step [{}] acc: {:.2f} loss: {:.6f}".format(epoch, step, acc,float(avg_loss))
                    self.log.fprint_log(log) # 将输出写到日志文件中
                    print(log)
                    total_loss = torch.FloatTensor([0])
                    init_num = 0
                    true_y, pred_y = [], []  # 重新初始化
                loss.backward() # 向后传播梯度求解
                self.optimizer.step() #更新模型的所有参数
            micro_f1,macro_f1,acc=self.eval_model('dev',dev_X,dev_y) # 每一个epoch跑完之后计算一下模型在验证集上的效果。需要保存效果最好的模型的参数。
            if acc>self.best_score: #保存效果最好的模型的参数，这里使用acc来判断效果
                self.early_stop_count = 0 # 训练提前停止计数初始化
                self.lr_decay_count = 0 # 学习率衰减计数初始化
                self.best_score = acc
                time_dic = get_time()
                time_str = "[{}-{:0>2d}-{:0>2d} {:0>2d}-{:0>2d}-{:0>2d}]".format(time_dic['year'], time_dic['month'],time_dic['day'],time_dic['hour'], time_dic['min'],time_dic['sec'])
                model_name =time_str+'_epoch_'+str(epoch)+'.pt'
                self.save_model(model_name) # 保存模型参数
                self.best_model_name=model_name # 记录最好模型保存的名字
                log = "Update! best dev acc: {:.2f}%".format(self.best_score*100)
                self.log.fprint_log(log)
                print(log)
            else:
                self.early_stop_count += 1 # 如果过了一个epoch效果没有提升，就要计数了。
                self.lr_decay_count += 1
            if self.early_stop!=0 and self.early_stop_count == self.early_stop:
                # early_stop=0代表不提前停止训练。否则的话就是过了early_stop个epoch后效果一直没有提升就停止训练。
                log = "{} epochs passed, has not improved, so early stop the train!".format(self.early_stop_count)
                self.log.fprint_log(log)
                print(log)
                break
            if self.lr_decay_count == self.lr_decay_every:
                # 过了 lr_decay_count个epoch效果还没有提升的话就降低学习率。
                self.lr_decay_count = 0
                self.adjust_learning_rate(self.optimizer, self.lr_decay_rate) # 调整学习率
                log = "{} epochs passed, has not improved, so adjust lr to {}".format(self.early_stop_count, self.lr)
                self.log.fprint_log(log)
                print(log)
        end_ = time.time()
        log='训练模型共用时：{:.3f}s'.format(end_ - start_)
        self.log.fprint_log(log)
        print(log)
    def model_result(self,test_data):
        """
        在测试集上评测模型结果
        :param test_data:
        :return:
        """
        test_X, test_y = self.get_X_y(self.vocab, test_data, self.label_id)
        log='加载dev上效果最好的模型评测test数据集：'
        self.log.fprint_log(log)
        print(log)
        self.eval_model('test',test_X,test_y,self.best_model_name)