# -*- coding: utf-8 -*-
# @Time    : 2020/10/31
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : classify.py
import os

import torch
import time
from torch import optim
from BiLSTM import BiLSTM
from evaluate import calc_micro_f1, calc_acc, calc_macro_f1
from utils import load_pkl_data, get_time, dump_pkl_data, get_batches, padding_data, returnDevice
import torch.nn.functional as F
start_=time.time()
device = returnDevice()


def get_X_y(vocab,data,label_id):
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

def get_test(train_word_id,label_id,data,batch_size):
    train_id=[]
    train_id_len=[]
    train_label_id=[]
    for one_data in data[batch_size*64:64*(batch_size+1)]:
        label = list(one_data.keys())[0]
        train_label_id.append(label_id.get(label))
        text_list = one_data.get(label)
        one_id=[]
        for i in text_list:
            if i in train_word_id.keys():
                one_id.append(train_word_id[i])
            else:
                one_id.append(train_word_id['unknow'])
        if len(one_id)>=64:
            one_id=one_id[:64]
            train_id_len.append(64)
        else:
            train_id_len.append(len(one_id))
            for j in range(64-len(one_id)):
                one_id.append(0)
        train_id.append(one_id)
    return train_id,train_label_id,train_id_len

train_word_id=load_pkl_data('data/hotel_word_id(train).pkl')
train_id_word=load_pkl_data('data/hotel_id_word(train).pkl')
label_id=load_pkl_data('data/hotel_label_id(train).pkl')
train_data=load_pkl_data('data/hotel_train.pkl') #13913
dev_data=load_pkl_data('data/hotel_dev.pkl') #4372
test_data=load_pkl_data('data/hotel_test.pkl') #2150
train_X,train_y=get_X_y(train_word_id,train_data,label_id)
dev_X,dev_y=get_X_y(train_word_id,dev_data,label_id)
test_X,test_y=get_X_y(train_word_id,test_data,label_id)

lr=0.0001
lr_decay_rate=0.1
batch_size=8
max_len=64
epochs=10
print_every_step=100

model=BiLSTM(train_word_id, label_id,max_len=max_len).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)



def eval(type,epoch):
    """
    评测结果
    :param type: 'dev'--验证集，'test'--测试集，值为'test'时会加载保存在 dev 上最好的模型参数。
    :param epoch: 'dev'上最好的模型的epoch
    :return: micro_f1,macro_f1,acc
    """
    total_loss = torch.FloatTensor([0])
    init_num = 0
    label = []
    pred_label = []
    if type=='dev':
        data_X,data_y=test_X,test_y
        model.eval()
    elif type=='test':
        data_X,data_y=test_X,test_y
        model.load_state_dict(torch.load('saved_models/epoch_'+str(epoch)+'.pt'))  #model.load_state_dict()函数把加载的权重复制到模型的权重中去
        model.eval()
    else:
        raise RuntimeError('type wrong!')
    for X, y in get_batches(data_X,data_y, batch_size=batch_size, shuffle=False):
        init_num+=1
        X, true_len = padding_data(X, max_seqlen=max_len, padding_value=0)
        X = torch.LongTensor(X).to(device)
        label.extend(y)
        y = torch.LongTensor(y).to(device)
        true_len = torch.LongTensor(true_len).to(device)
        probs = model(X, true_len)
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

    print('{}--> epoch:{}--> micro_f1={:.4f},macro_f1={:.4f},acc={:.4f},loss={:.4f}'.format(type,epoch, micro_f1, macro_f1, acc*100,float(total_loss/init_num)))
    return micro_f1,macro_f1,acc

def save_model(epoch,model_dir):
    """
    保存训练好的模型参数。
    :param epoch: 第几个epoch，用于命名
    :param model_dir: 模型存放的目录
    """
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    fname =os.path.join(model_dir,'epoch_'+str(epoch) + '.pt')
    torch.save(model.state_dict(), fname)
    print('model saved succeed in ' + fname)
def adjust_learning_rate(optim, lr_decay_rate):
    """
    调整学习率
    :param optim: 优化器对象
    :param lr_decay_rate: lr=lr*(1-lr_decay_rate)
    """
    for param_group in optim.param_groups:
        param_group['lr'] = param_group['lr'] * (1 - lr_decay_rate)
        global lr
        lr=param_group['lr']


best_score=0
best_score_epoch=0
early_stop=0 #为0时不提前结束
lr_decay_every=3

for epoch in range(epochs):
    step = 0
    total_loss=torch.FloatTensor([0])
    init_num=0
    true_y,pred_y=[],[]
    for X,y in get_batches(train_X,train_y,batch_size=batch_size,shuffle=False):
        init_num+=1
        true_y.extend(y)

        X, true_len = padding_data(X, max_seqlen=max_len, padding_value=0)
        X = torch.LongTensor(X).to(device)
        y = torch.LongTensor(y).to(device)
        true_len = torch.LongTensor(true_len).to(device)
        model.train()
        model.zero_grad()
        probs = model(X,true_len)
        _, pred = torch.max(probs, dim=1)
        pred = pred.view(-1).cpu().data.numpy()
        pred_y.extend(list(pred))

        log_probs = torch.log(probs)  # 由于我的lstm的输出已经进行了softmax，所以此处只用进行 log 和
        loss = F.nll_loss(log_probs, y)  #
        loss=loss.cpu()
        total_loss+=loss
        step+=1

        if step % print_every_step == 0:
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
        optimizer.step()
    micro_f1,macro_f1,acc=eval('dev',epoch)
    if acc>best_score:
        early_stop_count = 0
        lr_decay_count = 0
        best_score = acc
        best_score_epoch = epoch
        save_model(epoch,'saved_models')
        log = "Update! best dev acc: {:.2f}%".format(best_score*100)
        print(log)
    else:
        early_stop_count += 1
        lr_decay_count += 1
    if early_stop!=0 and early_stop_count == early_stop:
        log = "{} epochs passed, has not improved, so early stop the train!".format(early_stop_count)
        break
    if lr_decay_count == lr_decay_every:
        lr_decay_count = 0
        adjust_learning_rate(optimizer, lr_decay_rate)
        log = "{} epochs passed, has not improved, so adjust lr to {}".format(early_stop_count, lr)
        print(log)

print('加载dev上效果最好的模型评测test数据集：')
eval('test',best_score_epoch)
end_=time.time()
print('共用时：{:.3f}s'.format(end_-start_))