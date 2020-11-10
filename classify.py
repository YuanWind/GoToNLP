# -*- coding: utf-8 -*-
# @Time    : 2020/10/31
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : classify.py
import torch
from torch import optim, nn
from BiLSTM import BiLSTM
from evaluate import calc_micro_f1, calc_acc
from utils import load_pkl_data, get_time, dump_pkl_data, get_batches,padding_data
import torch.nn.functional as F


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

batch_size=16
train_word_id=load_pkl_data('data/hotel_word_id(train).pkl')
train_id_word=load_pkl_data('data/hotel_id_word(train).pkl')
label_id=load_pkl_data('data/hotel_label_id(train).pkl')
train_data=load_pkl_data('data/hotel_train.pkl') #13913
dev_data=load_pkl_data('data/hotel_dev.pkl') #4372
test_data=load_pkl_data('data/hotel_test.pkl') #2150
train_X,train_y=get_X_y(train_word_id,train_data,label_id)

test_X,test_y=get_X_y(train_word_id,test_data,label_id)
model=BiLSTM(train_word_id, label_id,max_len=64)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-8)
for epoch in range(5):
    step=0
    for X,y in get_batches(train_X,train_y,batch_size=batch_size,shuffle=False):
        X, true_len = padding_data(X, max_seqlen=64, padding_value=0)
        X = torch.LongTensor(X)
        y = torch.LongTensor(y)
        true_len = torch.LongTensor(true_len)
        model.train()
        model.zero_grad()
        probs = model(X,true_len)
        log_probs = torch.log(probs)  # 由于我的lstm的输出已经进行了softmax，所以此处只用进行 log 和
        loss = F.nll_loss(log_probs, y)  #
        step+=1
        if step % 10 == 0:
            _, pred = torch.max(probs, dim=1)
            pred = pred.view(-1).data.numpy()
            acc=calc_acc(y,pred)
            time_dic = get_time()
            time_str = "[{}-{:0>2d}-{:0>2d} {:0>2d}:{:0>2d}:{:0>2d}]".format(time_dic['year'], time_dic['month'],
                                                                             time_dic['day'], \
                                                                             time_dic['hour'], time_dic['min'],
                                                                             time_dic['sec'])
            log = time_str + " Epoch {} step [{}] acc: {:.2f} loss: {:.6f}".format(epoch, step, acc,float(loss.data))
            print(log)
        loss.backward()
        optimizer.step()
    label=[]
    pred_label=[]
    for X,y in get_batches(test_X,test_y,batch_size=batch_size,shuffle=False):
        X, true_len = padding_data(X, max_seqlen=64, padding_value=0)
        X = torch.LongTensor(X)
        y = torch.LongTensor(y)
        true_len = torch.LongTensor(true_len)
        model.eval()

        probs = model(X, true_len)
        _, pred = torch.max(probs, dim=1)
        pred = pred.view(-1).data.numpy()
        label.extend(y)
        pred_label.extend(list(pred))
    p,r,f1=calc_micro_f1(label, pred_label)
    dump_pkl_data(label,'res/label_'+str(epoch)+'.pkl')
    dump_pkl_data(pred_label,'res/pred_label_'+str(epoch)+'.pkl')
    print('Epoch:{},micro_p,micro_r,micro_f1:{},{},{}'.format(epoch,p,r,f1))
    # print(calc_macro_f1(dev_label_id,list(pred)))