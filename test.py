import os

import torch
import time
from torch import optim
from BiLSTM import BiLSTM
from evaluate import calc_micro_f1, calc_acc, calc_macro_f1
from utils import load_pkl_data, get_time, dump_pkl_data, get_batches, padding_data, returnDevice
import torch.nn.functional as F
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

lr=0.001
lr_decay_rate=0.1
batch_size=64
max_len=64
epochs=10

model=BiLSTM(train_word_id, label_id,max_len=max_len).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
start_=time.time()


def eval(type='dev',epoch=0):
    total_loss = torch.FloatTensor([0])
    init_num = 0
    label = []
    pred_label = []
    if type=='dev':
        data_X,data_y=test_X,test_y
        model.load_state_dict(torch.load('saved_models/epoch_' + str(epoch) + '.pt'))  # model.load_state_dict()函数把加载的权重复制到模型的权重中去
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

    print('{},micro_f1={:.4f},macro_f1={:.4f},acc={:.4f},loss={:.4f}'.format(type, micro_f1, macro_f1, acc*100,float(total_loss/init_num)))
    return micro_f1,macro_f1,acc
eval('dev',0)