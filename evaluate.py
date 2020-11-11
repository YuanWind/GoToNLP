# -*- coding: utf-8 -*-
# @Time    : 2020/10/31
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : evaluate.py

def calc_macro_f1(true,pred):
    labels = set(true)
    res = []
    for label in labels:
        TP = 0
        FP = 0
        FN = 0
        for idx, lbl in enumerate(true):
            if lbl == label and pred[idx] == label:
                TP += 1
            elif lbl != label and pred[idx] == label:
                FP += 1
            elif lbl == label and pred[idx] != label:
                FN += 1
        precision = 0
        recall = 0
        f1 = 0
        if TP!=0:
            precision=TP/(TP+FP)
            recall=TP/(TP+FN)
            f1=2*precision*recall/(precision+recall)
        res.append([precision, recall, f1])
    P,R,F1=0,0,0
    for i in res:
        P+=i[0]
        R+=i[1]
        F1+=i[2]
    P,R,F1=P/len(res),R/len(res),F1/len(res)
    return P,R,F1

def calc_micro_f1(true,pred):
    labels=set(true)
    res=[]
    for label in labels:
        TP=0
        FP=0
        FN=0
        for idx,lbl in enumerate(true):
            if lbl==label and pred[idx]==label:
                TP+=1
            elif lbl!=label and pred[idx]==label:
                FP+=1
            elif lbl==label and pred[idx]!=label:
                FN+=1
        res.append([TP,FP,FN])
    TP,FP,FN=0,0,0
    for i in res:
        TP+=i[0]
        FP+=i[1]
        FN+=i[2]
    P=TP/(TP+FP)
    R=TP/(TP+FN)
    F1=2*P*R/(P+R)
    return P,R,F1

def calc_acc(true,pred):
    total=0
    for idx,label in enumerate(true):
        if label==pred[idx]:
            total+=1
    return total/len(true)

# P,R,F1=calc_micro_f1([1,2,3],[1,1,3])
# print(P,R,F1)
#
# P,R,F1=calc_macro_f1([1,2,3],[1,1,3])
# print(P,R,F1)