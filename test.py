import utils
import evaluate


def ConfusionMat(label, targets, pre):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for target in enumerate(targets):
        if target[1] == label & label == pre[int(target[0])]:
            TP += 1
        if target[1] == label & label != pre[int(target[0])]:
            FN += 1
        if target[1] != label & label != pre[int(target[0])]:
            TN += 1
        if target[1] != label & label == pre[int(target[0])]:
            FP += 1

    return TP, FN, FP, TN


def Precision(TP, FP):
    return TP / (TP + FP)


def Recall(TP, FN):
    return TP / (TP + FN)


def get_F1(P, R):
    return 2 * (P * R) / (P + R)


def get_Acc(TP, TN, total):
    return (TP + TN) / total


def performance(targets, pre):
    label_list = list(set(targets))
    TP = []
    FN = []
    FP = []
    TN = []
    P = []
    R = []
    F1 = []
    Acc = []
    for label in label_list:
        tp, fn, fp, tn = ConfusionMat(label, targets, pre)
        p = Precision(tp, fp)
        r = Recall(tp, fn)
        f1 = get_F1(p, r)
        acc = get_Acc(tp, tn, tp + tn + fp + fn)
        TP.append(tp)
        FN.append(fn)
        FP.append(fp)
        TN.append(tn)
        P.append(p)
        R.append(r)
        F1.append(f1)
        Acc.append(acc)

    micro_P = Precision(sum(TP), sum(FP))
    micro_R = Recall(sum(TP), sum(FN))
    micro_F1 = get_F1(micro_P, micro_R)

    macro_F1 = sum(F1) / len(F1)

    ave_acc = sum(Acc) / len(Acc)

    return micro_F1, macro_F1, ave_acc
# for i in range(10):
#     label=utils.load_pkl_data('data/label_'+str(i)+'.pkl')
#     pred_label=utils.load_pkl_data('data/pred_label_'+str(i)+'.pkl')
#     print('macro P:{},macro R:{},macro F1:{}'.format(*(evaluate.calc_macro_f1(label,pred_label))))
#     print('micro P:{},micro R:{},micro F1:{}'.format(*(evaluate.calc_micro_f1(label,pred_label))))
#     print('acc:{} '.format(evaluate.calc_acc(label,pred_label)))
#     print(performance(label,pred_label))


