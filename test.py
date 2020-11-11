import utils
import evaluate
for i in range(5):
    label=utils.load_pkl_data('res/label_'+str(i)+'.pkl')
    pred_label=utils.load_pkl_data('res/pred_label_'+str(i)+'.pkl')
    p1, r1, micro_f1 = evaluate.calc_micro_f1(label, pred_label)
    p2, r2, macro_f1 = evaluate.calc_macro_f1(label, pred_label)
    acc = evaluate.calc_acc(label, pred_label)
    print('micro f1:{:.6f},macro f1:{:.6f},acc:{:.6f}'.format(micro_f1,macro_f1,acc))


