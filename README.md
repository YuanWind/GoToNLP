# 入门NLP项目—BiLSTM文本分类

运行项目前下载pre_embed_vec内的预训练模型，然后运行 main.py 即可

## 总结

### 1. 实验目的。

了解文本分类任务，写一个基于BiLSTM的文本分类器。

### 2. 数据清洗。

对数据进行 删除无用字符，大小写统一，繁体化简体，进行分词(tokenization)等。代码：

```python
def cut_str_by_len(str_, length):
    """
    将字符串str_按长度len拆分: ('abcdefg',3)-->['abc','def','g']
    :param str_: 待拆分的字符串
    :param length: 拆分长度
    :return: 拆分后的字符串列表
    """
    return [str_[i:i + length] for i in range(0, len(str_), length)]

def preprocess(sen):
    """
    针对hotel和phone数据的预处理模块。用来清洗评价数据，包括统一为小写，删除文本中的空格,换行，句号，问号，感叹号以及标签信息，将繁体转换为简体，最后利用jieba库进行tokenization操作
    :param sen: 待处理的字符串
    :return: list,处理并分词后的列表
    """
    import zhconv
    import jieba
    # import hanlp
    # tokenizer = hanlp.load('LARGE_ALBERT_BASE')
    sen.lower()
    sen = sen.replace(' ', '')
    sen = sen.replace('\n', '')
    pattern = re.compile(r'(?<=<).+?(?=>)')  # https://blog.csdn.net/z1102252970/article/details/70739804
    str1 = pattern.sub('', sen)
    str1 = str1.replace('<>', '')  
    cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")  # 匹配汉字，英文，数字
    str1 = zhconv.convert(str1, 'zh-cn')
    str1 = cop.sub('', str1)
    # tokens=tokenizer(str1) # hanlp问题：str1的长度不能超过126，多余的字符会被截断，所以需要对长字符串进行拆分
    # tokens=[]
    # if len(str1)>100:
    #     str1s=cut_str_by_len(str1,100)
    #     for split_text in str1s:
    #         tokens.extend(tokenizer(split_text))
    # else:tokens.extend(tokenizer(str1))
    tokens = jieba.cut(str1)
    return list(tokens)
```



### 3. 词表构建。

词表就是对分词后的 词语 进行编号(id)，需要建立映射：word<–>id ，同时要建立标签和id的映射：label <–> id （标签id化）。通常会对训练集建立词表（编号从0开始），这就会出现OOV问题（验证集和测试集中的词没有在建立的词表中），所以会在建立好的词表中再加入一个词 ”UNK“（unknown） ，id 为当前词表长度（也就是放到最后，不规定）。验证集和测试集中没有出现的词就统一为”UNK“即可。

### 4. 搭建BiLSTM模型

利用 pytorch 自带的 LSTM 网络搭建 BiLSTM 网络。 nn.LSTM网络默认的输入数据格式为[seq_max_len,batch_size,embedding]，输出数据格式：[seq_max_len,batch_size,num_directions*hidden_size]。具体解释看代码注释

```python
class BiLSTM(nn.Module):
    def __init__(self, opts,word_id, label_id):
        """
        模型初始化
        :param opts: 相关参数值
        :param word_id: 根据训练集建立的词典 vocab
        :param label_id: 标签到id的映射
        """
        super(BiLSTM, self).__init__()
        self.device = returnDevice(opts.cuda)
        self.input_size = opts.input_size
        self.hidden_size = opts.hidden_size
        self.num_layers = opts.num_layers
        self.dropout = opts.dropout
        self.batch_first = opts.batch_first
        self.embedding_dim=opts.embedding_dim
        self.bidirectional=opts.bidirectional
        #  nn.Embedding(num_embeddings: int, embedding_dim: int),(词汇表的大小，嵌入的维度）
        self.embeddings = nn.Embedding(len(word_id), self.embedding_dim)  # 将词汇表进行embedding，由于找的预训练词向量是300维的，因此这里的第二个维度是300维
        embedding = load_predtrained_emb_avg(word_id,opts.pre_embed_path)  #自定义的方法读取预训练的词向量， 中文词向量来源：https://github.com/Embedding/Chinese-Word-Vectors
        self.embeddings.weight.data.copy_(embedding)  # 将读取的词向量复制到嵌入向量中

        # bidirectional设为True即得到双向循环神经网络
        self.lstm = nn.LSTM(
            input_size=self.input_size,  # input_size 输入数据的特征维数，通常就是embedding_dim(词向量的维度)
            hidden_size=self.hidden_size,  # hidden_size　LSTM中隐层的维度
            num_layers=self.num_layers,  # num_layers　循环神经网络的层数
            dropout=self.dropout,  # 是否在除最后一个 RNN 层外的其他 RNN 层后面加 dropout 层。输入值是 0-1 之间的小数，表示概率。0表示0概率dripout，即不dropout
            batch_first=self.batch_first,
            # batch_first默认为False，则lstm输入的数据默认为：(seq_max_len,batch_size,input_size), 而我们常用的格式是(batch_size,seq_max_len,input_size),所以这里把batch_size设置为True，输入数据就已后者为准。
            bidirectional=self.bidirectional  # 是否为双向BiLSTM，双向：num_directions=2
        )

        # 全连接层：这里将前向LSTM隐藏层的输出和后向隐藏层的输出拼接到一起作为全连接层的输入，输出为类别的数目
        self.output = nn.Linear(2 * self.hidden_size, len(label_id))

    def bi_fetch(self,rnn_outs, seq_lengths, batch_size, max_len):
        """
        拼接前向LSTM隐藏层的输出和后向隐藏层的输出
        :param rnn_outs: BiLSTM的输出,(batch_size,seq_len,num_directions*hidden_size)
        :param seq_lengths: 输入一个batch的句子的真实长度，不包括padding的部分
        :param batch_size: 一个batch的大小
        :param max_len: 句子的最大长度
        :return: 拼接后的结果，(batch_size,2*hidden_size)
        """

        rnn_outs = rnn_outs.view(batch_size, max_len, 2, -1)  # 将双向的隐藏层参数分开
        # torch.index_select(),第一个参数是索引的对象，第二个参数指示选择哪个维度的数据，第三个参数是一个tensor，表示选择哪些数据。
        fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0]).to(self.device)))#
        fw_out = fw_out.view(batch_size * max_len, -1)
        bw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([1]).to(self.device)))
        bw_out = bw_out.view(batch_size * max_len, -1)
        # 每隔一个max_len长度是batch中的一个句子，我们要选择padding之前的隐藏层向量
        # 比如前两个句子实际长度为：3,4，padding后变成了max_len=12,那么我们就要选择序号为[3,4]隐藏层的向量，再往后就是padding的向量，就扔掉。
        # batch_range:[0*max_len,1*max_len,....,batch_size*max_len]
        batch_range = Variable(torch.LongTensor(list(range(batch_size))).to(self.device)) * max_len
        # batch_zeros:[0,0,...,0] 一共batch_size个0
        batch_zeros = Variable(torch.zeros(batch_size).to(self.device).long())

        fw_index = batch_range + seq_lengths.view(batch_size) - 1
        fw_out = torch.index_select(fw_out, 0, fw_index)  # (batch_size, hid)

        bw_index = batch_range + batch_zeros
        bw_out = torch.index_select(bw_out, 0, bw_index)

        outs = torch.cat([fw_out, bw_out], dim=1)
        return outs

    def forward(self, inputs, lengths,max_len):
        # lengths: (batch_size) 记录 input 中每个句子的真实长度，即不包含padding部分的长度
        # inputs：(batch_size,sentence) ,一个sentence是一句话中词的ID组成的list
        embeddings = self.embeddings(inputs)
        batch_size = len(embeddings)
        # lstm_input:(batch_size,seq_max_len,embedding)
        lstm_input = embeddings.view(batch_size, -1,self.embedding_dim )
        # output:(seq_max_len,batch_size,num_directions*hidden_size)
        #h_n(num_layers * num_directions, batch, hidden_size)
        #c_n(num_layers * num_directions, batch, hidden_size)
        output, (h_n,c_n) = self.lstm(lstm_input)
        # outs(batch_size,seq_len,num_directions*hidden_size)
        outs = output.contiguous().view(batch_size, -1, 2 * self.hidden_size)
        # bi_fetch取出前向LSTM的隐藏层和后向的LSTM的隐藏层输出拼接到一起返回
        # sentence_batch:(batch_size,2*hidden_size)
        sentence_batch = self.bi_fetch(outs, lengths, batch_size, max_len=max_len)

        out = self.output(sentence_batch)
        # out_prob = F.softmax(out.view(batch_size, -1))# 使用softmax来得到每个类别的概率值，取概率值最大的类别就是最终的分类结果。
        out_prob = F.log_softmax(out.view(batch_size, -1)) # 由于要计算交叉损失，需要取log，这里直接使用自带的log_softmax函数。不影响分类结果的

        return out_prob
```

### 5. 模型批数据准备

词表构建的结果vocab和label_id用来初始化BiLSTM网络的embedding部分和全连接层，现在来构建BiLSTM的forward() 函数需要的数据（inputs和lengths）max_len是一个超参数可以放到opts中，此处有点冗余。inputs 就是一批批的数据，想象成一个矩阵，每一行就是一句话，只不过数据是这句话分词的id组成的向量。由于每句话的长度不一样，而矩阵的每一行长度是一样的，因此有两个解决办法。第一个把长度一样的句子放到一起组成一批，第二个就是用padding的方式，长的截断，短的补值（一般补0）。这里用前者。

```python
def get_batches_by_len(data_x, data_y, batch_size=32):
    """
    批数据生成器
    :param data_x: 数据x
    :param data_y: 标签y
    :param batch_size:设置最大的batch_size,比如长度为20的句子有一百条，然而我们希望每一批数据最多（batch_size）只能有32条，所以需要对100条数据再次进行分批。
    :param shuffle: 是否打乱顺序
    :return:
    """
    seq_len={} #len_x---idxs
    for idx,x in enumerate(data_x):
        len_x=len(x)
        if len_x not in seq_len.keys():
            seq_len[len_x]=[idx]
        else:
            seq_len[len_x].append(idx)
    for len_x,idxs in seq_len.items():
        total_x=len(idxs)
        sum_x=0
        if total_x <= batch_size:
            X, y = [], []
            for idx in idxs:
                X.append(data_x[idx])
                y.append(data_y[idx])
            true_len = [len(X[0]) for _ in range(len(y))]
            yield X, y,true_len
        else:
            X, y = [], []
            for idx in idxs:
                X.append(data_x[idx])
                y.append(data_y[idx])
                sum_x+=1
                if sum_x==batch_size:
                    true_len = [len(X[0]) for _ in range(len(y))]
                    yield X, y,true_len
                    X, y = [], []
                    sum_x=0
            if len(y)!=0: # 如果batch_size=8,而idxs有18项，那么最后就会返回一个空的batch
                true_len = [len(X[0]) for _ in range(len(y))]
                yield X, y, true_len

```

### 6. 训练模型

上面的方法可以生成一批批数据（很多个batches)，每次喂给模型一个batch。数据准备好了就可以进行训练了。

```python
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
```

### 7. 评测模型在测试集上的结果

```python
trainer=trainer(opts)
trainer.train(train_data,dev_data)
trainer.model_result(test_data) # 评测模型结果
```

### 8. 一些坑

1. 如果模型的loss值不收敛，很有可能就是loss的计算代码有问题
2. 要注意保存中间结果，方便调试代码时节约时间
3. 超参数单独放一个文件保存，方便调参，比如我的超参都放在opts类里边
main.py