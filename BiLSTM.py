# -*- coding: utf-8 -*-
# @Time    : 2020/11/5
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : BiLSTM.py
import torch
from torch import nn
from utils import load_predtrained_emb_avg
import torch.nn.functional as F
from torch.autograd import Variable

from utils import  returnDevice

class BiLSTM(nn.Module):
    def __init__(self, opts,word_id, label_id):
        """
        模型初始化
        :param opts: 相关参数值
        :param word_id: 根据训练集建立的词典 vocab
        :param label_id: 标签到id的映射
        """
        super(BiLSTM, self).__init__()
        self.device = returnDevice(opts.cuda,opts.gpu_number)
        self.input_size = opts.input_size
        self.hidden_size = opts.hidden_size
        self.num_layers = opts.num_layers
        self.dropout = opts.dropout
        self.batch_first = opts.batch_first
        self.embedding_dim=opts.embedding_dim
        self.bidirectional=opts.bidirectional
        #  nn.Embedding(num_embeddings: int, embedding_dim: int),(词汇表的大小，嵌入的维度）
        self.embeddings = nn.Embedding(len(word_id), self.embedding_dim)  # 将词汇表进行embedding，由于找的预训练词向量是300维的，因此这里的第二个维度是300维, 这里加 padding的id
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
        # inputs：(batch_size,sentence)
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
        out_prob = F.log_softmax(out.view(batch_size, -1)) #
        # out_prob = F.softmax(out.view(batch_size, -1))

        return out_prob
