# -*- coding: utf-8 -*-
# @Time    : 2020/11/13
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : main.py
from classify import trainer
from opts import opts
opts=opts()
trainer=trainer(opts)
trainer.train()
trainer.model_result()