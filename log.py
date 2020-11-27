# -*- coding: utf-8 -*-
# @Time    : 2020/11/14
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : log.py
import os

class Log:
    def __init__(self, opts):
        if opts.log_fname == '':
            raise RuntimeError('-log_fname must be given')
        self.path = os.path.join(opts.log_dir,opts.log_fname)
        if not os.path.isdir(opts.log_dir):
            os.mkdir(opts.log_dir)
        self.opts = opts
        self.fprint_opts()

    def fprint_opts(self):

        with open(self.path, 'w', encoding='utf8') as f:
            f.write('opts:\n')
            f.write('\n'.join(['%s:%s' % item for item in self.opts.__dict__.items()]))
            f.write('\n\n')

    def fprint_log(self, text):
        with open(self.path, 'a', encoding='utf8') as f:
            f.write(text + '\n')

