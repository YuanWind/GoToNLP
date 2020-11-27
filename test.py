from opts import opts
opts=opts()
print('\n'.join(['%s:%s' % item for item in opts.__dict__.items()]))