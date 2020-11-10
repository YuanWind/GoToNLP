import torch
a=torch.randn(2,3,4)
print(a)
b=a.view(3,2,-1)
print(b)