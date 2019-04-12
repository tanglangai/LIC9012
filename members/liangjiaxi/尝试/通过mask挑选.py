import torch

a = torch.randn((2,2,2))

b = torch.zeros((2,2,2)).long()

b[0,0,0] = 1
b[0,0,1] = 1
b[1,0,0] = 1

print(a[b == 1])