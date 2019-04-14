import torch

a = torch.randn((2,2,2))
print(a)

b = a.max()
print(b)
print(a/b)