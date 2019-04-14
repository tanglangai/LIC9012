import torch

a = torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0])
b= torch.Tensor([])
print(torch.bernoulli(a, 0.5))