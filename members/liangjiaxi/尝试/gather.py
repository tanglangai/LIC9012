import torch

torch.manual_seed(1)
a = torch.randn((2,3))
print(a)
print(a.shape)

b = torch.LongTensor([0,1]).unsqueeze(1)
print(a[:,b])
print(a[:,b].shape)




# c = torch.gather(a, 1, b)
# print(c)