
import torch


a = torch.randn((11, 5))
b = torch.randn((1, 5))
print(a)
print(b)

print(b.expand_as(a))
b = b.expand_as(a)
assert a.shape == b.shape