import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ExponentialLR

lr = 1e-4
lr_list = []
for i in range(30):
    lr_list.append(lr)
    lr *= 0.85
plt.grid(True)
plt.plot(list(range(30)),lr_list)
plt.show()