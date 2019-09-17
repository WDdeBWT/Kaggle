import torch

a = torch.randn(3, 1)
b = torch.randn(3)
print(a)
print(b)

e = []

for c, d in zip(a, b):
    e.append([c.item(), d])
    # print(c)
    # print(d)

print(e)