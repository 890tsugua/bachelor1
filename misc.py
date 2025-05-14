import torch

a = torch.ones(4)
b = torch.ones(4) *2
bb = torch.ones(4) * 3
c = torch.stack([a, b, bb], dim=0)
print(c, c.shape)

cx = (c[:, 0] + c[:, 2]) / 2
cy = (c[:, 1] + c[:, 3])

print(cx, cy)
p = torch.stack([cx, cy], dim=1)
print(p, p.shape)
