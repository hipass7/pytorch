import torch
from torch.autograd import Variable

x = Variable(torch.ones(2, 2))
print(x)