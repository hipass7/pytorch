import torch
from torch.autograd import Variable

x = Variable(torch.ones(2,2), requires_grad = True)
print(x.requires_grad)