'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable

class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.classifier = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, 10)  
            )

    def forward(self, x):
        # x = Variable(x, requires_grad = True)
        out = self.classifier(x)
        return out