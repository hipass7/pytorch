import PIL
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

img = PIL.Image.open('./xray_data/1/0004a.jpg')

tf = transforms.ToTensor()
img_t = tf(img)

print(img_t.size())

tf2 = transforms.Resize((256, 256))
img_t = tf2(img_t)

print(img_t.shape) # 100, 3, 256, 256 만들어야함

img_t = torch.cat([img_t, img_t, img_t])

img_t = img_t.unsqueeze(0)

print(img_t.shape)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(74420, 500)
        self.fc2 = nn.Linear(500, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)

model = Net().to(device)

if __name__ == '__main__':
    classes = ('normal', 'cardiomegaly')

    PATH = './training3.pth'
    model.load_state_dict(torch.load(PATH, map_location=device))

    with torch.no_grad():
        img_t = img_t.to(device)
        outputs = model(img_t)
        print(outputs)
        _, predictions = torch.max(outputs, 1)
        print(predictions)