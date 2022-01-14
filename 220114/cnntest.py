import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from smote import *

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

num_epochs = 0
batch_size = 100
learning_rate = 0.001
img_size = 256, 256

transform = transforms.Compose(
    [transforms.Resize(img_size),transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder(root='./xray_data', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root='./xray_data', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
                            
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    sm = SMOTE()
    images, labels = sm.fit_generate(images, labels)
    # imshow(torchvision.utils.make_grid(images))

    # conv1 = nn.Conv2d(3, 6, 5) # 3개의 채널을 6개의 채널로 확장하고 5x5 filter, padding=0, stride = 1
    # pool = nn.MaxPool2d(2, 2) # 2x2 filter, stride=2
    # conv2 = nn.Conv2d(6, 16, 5) # 6개의 채널을 16개로 확장, 5x5 filter, 위와 동일
    # print(images.shape)
    # print(labels)

    # x = conv1(images)
    # print(x.shape)

    # x = pool(x)
    # print(x.shape)

    # x = conv2(x)
    # print(x.shape)

    # x = pool(x)
    # print(x.shape)

    conv1 = nn.Conv2d(3, 10, kernel_size=5)
    conv2 = nn.Conv2d(10, 20, kernel_size=5)
    conv2_drop = nn.Dropout2d()
    fc1 = nn.Linear(74420, 2)
    fc2 = nn.Linear(500, 50)
    fc3 = nn.Linear(50, 2)

    print(images.shape)

    x = conv1(images)
    print(x.shape)

    x = F.max_pool2d(x, 2)
    print(x.shape)

    x = F.relu(x)
    print(x.shape)

    x = conv2(x)
    print(x.shape)

    x = conv2_drop(x)
    print(x.shape)

    x = F.max_pool2d(x, 2)
    print(x.shape)

    x = F.relu(x)
    print(x.shape)

    x = x.view(x.size(0), -1)
    print(x.shape)

    x = F.dropout(x)
    print(x.shape)

    x = fc1(x)
    print(x.shape)

    x = F.log_softmax(x, dim=1)
    print(x.shape)
    print(x)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    cost = criterion(x, labels)
    print(cost)
    cost.backward()

    print(cost / 100)


