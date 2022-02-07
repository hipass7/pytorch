import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from vgg19 import *
import numpy as np
from torch.autograd import Variable
from smote import *

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

learning_rate = 0.001
training_epochs = 20
batch_size = 4

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = VGG('VGG19').to('cpu')

if __name__ == '__main__':
    set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(set, batch_size = 50000, num_workers = 0, shuffle = True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

    train_images, train_labels = iter(trainloader).next()

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    trainset = []
    trainlabel = []

    cnt = [0] * 10
    for i in range(50000):
        if train_labels[i] > 0:
            if cnt[train_labels[i]] >= 500:
                continue
        trainset.append(train_images[i])
        trainlabel.append(train_labels[i])
        cnt[train_labels[i]] += 1

    traindata = np.array(trainset)
    trainlabel = np.array(trainlabel)
    train_images_tensor = torch.tensor(traindata)
    X = model.features(train_images_tensor)
    X = X.view(X.size(0), -1)
    train_labels_tensor = torch.tensor(trainlabel, dtype = torch.long)

    smote = SMOTE()
    images, labels = smote.fit_generate(X, train_labels_tensor)

    trainset = torch.utils.data.TensorDataset(images, labels)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, num_workers = 0, shuffle = True)

    total_batch = len(trainloader)
    print(total_batch)

    # training part
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    

    for epoch in range(training_epochs):
        avg_cost = 0

        for X, Y in trainloader:
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = model.classifier(X)
            
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch

        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

    PATH = './cifar10_test.pth'
    torch.save(model.state_dict(), PATH)

    correct = 0
    total = 0

    # 10000개의 data 전체 accuracy prediction code
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # class별 accuracy prediction code
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                    accuracy))