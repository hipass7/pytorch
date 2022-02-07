import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from smote import *
from vgg19 import *
import torch.autograd as Variable

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = VGG('VGG19').to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

def data_setting(trainloader, model):
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
    train_labels_tensor = torch.tensor(trainlabel, dtype = torch.long)

    train = torch.utils.data.TensorDataset(train_images_tensor, train_labels_tensor)

    trainloade = torch.utils.data.DataLoader(train, batch_size = 4, num_workers = 2, shuffle = True)

    return trainloade

def preconvfeat(dataset, model):
    conv_features = []
    labels_list = []
    for data in dataset:
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        output = model(inputs)
        output = output.view(output.size(0), -1)
        conv_features.extend(output.data.cpu().numpy())
        labels_list.extend(labels.data.cpu().numpy())
    conv_features = np.concatenate([[feat] for feat in conv_features])
    conv_features = torch.tensor(conv_features)
    labels_list = torch.tensor(labels_list, dtype = torch.long)
    smote = SMOTE()
    conv_features, labels_list = smote.fit_generate(conv_features, labels_list)
    return (conv_features, labels_list)

def training(trainloader):
    for epoch in range(20):
        avg_cost = 0

        for X, Y in trainloader:
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = model.classifier(X)

            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()

            avg_cost += cost.item() / 12500

        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

    PATH = './cifar10_test.pth'
    torch.save(model.state_dict(), PATH)

if __name__ == '__main__':
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(set, batch_size = 50000, num_workers = 0, shuffle = True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    new_trainloader = data_setting(trainloader, model)

    images, labels = preconvfeat(new_trainloader, model.features)

    train_set = torch.utils.data.TensorDataset(images, labels)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 4, num_workers = 2, shuffle = True)

    # training(train_loader)

    classifier = nn.Linear(512,10)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.0001, momentum=0.9)

    training(train_loader)  

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
