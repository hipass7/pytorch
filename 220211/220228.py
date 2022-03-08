import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from vgg19 import *
from encoding import *
from classifier import *
from torch.autograd import Variable

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

learning_rate = 0.001
training_epochs = 50
batch_size = 4

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

encoder = encoding('VGG19').to(device)
classifi = classifier().to(device)
classifi = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, 10)  
            ).to(device)
            
# PATH = './classifi.pth'
# classifi.load_state_dict(torch.load(PATH, map_location=device))

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
            if cnt[train_labels[i]] >= 2000:
                continue
        trainset.append(train_images[i])
        trainlabel.append(train_labels[i])
        cnt[train_labels[i]] += 1

    traindata = np.array(trainset)
    trainlabel = np.array(trainlabel)
    train_images_tensor = torch.tensor(traindata)
    train_labels_tensor = torch.tensor(trainlabel, dtype = torch.long)
    trainset = torch.utils.data.TensorDataset(train_images_tensor, train_labels_tensor)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, num_workers = 0, shuffle = True)

    total_batch = len(trainloader)
    print(total_batch)

    model = VGG('VGG19').to(device)
    # training part
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


    # with torch.no_grad():
    # images, labels = smote_and_encoder(train_images_tensor, train_labels_tensor)
    # images, labels = Variable(images), Variable(labels)

    # smoteset = torch.utils.data.TensorDataset(images, labels)
    # smoteloader = torch.utils.data.DataLoader(smoteset, batch_size = batch_size, num_workers = 0, shuffle = True)

    tempset = []
    templabel = []

    for epoch in range(1):
        avg_cost = 0

        for X, Y in trainloader:
            X = X.to(device)
            Y = Y.to(device)

            # optimizer.zero_grad()
            X = model.features(X)
            out = X.view(X.size(0), -1)

            out_images = out.detach().cpu()
            out_labels = Y.detach().cpu()
            out_images = np.array(out_images)
            out_labels = np.array(out_labels)
            tempset.append(out_images)
            templabel.append(out_labels)

        #     hypothesis = model.classifier(out)
            
        #     cost = criterion(hypothesis, Y)
        #     cost.backward()
        #     optimizer.step()

        #     avg_cost += cost / total_batch

        # print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

    traindata = np.array(tempset)
    trainlabel = np.array(templabel)
    train_images_tensor = torch.tensor(traindata).view(-1, 512)
    print(train_images_tensor.shape)
    train_labels_tensor = torch.tensor(trainlabel, dtype = torch.long).view(-1)
    smoteset = torch.utils.data.TensorDataset(train_images_tensor, train_labels_tensor)
    smoteloader = torch.utils.data.DataLoader(smoteset, batch_size = batch_size, num_workers = 0, shuffle = True)

    PATH = './220228-1.pth'
    # torch.save(model.state_dict(), PATH)
    model.load_state_dict(torch.load(PATH, map_location=device))

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            # X = model.features(images)
            # out = X.view(X.size(0), -1)
            # output = classifi(out)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    optimizer = torch.optim.SGD(classifi.parameters(), lr=learning_rate, momentum=0.9)
    for epoch in range(training_epochs):
        avg_cost = 0

        for X, Y in smoteloader:
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = classifi(X)
            
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch

        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

    PATH = './classifi.pth'
    torch.save(classifi.state_dict(), PATH)

    correct = 0
    total = 0

    # 10000개의 data 전체 accuracy prediction code
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            X = model.features(images)
            out = X.view(X.size(0), -1)
            output = classifi(out)
            # output = model(images)
            _, predicted = torch.max(output.data, 1)
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