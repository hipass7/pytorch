import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from vgg19 import *
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

learning_rate = 0.001
training_epochs = 20
batch_size = 4

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = VGG('VGG19').to(device)

def unpickle(file):
    import pickle
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding = "bytes")
    return dict

def pickle_to_images_and_labels(root):
    data = unpickle(root)
    data_images = data[b'data'] / 255
    data_images = data_images.reshape(-1, 3, 32, 32).astype("float32")
    data_labels = data[b'labels']
    return data_images, data_labels


if __name__ == '__main__':
    images1, label1 = pickle_to_images_and_labels("./data/cifar-10-batches-py/data_batch_1")
    images2, label2 = pickle_to_images_and_labels("./data/cifar-10-batches-py/data_batch_2")
    images3, label3 = pickle_to_images_and_labels("./data/cifar-10-batches-py/data_batch_3")
    images4, label4 = pickle_to_images_and_labels("./data/cifar-10-batches-py/data_batch_4")
    images5, label5 = pickle_to_images_and_labels("./data/cifar-10-batches-py/data_batch_5")
    train_images = np.concatenate([images1, images2, images3, images4, images5], axis = 0)
    train_labels = np.concatenate([label1, label2, label3, label4, label5], axis = 0)
    trainset = []
    trainlabel = []

    cnt = [0] * 10
    for i in range(50000):
        if cnt[train_labels[i]] < 500 * (10 - train_labels[i]):
            trainset.append(train_images[i])
            trainlabel.append(train_labels[i])
        cnt[train_labels[i]] += 1
            
    # cnt1 = cnt2 = cnt3 = cnt4 = cnt5 = cnt6 = cnt7 = cnt8 = cnt9 = 0
    # for i in range(50000):
    #     if train_labels[i] == 0:
    #         trainset.append(train_images[i])
    #         trainlabel.append(train_labels[i])            
    #     elif train_labels[i] == 1:
    #         if cnt1 >= 4500:
    #             continue
    #         cnt1 += 1
    #         trainset.append(train_images[i])
    #         trainlabel.append(train_labels[i])    
    #     elif train_labels[i] == 2:
    #         if cnt2 >= 4000:
    #             continue
    #         cnt2 += 1
    #         trainset.append(train_images[i])
    #         trainlabel.append(train_labels[i])   
    #     elif train_labels[i] == 3:
    #         if cnt3 >= 3500:
    #             continue
    #         cnt3 += 1
    #         trainset.append(train_images[i])
    #         trainlabel.append(train_labels[i])   
    #     elif train_labels[i] == 4:
    #         if cnt4 >= 3000:
    #             continue
    #         cnt4 += 1
    #         trainset.append(train_images[i])
    #         trainlabel.append(train_labels[i])   
    #     elif train_labels[i] == 5:
    #         if cnt5 >= 2500:
    #             continue
    #         cnt5 += 1
    #         trainset.append(train_images[i]) 
    #         trainlabel.append(train_labels[i])              
    #     elif train_labels[i] == 6:
    #         if cnt6 >= 2000:
    #             continue
    #         cnt6 += 1
    #         trainset.append(train_images[i])
    #         trainlabel.append(train_labels[i])   
    #     elif train_labels[i] == 7:
    #         if cnt7 >= 1500:
    #             continue
    #         cnt7 += 1
    #         trainset.append(train_images[i]) 
    #         trainlabel.append(train_labels[i])     
    #     elif train_labels[i] == 8:
    #         if cnt8 >= 1000:
    #             continue
    #         cnt8 += 1
    #         trainset.append(train_images[i])   
    #         trainlabel.append(train_labels[i])     
    #     elif train_labels[i] == 9:
    #         if cnt9 >= 500:
    #             continue
    #         cnt9 += 1
    #         trainset.append(train_images[i])  
    #         trainlabel.append(train_labels[i])

    traindata = np.array(trainset)
    trainlabel = np.array(trainlabel)
    train_images_tensor = torch.tensor(traindata)
    train_labels_tensor = torch.tensor(trainlabel, dtype = torch.long)
    trainset = torch.utils.data.TensorDataset(train_images_tensor, train_labels_tensor)
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                      download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, num_workers = 2, shuffle = True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

    total_batch = len(trainloader)
    print(total_batch)

    # training part
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(training_epochs):
        avg_cost = 0

        for X, Y in trainloader:
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = model(X)
            
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