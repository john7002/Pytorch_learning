import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import numpy as np


trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,)),transforms.ToPILImage(),transforms.Pad(2),transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True,transform=trans)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=20,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True,transform=trans)
testloader = torch.utils.data.DataLoader(testset, batch_size=40,
                                         shuffle=False, num_workers=2)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,kernel_size=5) #out is n-f+1 = 32-5+1 : 28x28x6
        self.pool1 = nn.AvgPool2d(kernel_size=2,stride=2) #out is 14x14x6

        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5) #out is 16*5*5

        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        self.output = nn.Softmax()



    def forward(self,x):
        x = self.pool1(torch.tanh(self.conv1(x)))
        x = self.pool1(torch.tanh(self.conv2(x)))
        x = x.view(-1, 16*5*5)  # if only x.view(-1), it returns shape of 100x16x5x5 = 40000, but we want 100x400 dimensions => xview(-1, 16*5*5)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.output(x)

        return x


net = Net()
#The input are organized in [N, C, W, H]
#input = torch.randn(1, 1, 32, 32)
#out = net(input)
#print(out)
#print(out.shape)
#summary(your_model, input_size=(channels, H, W))
summary(net, (1, 32, 32),device="cpu")


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step() #All optimizers implement a step() method, that updates the parameters

        # print statistics
        running_loss += loss.item()
        if i % 1000 ==0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

print('Finished Training')

print("Do some tests on several examples:")

dat= iter(testloader)
rand_picture,label_rand = dat.next()

_,max_idx = torch.max(net(rand_picture[4].unsqueeze(0)),1)
print("for a number {0}, we predict a {1}".format(label_rand[4].unsqueeze(0),max_idx))
_,max_idx = torch.max(net(rand_picture[17].unsqueeze(0)),1)
print("for a number {0}, we predict a {1}".format(label_rand[17].unsqueeze(0),max_idx))
_,max_idx = torch.max(net(rand_picture[11].unsqueeze(0)),1)
print("for a number {0}, we predict a {1}".format(label_rand[11].unsqueeze(0),max_idx))
_,max_idx = torch.max(net(rand_picture[9].unsqueeze(0)),1)
print("for a number {0}, we predict a {1}".format(label_rand[9].unsqueeze(0),max_idx))