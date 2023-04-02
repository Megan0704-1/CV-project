import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd

class InputData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x_item = torch.FloatTensor(self.X[index])
        y_item = torch.FloatTensor(self.y[index])
        return x_item, y_item
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(114, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    

print('Accuracy of the network on the test data: %d %%' %(100*correct / total))

if __name__ == '__main__':
    train_data = pd.read_csv("./train.csv")
    test_data = pd.read_csv("./test.csv")
    X_train = train_data.drop(['Date', 'Loc', 'WindDir', 'DayWindDir', 'NightWindDir', 'Weather'])
    X_test = test_data.drop(['Date', 'Loc', 'WindDir', 'DayWindDir', 'NightWindDir', 'Weather'])
    y_train = train_data.Weather
    dataset = InputData(X_train, y_train)
    batch_size = 32
    shuffle = True

    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    model = Net()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("Epoch %d loss %.3f" %(epoch+1, running_loss/len(trainloader)))

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = model(inputs)
            predicted = (outputs>0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()