import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_xy = np.loadtxt('测试集.csv', delimiter=',', dtype=np.float32)
x_train_tensor = torch.from_numpy(train_xy[:, 0:-1])
y_train_tensor = torch.from_numpy(train_xy[:, [-1]])
#y_train_tensor = y_train_tensor.squeeze()
#y_train_tensor = y_train_tensor.to(torch.float32)
#y_train_tensor = torch.as_tensor(y_train_tensor)

#a = torch.range(0, 1079)
#a = torch.unsqueeze(a, dim = 1)

#plt.plot(a.numpy(), train_xy[:, [-1]], label='testy')
#plt.xlim(0, 1080)
#plt.legend(loc= 'upper right')
#plt.show()


test_xy = np.loadtxt('训练集.csv', delimiter=',', dtype=np.float32)
x_test_tensor = torch.from_numpy(test_xy[:, 0:-1])
y_test_tensor = torch.from_numpy(test_xy[:, [-1]])
#y_test_tensor = y_test_tensor.squeeze()
#y_test_tensor = y_test_tensor.to(torch.float32)
#y_test_tensor = torch.as_tensor(y_test_tensor)


# Builds dataset with ALL data
origin_train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

# Splits randomly into train and validation datasets
train_dataset, val_dataset = random_split(origin_train_dataset, [int(x_train_tensor.shape[0] * 0.8), int(x_train_tensor.shape[0] * 0.2)])

# Builds a loader for each dataset to perform mini-batch gradient descent
train_loader = DataLoader(dataset=train_dataset, batch_size=1000)
val_loader = DataLoader(dataset=val_dataset, batch_size=1000)

test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader  = DataLoader(dataset=test_dataset, batch_size=1000)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.bn1 = nn.BatchNorm1d(7)
        self.fc1 = nn.Linear(7, 20)
        self.bn2 = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 10)
        self.bn3 = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10, 5)
        self.bn4 = nn.BatchNorm1d(5)
        self.fc4 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.bn1(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.bn3(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.bn4(x)
        x = self.fc4(x)
        return x

n_epochs = 100
lr = 0.01
momentum = 0.05

def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        yh = model(x)
        loss = loss_fn(yh, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        optimizer.zero_grad()
        num_correct = (yh == y).sum().item()
        acc = num_correct / x.shape[0]

        return [loss.item(), acc]
    return train_step

model = Net().to(device)

loss_fn = nn.MSELoss()

# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

train_step = make_train_step(model, loss_fn, optimizer)

training_losses = []
validation_losses = []

for epoch in range(n_epochs):
    training_accuracies = 0
    validation_accuracies = 0
    batch_losses = []
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        loss, acc = train_step(x_batch, y_batch)
        batch_losses.append(loss)
        training_accuracies = training_accuracies + acc

    training_loss = np.mean(batch_losses)
    training_losses.append(training_loss)

    with torch.no_grad():
        val_losses = []
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            model.eval()
            yh = model(x_val)
            val_loss = loss_fn(yh, y_val)
            val_losses.append(val_loss)
            num_correct = (yh == y_val).sum().item()
            acc = num_correct / x_val.shape[0]
            validation_accuracies += acc

        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)
        #print(
            #f"[{epoch + 1:2}] Training loss: {training_loss:.3f} acc: {training_accuracies / len(train_loader):.3f} \t "
            #f"Validation loss: {validation_loss:.3f} acc: {validation_accuracies / len(val_loader):.3f}")