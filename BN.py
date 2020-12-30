import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_xy = np.loadtxt('trainset.csv', delimiter=',', dtype=np.float32)
x_train_tensor = torch.from_numpy(train_xy[:, 0:-1])
y_train_tensor = torch.from_numpy(train_xy[:, [-1]])


test_xy = np.loadtxt('testset.csv', delimiter=',', dtype=np.float32)
x_test_tensor = torch.from_numpy(test_xy[:, 0:-1])
y_test_tensor = torch.from_numpy(test_xy[:, [-1]])

origin_train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_dataset, val_dataset = random_split(origin_train_dataset, [int(x_train_tensor.shape[0] * 0.8), int(x_train_tensor.shape[0] * 0.2)])

train_loader = DataLoader(dataset=train_dataset, batch_size=512)
val_loader = DataLoader(dataset=val_dataset, batch_size=256)

test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=256)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.bn0 = nn.BatchNorm1d(7)
        self.fc1 = nn.Linear(7, 20)
        self.bn1 = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 10)
        self.bn2 = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10, 5)
        self.bn3 = nn.BatchNorm1d(5)
        self.fc4 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.bn0(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x


#def weight_init(m):
    #classname = m.__class__.__name__ # 得到网络层的名字，如ConvTranspose2d
    #if classname.find('BatchNorm1d') != -1:  # 使用了find函数，如果不存在返回值为-1，所以让其不等于-1
    #    m.weight.data.normal_(0.0, 0.02)
    #elif classname.find('Linear') != -1:
      #  m.weight.data.normal_(1.0, 0.02)
      # m.bias.data.fill_(0)

n_epochs = 500

model = Net().to(device)
model.load_state_dict(torch.load('params.pth'))
#model.apply(weight_init)

loss_fn = nn.MSELoss()

#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

training_losses = []
validation_losses = []

for epoch in range(n_epochs):
    batch_losses = []
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        model.train()
        yh = model(x_batch)
        loss = loss_fn(yh, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        optimizer.zero_grad()
        batch_losses.append(loss.detach().cpu().numpy())
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
            val_losses.append(val_loss.cpu().numpy())
        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)
        print(
           f"[{epoch + 1:2}] Training loss: {training_loss:.3f} \t "
           f"Validation loss: {validation_loss:.3f} ")

y_pred = model(x_test_tensor.to(device))
test_loss = loss_fn(y_pred, y_test_tensor.to(device))
print(test_loss)
params = list(model.named_parameters())
#print(params)
#torch.save(model.state_dict(), 'params.pth')

a = torch.arange(500, 700)
plt.plot(a.numpy(), y_pred[500:700].detach().cpu().numpy(), 'r-', label='predict')
plt.plot(a.numpy(), y_test_tensor[500:700].detach().cpu().numpy(), color='blue', marker='o', label='true')
plt.legend(loc='upper right')
plt.show()
