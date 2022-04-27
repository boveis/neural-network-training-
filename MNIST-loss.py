import torch
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

class BaharFF(torch.nn.Module):
    def __init__(self, inputSize, hidden_size, outputSize):
        super(BaharFF, self).__init__()
        self.linear1 = torch.nn.Linear(inputSize, hidden_size)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(hidden_size, outputSize)
        self.sigmoid3 = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.linear2(self.sigmoid1(self.linear1(x)))
        return out


train_dataset = mnist.MNIST(root='./', train=True, transform=ToTensor(), download=True)
test_dataset = mnist.MNIST(root='./', train=False, transform=ToTensor(), download=True)

bs = 25
LR = 0.05
epochs = 5
scale = 0.01

train_loader = DataLoader(train_dataset, batch_size=bs)
test_loader = DataLoader(test_dataset, batch_size=bs)

model = BaharFF(784, 100, 784)
loss_fn = torch.nn.MSELoss()

for epoch in range(epochs):
    for data in train_loader:
        data_input = data[0]
        old_shape = data_input.shape
        data_input = data_input.reshape(old_shape[0], old_shape[2]*old_shape[3])
        noise = torch.normal(mean=0, std=0.01, size=data_input.shape)/scale
        noisy_data_input = data_input + noise
        outputs = model(noisy_data_input)
        loss = loss_fn(outputs, data_input)
        loss.backward()
        for param in model.parameters():
            param.data -= LR * param.grad.data

train_loss = 0
test_loss = 0
for data in train_loader:
    data_input = data[0]
    old_shape = data_input.shape
    data_input = data_input.reshape(old_shape[0], old_shape[2] * old_shape[3])
    noise = torch.normal(mean=0, std=0.01, size=data_input.shape) / scale
    noisy_data_input = data_input + noise
    outputs = model(noisy_data_input)
    loss = loss_fn(outputs, data_input)
    train_loss += loss


for data in test_loader:
    data_input = data[0]
    old_shape = data_input.shape
    data_input = data_input.reshape(old_shape[0], old_shape[2] * old_shape[3])
    noise = torch.normal(mean=0, std=0.01, size=data_input.shape) / scale
    noisy_data_input = data_input + noise
    outputs = model(noisy_data_input)
    loss = loss_fn(outputs, data_input)
    test_loss += loss

print('Scale: ', scale)
print('Train Loss: ', train_loss/len(train_dataset))
print('Test Loss: ', test_loss/len(test_dataset))
