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

bs = 10
LR = 0.05
epochs = 100
train_loader = DataLoader(train_dataset , batch_size=bs)
test_loader = DataLoader(test_dataset , batch_size=bs)

model = BaharFF(784, 100, 10)
loss_fn = torch.nn.MSELoss()

print(train_loader.__sizeof__)
for epoch in range(epochs):
    for data in train_loader:
        data_input = data[0]
        old_shape = data_input.shape
        data_input = data_input.reshape(old_shape[0], old_shape[2]*old_shape[3])
        data_label = data[1]
        new_data_label = torch.zeros(bs, 10)
        for idx in range(bs):
            new_data_label[idx, data_label[idx]] = 1
        outputs = model(data_input)
        loss = loss_fn(outputs, new_data_label)
        loss.backward()
        for param in model.parameters():
            param.data -= LR * param.grad.data
            
corr_num = 0
for data in test_loader:
    data_input = data[0]
    old_shape = data_input.shape
    data_input = data_input.reshape(old_shape[0], old_shape[2] * old_shape[3])
    data_label = data[1]

    outputs = model(data_input)

    values, indices = torch.max(outputs, dim=1)

    diff = indices - data_label
    correct_num = torch.numel(diff[diff == 0])
    corr_num += correct_num
print('Accuracy: {}%'.format(100*corr_num/len(test_dataset)))