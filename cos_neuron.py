import matplotlib.pyplot as plt
import torch
     
class BaharFF(torch.nn.Module):
    def __init__(self, inputSize, hidden_size, outputSize):
        super(BaharFF, self).__init__()
        self.linear1 = torch.nn.Linear(inputSize, hidden_size)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(hidden_size, outputSize)

    def forward(self, x):
        out = self.linear2(self.sigmoid1(self.linear1(x)))
        return out


model_with1000_neuron = BaharFF(1, 50 , 1)
x_train = torch.linspace(-5, 5, 100).reshape(-1, 1)
y_train = torch.cos(x_train).reshape(-1, 1)

criterion = torch.nn.MSELoss() 

for epoch in range(2000):
    model_with1000_neuron.zero_grad()
    outputs = model_with1000_neuron(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    for param in model_with1000_neuron.parameters():
        param.data -= 0.05*param.grad.data    

loss = criterion(y_train , model_with1000_neuron(x_train))
print(loss.item())
