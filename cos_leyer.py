import matplotlib.pyplot as plt
import torch
     
class BaharFF(torch.nn.Module):
    def __init__(self, inputSize, hidden_size_1 , hidden_size_2 , outputSize):
        super(BaharFF, self).__init__()
        self.linear1 = torch.nn.Linear(inputSize, hidden_size_1)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(hidden_size_1 , hidden_size_2)
        self.sigmoid2 = torch.nn.Sigmoid()
        self.linear3 = torch.nn.Linear(hidden_size_2 , outputSize)
        
    def forward(self, x):
        out =  self.linear3(self.sigmoid2(self.linear2(self.sigmoid1(self.linear1(x)))))  
        return out


model = BaharFF(1, 100,100, 1)
criterion = torch.nn.MSELoss() 
x_train = torch.linspace(-5, 5, 100).reshape(-1, 1)
y_train = torch.cos(x_train).reshape(-1, 1)

for epoch in range(2000):
    model.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    for param in model.parameters():
        param.data -= 0.05*param.grad.data    
    

plt.plot(x_train, y_train, label='main data')
plt.plot(x_train, model(x_train).detach().numpy(), label='trained_data')
plt.legend()
plt.savefig('cos.png')

loss = criterion(model(x_train) , y_train)
print(loss.item())