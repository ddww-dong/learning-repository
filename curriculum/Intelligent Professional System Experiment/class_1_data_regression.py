import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x=torch.unsqueeze(torch.linspace(-np.pi,np.pi,100),dim=1)# 构建等差数列并转换成二维数组

y=torch.sin(x)+0.5*torch.rand(x.size())#添加随机数

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.predict = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
    def forward(self, x):
        return self.predict(x)


#训练网络

net = Net()
optimizer = torch.optim.SGD(net.parameters(),lr = 0.05)
loss_func = nn.MSELoss()

for epoch in range(1,1001):
    plt.ion()
    if epoch % 100==0  :
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), out.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, f'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.title('epoch = %d' % epoch)
        plt.pause(0.1)
    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.ioff()
plt.show()