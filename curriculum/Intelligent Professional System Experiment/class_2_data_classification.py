import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

data = torch.ones(100,2)
x0 = torch.normal(2*data,1)
x1 = torch.normal(-2*data, 1)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((torch.zeros(100), torch.ones(100))).type(torch.LongTensor)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(2, 15),
            nn.ReLU(),
            nn.Linear(15, 2),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        classifiction = self.classify(x)
        return classifiction


net = Net()
optimizer = torch.optim.SGD(net.parameters(),lr = 0.03)
loss_func = nn.CrossEntropyLoss() #分类问题使用交叉熵损失函数
for epoch in range(101):
    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print('epoch = %d, loss = %.4f' % (epoch, loss.data.numpy()))
        classification = torch.max(out,1)[1]
        class_y = classification.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c = class_y,s=100,cmap="RdYlGn")
        accuracy =  float((class_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.title('epoch = %d' % epoch)
        plt.show()