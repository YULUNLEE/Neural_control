import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import csv

u_data=[]
x_data=[]
y_data=[]
u_data_test=[]
x_data_test=[]
y_data_test=[]
loss_list=[]
i=0

#create training data
u_data.append(float((np.sin((2*np.pi*i)/25)+np.sin((2*np.pi*i)/10))))
x_data.append(np.sin(i)) #x(k)=sin(k)
y_data.append((x_data[i]/(1+x_data[i]**2))+u_data[i]**3)
for i in range(1, 101):
    u_data.append(float((np.sin((2 * np.pi * i) / 25) + np.sin((2 * np.pi * i) / 10))))
    x_data.append(y_data[i-1])
    y_data.append((x_data[i] / (1 + x_data[i] ** 2)) + u_data[i] ** 3)



print(x_data)
print(u_data)
print(y_data)

#write training data in file
with open('train_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x_data', 'u_data', 'y_data'])
    for i in range(101):
        writer.writerow([x_data[i], u_data[i], y_data[i]])


i=0
#create testing data
u_data_test.append(float((np.sin((2*np.pi*i)/25)+np.sin((2*np.pi*i)/10))))
x_data_test.append(1.4*np.cos(i)) #x(k)=sin(k)
y_data_test.append((x_data_test[i]/(1+x_data_test[i]**2))+u_data_test[i]**3)
for i in range(1, 101):
    u_data_test.append(float((np.sin((2 * np.pi * i) / 25) + np.sin((2 * np.pi * i) / 10))))
    x_data_test.append(y_data_test[i-1])
    y_data_test.append((x_data_test[i] / (1 + x_data_test[i] ** 2)) + u_data_test[i] ** 3)



print(x_data_test)
print(u_data_test)
print(y_data_test)

#write testing data in file
with open('test_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x_data', 'u_data', 'y_data'])
    for i in range(101):
        writer.writerow([x_data_test[i], u_data_test[i], y_data_test[i]])


#繪圖
plt.plot(np.linspace(0,100,101), y_data)
plt.plot(np.linspace(0,100,101), y_data_test)
plt.show()

#train list --> array
array_x=np.array(x_data)[:,np.newaxis]
array_u=np.array(u_data)[:,np.newaxis]
array_y=np.array(y_data)[:,np.newaxis]
print(array_x.shape)
print(array_u.shape)
print(array_y.shape)

array_input=np.hstack((array_x, array_u))
print(array_input.shape)
# print(array_input)

#train data numpy --> tensor -->variable
input_data=torch.FloatTensor(array_input)
y_data=torch.FloatTensor(array_y)
print("torch_data:")
print(input_data)
print(y_data.shape)

#test list --> array
array_x_test=np.array(x_data_test)[:,np.newaxis]
array_u_test=np.array(u_data_test)[:,np.newaxis]
array_y_test=np.array(y_data_test)
print(array_x_test.shape)
print(array_u_test.shape)
print(array_y_test.shape)

array_input_test=np.hstack((array_x_test, array_u_test))
print(array_input_test.shape)
# print(array_input_test)


#model
class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # 隐藏层线性输出
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2, n_output)   # 输出层线性输出
    def forward(self, x):
        x=F.relu(self.hidden1(x))
        x=F.relu(self.hidden2(x))
        x=self.predict(x)
        return x

net=Net(n_feature=2, n_hidden1=8, n_hidden2=4, n_output=1)
print(net)

optimizer=torch.optim.Adam(net.parameters())
loss_func=torch.nn.MSELoss()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.linspace(0,80,81), y_data[0:81])
plt.ion()

for i in range(1001):
    prediction=net(input_data)
    loss=loss_func(prediction, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i %100==0:
        # plot and show learning process
        print(f'train cost {i}: {loss}')
        plt.pause(0.1)
        try:
            ax.lines.remove(lines[0])
            textvar.remove()

        except Exception:
            pass
        lines = ax.plot(np.linspace(0, 80, 81), prediction[:81].data.numpy(), 'r-', lw=2)
        textvar = ax.text(30, -7, f'Loss=%.4f' % loss, fontdict={'size': 10, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

#val plot
plt.plot(np.linspace(81, 100, 20), y_data[81:].data.numpy())
plt.plot(np.linspace(81, 100, 20), prediction[81:].data.numpy())
plt.title("Validation curve")
plt.show()


