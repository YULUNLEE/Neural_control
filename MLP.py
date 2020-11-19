import numpy as np
import csv
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as k

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
# k=np.random.rand()*100
# u_data.append(float((np.sin((2 * np.pi * k) / 25) + np.sin((2 * np.pi * k) / 10))))
x_data.append(np.sin(i)) #x(k)=sin(k)
y_data.append((x_data[i]/(1+x_data[i]**2))+u_data[i]**3)
for i in range(1, 101):
    u_data.append(float((np.sin((2 * np.pi * i) / 25) + np.sin((2 * np.pi * i) / 10))))
    # k = np.random.rand() * 100
    # u_data.append(float((np.sin((2 * np.pi * k) / 25) + np.sin((2 * np.pi * k) / 10))))
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
array_y=np.array(y_data)
print(array_x.shape)
print(array_u.shape)
print(array_y.shape)

array_input=np.hstack((array_x, array_u))
print(array_input.shape)
print(array_input)

#test list --> array
array_x_test=np.array(x_data_test)[:,np.newaxis]
array_u_test=np.array(u_data_test)[:,np.newaxis]
array_y_test=np.array(y_data_test)
print(array_x_test.shape)
print(array_u_test.shape)
print(array_y_test.shape)

array_input_test=np.hstack((array_x_test, array_u_test))
print(array_input_test.shape)
print(array_input_test)


# def root_mean_squared_error(y_true, y_pred):
#     return k.sqrt(k.mean(k.square(y_pred - y_true)))

# 建立一個 squential 的 model
model = Sequential()


# 將神經層加到 model 裡
model.add(Dense(input_dim=2, units=8, activation='relu'))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=1))
model.compile(loss='MSE', optimizer='sgd')


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.linspace(0,80,81), y_data[0:81])
plt.ion()

with open('MLP_weights.txt', 'w') as file:
    for step in range(2001):
        cost = model.train_on_batch(array_input[0:81], array_y[0:81])

        if step % 20 == 0:
            print(f'train cost {step}: {cost}')
            loss_list.append(cost)
            # print('1', model.get_weights()[0])
            # print('2', model.get_weights()[1])
            # print('3', model.get_weights()[2])
            # print('4', model.get_weights()[3])
            # print('5', model.get_weights()[4])
            # print('6', model.get_weights()[5])
            plt.pause(0.1)
            try:
                ax.lines.remove(lines[0])
                textvar.remove()

            except Exception:
                pass
            lines = ax.plot(np.linspace(0,80,81), model.predict(array_input[0:81]), 'r-', lw=2)
            textvar=ax.text(30, -7, f'Loss=%.4f' % cost, fontdict={'size': 10, 'color': 'red'})
            plt.pause(0.1)

        if step % 100 == 0:
            file.writelines(f"{step} steps weights record\n\n")
            file.writelines(f"input layer:\n{model.get_weights()[0]}\n\n")
            file.writelines(f"hidden layer:\n{model.get_weights()[1]}\n\n")
            file.writelines(f"hidden layer:\n{model.get_weights()[2]}\n\n")
            file.writelines(f"hidden layer:\n{model.get_weights()[3]}\n\n")
            file.writelines(f"hidden layer:\n{model.get_weights()[4]}\n\n")
            file.writelines(f"output layer:\n{model.get_weights()[5]}\n\n")
            file.writelines("#######################################################")


plt.ioff()
plt.show()


#驗證訓練模型
val_loss=model.evaluate(array_input[81:101], array_y[81:101])
print("validation_loss:", val_loss)
plt.plot(np.linspace(81, 100, 20), y_data[81:])
plt.plot(np.linspace(81, 100, 20), model.predict(array_input[81:]))
plt.show()

#畫出訓練loss 圖
plt.plot(np.linspace(0,2000, 101), loss_list)
plt.title("loss")
plt.xlabel('steps')
plt.ylabel("RMSE")
plt.show()

test_loss=model.evaluate(array_input_test, array_y_test)
print("test_loss:", test_loss)

print(model.summary())

from keras.utils import plot_model
plot_model(model, to_file='model_MLP.png', show_shapes=True)


#
# plt.plot(np.linspace(0,100,101), y_data)
# plt.plot(np.linspace(0,100,101), model.predict(array_input[:]), 'r-')
# plt.show()


# for i in range(101):
#     y_pred.append(model.predict(array_input[i]))
# print(y_pred)

# #model
# model=Sequential()
# model.add(Dense(input_dim=1, units=5, activation='relu'))
# model.add(Dense(units=1, activation='relu'))
#
# adam=keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# model.compile(loss='mse', optimizer=adam)
#
# for step in range(10001):
#     # cost = model.train_on_batch(feature, train_data_y_array_train)
#     cost = model.train_on_batch(array_x, array_y)
#     if step % 100 == 0:
#         print("step:", step, 'train cost: ', cost)






# 直線
# from keras.models import Sequential
# from keras.layers import Dense
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 建立 X, Y 兩組資料用來練習 keras 的使用
# X = np.linspace(-1, 1, 200)
# np.random.shuffle(X)
# Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
# # 將資料分成兩組，一組是用來 train model, 另一組用來測試 model 預測的效果。
# X_train, Y_train = X[:160], Y[:160]
# X_test, Y_test = X[160:], Y[160:]
#
# # 建立一個 squential 的 model
# model = Sequential()
# # 建立一個輸入及輸出都是一維(輸入 X 輸出 Y)的全連接型態的神經層
# dense = Dense(units=1, input_dim=1)
# # 將神經層加到 model 裡
# model.add(dense)
# # compile 是用來安排學習過程的，optimizer 可以輸入一個 optimizer instance 或直接輸入該 optimizer class 的名字的字串。loss 也是一樣的用法。
# # compile() 其實還有第三個參數 metrics, 那是用在「分類」的問題上。
# # compile 文件: https://keras.io/getting-started/sequential-model-guide/#compilation
# # https://keras.io/optimizers/
# # https://keras.io/losses/
# print(X_train.shape)
# print(Y_train.shape)
# model.compile(loss='mse', optimizer='sgd')
#
# # train 這個 model 300 次
# for step in range(301):
#     cost = model.train_on_batch(X_train, Y_train)
#     if step % 100 == 0:
#         print('train cost: {}'.format(cost))
# # 用測試的那一組資料來測試 model 的學習效果, 用 model.evaluate 取得 loss 值。若在 compile 時有指定 metrics，這裡也會回傳 metrics。
# # https://keras.io/models/model/
# cost = model.evaluate(X_test, Y_test, batch_size=40)
# print("test cost: {}".format(cost))
# W, b = model.layers[0].get_weights()
# print("weights = {}, biases= {}".format(W, b))
#
# Y_pred = model.predict(X_test)
# plt.scatter(X_test, Y_test)
# plt.plot(X_test, Y_pred)
# plt.show()
