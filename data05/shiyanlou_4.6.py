from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv("three_class_data.csv", header=0)

feature = data[['x', 'y']].values
target = data['class'].values

x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.3, random_state=50)

model = KNeighborsClassifier()

model.fit(x_train, y_train)

results = model.predict(x_test)

print(model.score(x_test, y_test))

cm0 = plt.cm.Oranges
cm1 = plt.cm.Greens
cm2 = plt.cm.Reds
cm_color = ListedColormap(['red', 'yellow'])

x_min, x_max = data['x'].min() - .5, data['x'].max() + .5
y_min, y_max = data['y'].min() - .5, data['y'].max() + .5

xx, yy =np.meshgrid(np.arange(x_min, x_max, .1), np.arange(y_min, y_max, .1))

Z0 = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 0]
Z1 = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z2 = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 2]

Z0 = Z0.reshape(xx.shape)
Z1 = Z1.reshape(xx.shape)
Z2 = Z2.reshape(xx.shape)

plt.contourf(xx, yy, Z0, cmap=cm0, alpha=.9)
plt.contourf(xx, yy, Z1, cmap=cm0, alpha=.5)
plt.contourf(xx, yy, Z2, cmap=cm0, alpha=.4)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_color)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_color, edgecolors='black')
plt.show()


#颜色上面有一丝问题啊
#下面为实验楼正确代码
#2018.03.13

#
# from matplotlib import pyplot as plt
# from matplotlib.colors import ListedColormap
#
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
#
# # 导入数据
# data = pd.read_csv('three_class_data.csv', header=0)
#
# # 定义特征变量和目标变量
# feature = data[['x', 'y']].values
# target = data['class'].values
#
# # 对数据集进行切分，70% 为训练集，30% 为测试集。
# x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.3, random_state=50)
#
# # 构建模型
# model = KNeighborsClassifier()
#
# # 训练模型
# model.fit(x_train, y_train)
#
# # 预测
# results = model.predict(x_test)
#
# print(model.score(x_test, y_test))
#
# # 绘制决策边界等高线图
# cm0 = plt.cm.Oranges
# cm1 = plt.cm.Greens
# cm2 = plt.cm.Reds
# cm_color = ListedColormap(['red', 'yellow'])
#
# x_min, x_max = data['x'].min() - .5, data['x'].max() + .5
# y_min, y_max = data['y'].min() - .5, data['y'].max() + .5
#
# xx, yy = np.meshgrid(np.arange(x_min, x_max, .1),
#                      np.arange(y_min, y_max, .1))
#
# Z0 = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 0]
# Z1 = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
# Z2 = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 2]
#
# Z0 = Z0.reshape(xx.shape)
# Z1 = Z1.reshape(xx.shape)
# Z2 = Z2.reshape(xx.shape)
#
# plt.contourf(xx, yy, Z0, cmap=cm0, alpha=.9)
# plt.contourf(xx, yy, Z1, cmap=cm1, alpha=.5)
# plt.contourf(xx, yy, Z2, cmap=cm2, alpha=.4)
#
# # 绘制训练集和测试集
# plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_color)
# plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_color, edgecolors='black')
# plt.show()
#
