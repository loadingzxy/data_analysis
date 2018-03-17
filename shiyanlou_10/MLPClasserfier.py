import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

df = pd.read_excel("user_fit.xlsx", header=0)

x = df.iloc[:, 0:8]
y = df['用户是否为会员']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = MLPClassifier(activation='logistic',max_iter=50, hidden_layer_sizes=(50, 50, 50))

model.fit(x_train, y_train)

score_trainset = model.score(x_train, y_train) * 100
score_testset = model.score(x_test, y_test) * 100

print("训练集预测准确率： %2.2f%%" % score_trainset)
print( score_testset)
print('done')