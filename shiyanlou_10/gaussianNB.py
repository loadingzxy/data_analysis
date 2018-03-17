import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_excel('user_fit.xlsx', header=0)

feature = df.iloc[:, 0:8]

target = df['用户是否为会员']

x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.3)

model_Gaussian = GaussianNB()

model_Gaussian.fit(x_train, y_train)

score = model_Gaussian.score(x_test, y_test)

print(score)

print('dnoe')