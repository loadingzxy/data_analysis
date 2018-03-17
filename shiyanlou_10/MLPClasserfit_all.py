import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

df = pd.read_excel("user_fit.xlsx", header=0)

x = df.iloc[:, 0:8]
y = df['用户是否为会员']

df_pre = pd.read_excel("user_prediction.xlsx", header=0)

predict_x = df_pre.iloc[:, 0:8]

model = MLPClassifier(activation='logistic', max_iter=50, hidden_layer_sizes=(50,50,50))

model.fit(x,y)

results = model.predict_proba(predict_x) * 100

results_df = pd.DataFrame(np.around(results, 2), columns=['非会员概率', '会员概率'])

df_merged = pd.concat([df_pre.drop("用户是否为会员", axis=1), results_df['会员概率']], axis=1)

print(df_merged.sort_values(by='会员概率', ascending=False).head(20))