import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB

df = pd.read_excel("user_fit.xlsx", header=0)

feature = df.iloc[: , 0:8]
target = df['用户是否为会员']

model = GaussianNB()

model.fit(feature, target)

joblib.dump(model, 'model_GaussianNB.pkl')

print('done')