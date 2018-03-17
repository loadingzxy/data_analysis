import pandas as pd
import numpy as np
from sklearn.externals import joblib

df = pd.read_excel("user_prediction.xlsx", header=0)

feature = df.iloc[:, 0:8]

model_GaussianNB = joblib.load("model_GaussianNB.pkl")

results = model_GaussianNB.predict_proba(feature) * 100

result_df = pd.DataFrame(np.around(results, 2), columns=['非会员概率', '会员概率'])

merged = pd.concat([df.drop("用户是否为会员", axis=1), result_df['会员概率']], axis=1)

print(merged.head(20))