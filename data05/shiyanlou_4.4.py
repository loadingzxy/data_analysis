import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA


data = pd.read_csv("zoo.csv", header=0)

feature = data.iloc[:, 1:17].values
target = data['type'].values

x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.3, random_state=50)

model = SVC()

model.fit(x_train, y_train)

results = model.predict(x_test)

print(model.score(x_test, y_test))