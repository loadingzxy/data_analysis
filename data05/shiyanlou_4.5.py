from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA


data = pd.read_csv("zoo.csv", header=0)

feature = data.iloc[:, 1:17].values
target = data['type'].values

print(feature)

pca = PCA(n_components=2)
feature_pca = pca.fit_transform(feature)

x_train, x_test, y_train, y_test = train_test_split(feature_pca,target ,test_size=0.3,random_state=50)

model = SVC()

model.fit(x_train, y_train)

results = model.predict(x_test)

print(model.score(x_test, y_test))
print(x_train)

plt.scatter(x_train[:, 0], x_train[:, 1], alpha=0.3)

plt.scatter(x_test[:, 0], x_test[:, 1], marker=',', c=y_test)

for i, txt in enumerate(results):
    plt.annotate(txt, (x_test[:, 0][i], x_test[:, 1][i]))

plt.show()
