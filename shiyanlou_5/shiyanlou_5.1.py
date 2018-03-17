from sklearn.cluster import k_means
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
import  pandas as pd

data = pd.read_csv("three_class_data.csv", header=0)

x = data[['x', 'y']].values

score = []

for i in range(10):
    model = k_means(x, n_clusters=i+2)
    score.append(silhouette_score(x, model[1]))

plt.subplot(1, 2, 1)
plt.scatter(data['x'], data['y'])

plt.subplot(1, 2, 2)
plt.plot(range(2, 12, 1), score)
plt.show()