from sklearn import cluster
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time

data = pd.read_csv("three_class_data.csv", header=0)

x = data[['x', 'y']]

cluster_name = ['KMeans', 'MiniBatchMeans', 'AffinityPropagation', 'MeanShift', 'SpectralClustering',
                 'AgglomeractiveClustering', 'Birch', 'DBSCAN']

cluster_estimators = [
    cluster.KMeans(n_clusters=3),
    cluster.MiniBatchKMeans(n_clusters=3),
    cluster.AffinityPropagation(),
    cluster.MeanShift(),
    cluster.SpectralClustering(n_clusters=3),
    cluster.AgglomerativeClustering(n_clusters=3),
    cluster.Birch(n_clusters=3),
    cluster.DBSCAN()
]

plot_num = 1

for name, model in zip(cluster_name, cluster_estimators):
    tic = time.time()

    model.fit(x)

    plt.subplot(2, len(cluster_estimators) / 2, plot_num)

    x_min, x_max = data['x'].min() - 1, data['x'].max() + 1
    y_min, y_max = data['y'].min() - 1, data['y'].max() + 1
    xx, yy =np.meshgrid(np.arange(x_min, x_max, .01), np.arange(y_min, y_max, .01))

    if hasattr(model, 'predict'):
        result = model.predict(np.c_[xx.ravel(), yy.ravel()])

        result = result.reshape(xx.shape)

        plt.contourf(xx, yy, result, cmap=plt.cm.Greens)

    plt.scatter(data['x'], data['y'], c=model.labels_, s=15)

    if hasattr(model, 'cluster_centers_'):
        center = model.cluster_centers_
        plt.scatter(center[:, 0], center[:, 1], marker='p', linewidths=2, color='b', edgecolors='w', zorder= 20)

    toc = time.time()
    cluster_time = (toc - tic)*1000

    plt.title(str(name) + ", " + str(int(cluster_time)) + "ms")
    plot_num += 1

plt.show()
