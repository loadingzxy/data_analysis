import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

data = pd.read_csv("two_class_data.csv", header=0)

x = data['x']
y = data['y']
c = data['class']

plt.scatter(x, y, c=c)
plt.show()