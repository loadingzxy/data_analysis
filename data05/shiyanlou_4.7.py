from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


iris = load_iris()

x_train = iris.data[:120]
x_test = iris.data[120:]

y_train = iris.target[:120]
y_test = iris.target[120:]

model_tree = tree.DecisionTreeClassifier()
model_random =RandomForestClassifier()

model_tree.fit(x_train, y_train)
s1 = model_tree.score(x_test, y_test)

model_random.fit(x_train, y_train)
s2 = model_random.score(x_test, y_test)

print('DecisionTree:', s1)
print('RandomForest:', s2)