import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

x = np.load('data/x_train_vec.npy')

y = np.load('data/y_train.npy')

model = SVC(kernel='rbf', verbose=True)

model.fit(x, y)

joblib.dump(model, 'data/svm_model.pkl')

print(cross_val_score(model, x, y))

print('done')

