import pandas as pd
import numpy as np
import jieba

neg = pd.read_excel("data/neg.xls", header=None, index=None)
pos = pd.read_excel("data/pos.xls", header=None, index=None)

word_cut = lambda x: jieba.lcut(x)
pos['words'] = pos[0].apply(word_cut)
neg['words'] = neg[0].apply(word_cut)

x = np.concatenate((pos['words'], neg['words']))
y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

np.save('data/x_train.npy', x)
np.save('data/y_train.npy', y)

print('done')