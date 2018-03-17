import numpy as np
from gensim.models.word2vec import Word2Vec

x_train = np.load('data/x_train.npy')

w2v = Word2Vec(size=300, min_count=10)
w2v.build_vocab(x_train)
w2v.train(x_train, total_examples=w2v.corpus_count, epochs=w2v.iter)

def average_vec(text):
    vec = np.zeros(300).reshape((1, 300))
    for word in text:
        try:
            vec += w2v[word].reshape(1, 300)
        except KeyError:
            continue
    return vec

train_vec = np.concatenate([average_vec(z) for z in x_train])

w2v.save('data/w2v_model.pkl')
np.save('data/x_train_vec.npy', train_vec)

print('done')
