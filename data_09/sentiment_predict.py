from gensim.models.word2vec import Word2Vec
import numpy as np
import jieba
from sklearn.externals import joblib
import pandas as pd


def average_vec(words):
    w2v = Word2Vec.load('data/w2v_model.pkl')
    vec = np.zeros(300).reshape((1, 300))
    for word in words:
        try:
            vec += w2v[word].reshape((1, 300))
        except KeyError:
            continue
    return vec

def svm_predict():
    df = pd.read_csv("comments.csv", header=0)

    comment_sentiment = []
    for string in df['评论内容']:
        words = jieba.lcut(str(string))
        words_vec = average_vec(words)

        model = joblib.load('data/svm_model.pkl')
        result = model.predict(words_vec)
        comment_sentiment.append(result[0])

        if int(result[0]==1):
            print(string, "[积极]")
        else:
            print(string, "[消极]")

    merged = pd.concat([df, pd.Series(comment_sentiment, name='用户情绪')], axis=1)

    pd.DataFrame.to_csv(merged, 'comment_sentiment.csv')
    print('done')

svm_predict()