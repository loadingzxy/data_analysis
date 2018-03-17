import pandas
import numpy
from pandas import Series,DataFrame

def analysis(file_name, user_id):
    times = 0
    minutes = 0

    df = pandas.read_json(file_name)
    data = df[df['user_id']==user_id]
    for i in data['minutes']:
        times+=1
        minutes+=i
    print(times, minutes)
    # print(data)

    return times, minutes

if __name__ == '__main__':
    path = 'user_study.json'
    uid = 199071
    analysis(path, uid)