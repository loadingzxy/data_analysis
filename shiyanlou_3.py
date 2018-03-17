import pandas as pd
import numpy as np

def find_outlier(data):
    outlier = []

    data = order(data)
    n = len(data)

    q1 = data[n/4]
    q3 = data[(n/4)*3]
    iqr = q3 - q1
    for i in data:
        if i < q1-1.5*iqr or q3+1.5*iqr:
            outlier += i

    return outlier

def order(lists):
    count = len(lists)
    for i in range(count):
        lists[i] = min(lists[i:])
    return lists