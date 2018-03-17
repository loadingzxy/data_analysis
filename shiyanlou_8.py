import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jieba import analyse

course_ori = pd.read_table("courses.txt", sep=',', header=0)

# print(course_ori.head())

i = pd.to_datetime(course_ori['创建时间'])

# print(i.head())

course_ts = pd.DataFrame(data=course_ori.values, columns=course_ori.columns, index=i)

# print(course_ts.head())

# course_ts_W = course_ts.resample('W').sum()
# plt.plot_date(course_ts_W.index, course_ts_W['学习时间'], '-')
# course_ts_W['id'] = range(0,len(course_ts_W.index.values))
# sns.regplot("id", "学习时间", data=course_ts_W, scatter_kws={"s":10}, order=8, ci=None, truncate=True)
# sns.regplot("id", "学习人数", data=course_ts_W, x_bins=10)
# plt.xlabel('Time Series')
# plt.ylabel('Study Time')
# plt.show()

course_ts_A = course_ts.copy()

course_ts_A['平均学习时间'] = course_ts_A['学习时间']/course_ts_A['学习人数']

a = []
for i in course_ts_A['课程名称']:
    a.append(analyse.extract_tags(i, topK=2, withWeight=False, allowPOS=('eng', 'n', 'vn', 'v')))
keywords = pd.DataFrame(a, columns=['关键词1', '关键词2'])
# print(keywords.head())
# print(course_ts_A.head())

course_ts_C = course_ts_A.copy()
course_ts_C = course_ts_C.reset_index()

course_ts_merged = pd.concat([course_ts_C, keywords], axis=1).drop("创建时间", axis=1)

print(course_ts_C.head())

#  第38行有错误  未被改正