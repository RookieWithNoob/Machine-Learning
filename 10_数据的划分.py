from sklearn.datasets import load_iris, fetch_20newsgroups, load_boston
from sklearn.model_selection import train_test_split
import pprint

# li = load_iris()

# 特征值
# print(li.data)
# # 目标值　标签值
# print(li.target)
# # 数据集的描述
# print(li.DESCR)

# 注意返回值　训练集:train　特征值:x_train二维 目标值:y_train　一维
# 测试集test x_test y_test 一般测试集为25%
# x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25)
# print("训练集特征值和目标值\n", x_train,"\n", y_train)
# print("测试集特征值和目标值\n", x_test, "\n", y_test)

# news = fetch_20newsgroups(data_home="/home/python/桌面/", subset="all")
# pprint.pprint(news.data)
# pprint.pprint(news.target)


# 回归数据集
lb = load_boston()
print(lb.data)
print(lb.target)
print(lb.DESCR)