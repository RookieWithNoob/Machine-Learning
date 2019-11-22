import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz

def decision():
    """
    决策树对泰坦尼克号进行预测生死
    :return: None
    """

    # 获取数据
    titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
    # print(titan.head(5))

    # 处理数据 筛选出特征值和目标值
    x = titan[["pclass", "age", "sex"]] # 特征值
    y = titan[["survived"]] # 目标值

    # 处理缺失值
    # x["age"].fillna(x["age"].mean(), inplace=True)

    # 分割数据
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行处理　特征工程  特征-类别-one_hot 编码
    # 文本类型才用TfidfVectorizer 分词　重要性 单个属性用DictVectorizer 注意转换成字典格式
    # [{"age":11, "pclass": "1st", "sex":"male"},{},{}]
    dict = DictVectorizer(sparse=False)
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    print(dict.get_feature_names())
    x_test = dict.transform(x_test.to_dict(orient="records"))
    # print(x_train)

    # 缺失值
    data = SimpleImputer(missing_values=np.nan, strategy='mean')
    x_train = data.fit_transform(x_train)
    x_test = data.transform(x_test)
    print("处理缺失值后的:\n", x_train)
    # print(x)

    # 用决策树进行预测
    dec = DecisionTreeClassifier(max_depth=8)
    dec.fit(x_train, y_train)

    # 预测准确率
    gc = GridSearchCV(dec, param_grid={"max_depth": [5, 6],}, cv=2)
    gc.fit(x_train, y_train)
    print(y_train)
    print("预测的准确率:", dec.score(x_test, y_test))
    print("在测试集上准确率:", gc.score(x_test, y_test))
    print("在交叉验证当中最好的结果:", gc.best_score_)
    print("选择最好的模型是:", gc.best_estimator_)
    print("每个超参数每次交叉验证的结果:", gc.cv_results_)

    # 决策树的结构　本地保存 DOT格式
    # 导出决策树的结构
    export_graphviz(dec, out_file="./tree.dot", feature_names=['年龄', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女性', '男性'])



    return None

if __name__ == "__main__":
    decision()