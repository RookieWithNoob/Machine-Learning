from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report,confusion_matrix
from sklearn.linear_model import Ridge
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from plot_混淆矩阵 import plot_confusion_matrix,plt




def logistic():
    """
    逻辑回归做二分类进行癌症预测　根据细胞的特征属性
    :return: None
    """

    # pd读取数据
    # 没有列名　指定列名 names = column_names
    column_names = ['Sample code number','Clump Thickness', 'Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion', 'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", names=column_names)
    print(data.head(5))

    # 缺失值进行处理
    data = data.replace(to_replace="?", value=np.nan)  # 替换 replace("?", np.nan)
    data = data.drop(["Sample code number"], axis=1)
    # 删除
    # data = data.dropna()
    # 替换
    si = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = si.fit_transform(data)
    print(data)

    #数据分割
    x_train, x_test, y_train, y_test = train_test_split(data[:,:9], data[:,9], test_size=0.25)

    # 标准化处理
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 逻辑回归预测
    lg = LogisticRegression(solver='liblinear')
    lg.fit(x_train, y_train)

    y_predict = lg.predict(x_test)

    # 绘图
    cnf_matrix = confusion_matrix(y_test, y_predict)
    class_names = ["Good", "Bad"]
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')


    print("权重参数:", lg.coef_)

    print("准确率:", lg.score(x_test, y_test))
    # 列别需要对应　
    print("召回率:", classification_report(y_test, y_predict, target_names=["良性", "恶性"], labels=[2, 4]))
    # precision 精确率
    # recall 召回率

    return None


if __name__ == "__main__":
    logistic()