from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
# from sklearn.externals import joblib
import joblib


def mylinear():
    """
    线性回归两种方式　预测房子价格
    :return: None
    """

    # 获取数据
    lb = load_boston()

    # 分割数据集　训练集　测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    # print(y_train)
    # 进行标准化处理　ｋ近邻和线性回归都要标准化
    # 特征值　目标值都需要进行准备化处理 实例化两个准备化api
    std_x = StandardScaler() # 对特征值标准化
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 目标值
    std_y = StandardScaler() # 对目标值标准化
    y_train = std_y.fit_transform(y_train.reshape(-1,1))
    y_test = std_y.transform(y_test.reshape(-1,1))

    # 正规方求解方式预测结果
    lr = LinearRegression()

    lr.fit(x_train, y_train.ravel())

    print(lr.coef_)  # 已经得到参数

    # 保存训练好的模型
    joblib.dump(lr, "./test.pkl")

    # 预测测试集的房子价格
    y_lr_predict = std_y.inverse_transform(lr.predict(x_test))

    print("测试集里面的每个房子的价格:", y_lr_predict)
    print("正规方程的均方误差:", mean_squared_error(std_y.inverse_transform(y_test), y_lr_predict))

    # # 梯度下降进行预测
    # sgd = SGDRegressor()
    #
    # sgd.fit(x_train, y_train.ravel())
    #
    # print(sgd.coef_)
    #
    # # 预测测试集的房子价格
    # y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))
    #
    # print("测试集里面的每个房子的价格:", y_sgd_predict)
    # print("梯度下降的均方误差:", mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))
    #
    # # 岭回归进行预测
    # rd = Ridge(alpha=1.0)
    #
    # rd.fit(x_train, y_train.ravel())
    # # 权重参数
    # print(rd.coef_)
    #
    # # 预测测试集的房子价格
    # y_rd_predict = std_y.inverse_transform(rd.predict(x_test))
    #
    # print("测试集里面的每个房子的价格:", y_rd_predict)
    # print("岭回归的均方误差:", mean_squared_error(std_y.inverse_transform(y_test), y_rd_predict))


    return None



if __name__ == "__main__":
    mylinear()