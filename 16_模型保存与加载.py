from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
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
    std_x = StandardScaler()  # 对特征值标准化
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 目标值
    std_y = StandardScaler()  # 对目标值标准化
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))

    # 预测房价结果
    model = joblib.load("./test.pkl")

    y_predict = std_y.inverse_transform(model.predict(x_test))

    print("保存的模型的预测的结果:", y_predict)
    print("均方误差:", mean_squared_error(std_y.inverse_transform(y_test), y_predict))

    return None


if __name__ == "__main__":
    mylinear()