from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

def knncls():
    """
    k-近邻
    :return:
    """

    # 读取数据
    data = pd.read_csv()
    print(data.head(6))

    # 处理数据
    # 1.缩小数据 查询数据筛选
    data = data.query("x > 1.0 & x < 1.25 & 2.5<y<2.75")

    # 2.处理时间的数据
    time_values = pd.to_datetime(data["time"], unit="s")
    print(time_values)

    # 把日期格式转换为　字典格式
    time_values = pd.DatetimeIndex(time_values)

    # 构造一些新的特征
    data["day"] = time_values.day
    data["hour"] = time_values.hour
    data["weekday"] = time_values.weekday

    # 把时间戳 row_id特征删除
    data = data.drop(["time"], axis=1)
    data = data.drop(["row_id"], axis=1)
    print(data)

    # 把签到数量少于n个的　目标位置删除
    place_count = data.groupby("place_id").count()
    tf = place_count[place_count.row_id > 3].reset_index()
    data = data[data["place_id"].isin(tf.place_id)]

    # 取出数据当中的特征值和目标值
    y = data["place_id"] #  目标值
    x = data.drop(["[place_id"], axis=1) # 特征值

    # 进行数据的分割 训练集　测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程　标准化
    std = StandardScaler()

    # 要对训练集　和　测试集　的特征值　进行标准化
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 进行算法流程
    knn = KNeighborsClassifier()

    # 进行网格搜索
    gc = GridSearchCV(knn,param_grid={"n_neighbors":[1,3,5]}, cv=10)

    gc.fit(x_train, y_train)

    # 预测准确率
    print("在测试集上准确率:",gc.score(x_test, y_test))
    print("在交叉验证当中最好的结果:", gc.best_score_)
    print("选择最好的模型是:", gc.best_estimator_)
    print("每个超参数每次交叉验证的结果:", gc.cv_results_)
    # fit predict,score
    # knn.fit(x_train, y_train)
    #
    # # 得出预测结果　
    # y_predict = knn.predict(x_test)
    # print("预测的目标签到位置为:", y_predict)
    #
    # # 得出准确率
    # print("预测的准确率:", knn.score(x_test, y_test))

if __name__ == "__main__":
    knncls()
