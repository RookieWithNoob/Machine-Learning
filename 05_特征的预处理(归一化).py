# 对数据进行处理　缺失值..
# [[90,2,10,40],
# [60,4,15,45],
# [75,3,13,46]]
from sklearn.preprocessing import MinMaxScaler

def mm():
    """
    归一化处理
    :return: None
    """
    mm = MinMaxScaler()
    # mm = MinMaxScaler(feature_range=(2,3))  范围
    # 二维数组!
    data = mm.fit_transform(
            [[90, 2, 10, 40],
             [60, 4, 15, 45],
             [75, 3, 13, 46]]
    )

    print(data)
    print(type(data))
    # 多个特征　同等重要的时候　进行归一化
    # 目的: 使得某一个特征不会对最终结果不会造成更大的影响


if __name__ == "__main__":
    mm()