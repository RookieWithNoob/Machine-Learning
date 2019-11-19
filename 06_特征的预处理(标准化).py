# 对数据进行处理　缺失值..

from sklearn.preprocessing import StandardScaler


def mm():
    """
    标准化缩放
    :return: None
    """
    mm = StandardScaler()
    # 均值为0,方差为1范围内
    # 二维数组!
    data = mm.fit_transform(
        [[1., -1., 3.],
         [2., 4., 2.],
         [4., 6., -1.]]

    )

    print(data)
    print(type(data))
    # 多个特征　同等重要的时候　进行归一化
    # 目的: 使得某一个特征不会对最终结果不会造成更大的影响
    print(mm.mean_)


if __name__ == "__main__":
    mm()
