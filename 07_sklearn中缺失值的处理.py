# 对数据进行处理　缺失值..

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.impute import SimpleImputer


def mm():
    """
    标准化缩放
    :return: None
    """
    # mm = StandardScaler()
    # 均值为0,方差为1范围内
    # 二维数组!
    # 以列取均值  中位数
    # SimpleImputer(missing_values=np.nan, strategy='mean')
    mm = SimpleImputer(missing_values=np.nan, strategy='mean')
    # mm = Imputer(missing_values="NaN", strategy="mean", axis=0)
    # replace("?", np.nan)
    data = mm.fit_transform(
        [[1, 2],
         [np.nan, 3],
         [7, 6]]
    )

    # pandas dropnan fillna 数据当中的缺失值:np.nan float类型
    print(data)
    print(type(data))
    # 多个特征　同等重要的时候　进行归一化
    # 目的: 使得某一个特征不会对最终结果不会造成更大的影响
    # print(mm.mean_)


if __name__ == "__main__":
    mm()
