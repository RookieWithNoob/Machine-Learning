# [[2,8,4,5],
# [6,3,0,8],
# [5,4,9,1]]
from sklearn.decomposition import PCA

def pca():
    """
    主成分分析进行特征降维
    :return:
    """
    pca = PCA(n_components=0.9) # 需要保留的数据的百分值 90-95之间
    data = pca.fit_transform(
        [[2, 8, 4, 5],
         [6, 3, 0, 8],
         [5, 4, 9, 1]]

    )
    print(type(data))
    print(data)
    return None

if __name__ == "__main__":
    pca()