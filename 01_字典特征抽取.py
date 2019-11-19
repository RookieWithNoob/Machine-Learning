
from sklearn.feature_extraction import DictVectorizer


def dictvec():
    """
    字典数据抽取
    :return: None
    """
    # 实例化
    dict = DictVectorizer(sparse=False)

    # 调用fit_transform
    data = dict.fit_transform(
        [{'city': '北京', 'temperature': 100},
         {'city': '上海', 'temperature': 60},
         {'city': '深圳', 'temperature': 30}]
    )
    print(dict.get_feature_names())
    # 字典数据抽取　把字典中一些类别数据　分别进行转换成数据特征
    print(dict.inverse_transform(data))
    print(data)
    # scipy.sparse.csr.csr_matrix 节约内存　方便读取处理
    print(type(data))
    return None


if __name__ == "__main__":
    dictvec()