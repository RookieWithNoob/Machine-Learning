from sklearn.feature_extraction.text import CountVectorizer


def countvec():
    """
    文本特征抽取
    :return: None
    """
    text = CountVectorizer()

    # data = text.fit_transform(
    #     ["life is short,i like python",
    #      "life is too long,i dislike python"]
    # )
    # 中文
    data = text.fit_transform(
        ["人生 苦短,我 喜欢 python",
         "人生 漫长,不用 python"]
    )

    # 单个字母(汉字)不会统计
    print(data.toarray()) # 统计每个词出现的次数
    # 统计所有文章当中所有的词　重复的只看做一次
    print(text.get_feature_names()) # 词的列表 8个
    print(type(data.toarray()))

    return None


if __name__ == "__main__":
    countvec()