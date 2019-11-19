from sklearn.feature_extraction.text import CountVectorizer
import jieba


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
    # 中文　进行分词
    data = text.fit_transform(
        ["人生 苦短,我 喜欢 python",
         "人生 漫长,不用 python"]
    )

    # 单个字母(汉字)不会统计
    print(data.toarray())  # 统计每个词出现的次数
    # 统计所有文章当中所有的词　重复的只看做一次
    print(text.get_feature_names())  # 词的列表 8个
    print(type(data.toarray()))

    return None


def cutword():
    """
        1、今天很残酷，明天更残酷，后天很美好，
        但绝对大部分是死在明天晚上，所以每个人不要放弃今天。

        2、我们看到的从很远星系来的光是在几百万年之前发出的，
        这样当我们看到宇宙时，我们是在看它的过去。

        3、如果只用一种方式了解某样事物，你就不会真正了解它。
        了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。
        """
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")
    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")
    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")
    # 转换成列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)
    # 把列表转换成字符串
    c1 = " ".join(content1)
    c2 = " ".join(content2)
    c3 = " ".join(content3)
    # print(c1)
    # print(c2)
    # print(c3)

    return c1, c2, c3


def chinese_valuesvec():
    """
    1、今天很残酷，明天更残酷，后天很美好，
    但绝对大部分是死在明天晚上，所以每个人不要放弃今天。

    2、我们看到的从很远星系来的光是在几百万年之前发出的，
    这样当我们看到宇宙时，我们是在看它的过去。

    3、如果只用一种方式了解某样事物，你就不会真正了解它。
    了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。
    """
    cv = CountVectorizer()

    c1, c2, c3 = cutword()

    data = cv.fit_transform([c1, c2, c3])

    print(data.toarray())

    print(cv.get_feature_names())


if __name__ == "__main__":
    # countvec()
    # c1, c2, c3 = cutword()
    chinese_valuesvec()
