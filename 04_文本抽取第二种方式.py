from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from sklearn.preprocessing import MinMaxScaler

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


def tfidfvec():
    """
    1、今天很残酷，明天更残酷，后天很美好，
    但绝对大部分是死在明天晚上，所以每个人不要放弃今天。

    2、我们看到的从很远星系来的光是在几百万年之前发出的，
    这样当我们看到宇宙时，我们是在看它的过去。

    3、如果只用一种方式了解某样事物，你就不会真正了解它。
    了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。
    """

    # tf * itf log(数值) 数值=文档数量/该文档中词的数量　词越多重要程度越小 tfitf 更好　但是都已经过时了
    cv = TfidfVectorizer()

    c1, c2, c3 = cutword()

    data = cv.fit_transform([c1, c2, c3])

    print(data.toarray())

    print(cv.get_feature_names())

    return None

def mm():
    """
    归一化处理
    :return:None
    """
    pass


if __name__ == "__main__":
    tfidfvec()