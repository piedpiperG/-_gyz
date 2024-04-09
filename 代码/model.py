# -*- coding: utf-8 -*-

# 正则包
import re
# html 包
import html
# 自然语言处理包
import jieba
import jieba.analyse
# 机器学习包
from sklearn.metrics.pairwise import cosine_similarity
# 编辑距离包
import Levenshtein
# 数据集处理包
from datasketch import MinHash
import math


class CosineSimilarity(object):
    """
    余弦相似度
    """

    def __init__(self, content_x1, content_y2):
        self.s1 = content_x1
        self.s2 = content_y2

    @staticmethod
    def extract_keyword(content):  # 提取关键词
        # 切割
        seg = [i for i in jieba.cut(content, cut_all=True) if i != '']
        # 提取关键词
        keywords = jieba.analyse.extract_tags("|".join(seg), topK=4, withWeight=False)
        return keywords

    @staticmethod
    def one_hot(word_dict, keywords):  # oneHot编码
        # cut_code = [word_dict[word] for word in keywords]
        cut_code = [0] * len(word_dict)
        for word in keywords:
            cut_code[word_dict[word]] += 1
        return cut_code

    def main(self):
        # 去除停用词
        jieba.analyse.set_stop_words('data/stopwords.txt')

        # 提取关键词
        keywords1 = self.extract_keyword(self.s1)
        keywords2 = self.extract_keyword(self.s2)
        # 词的并集
        union = set(keywords1).union(set(keywords2))
        # 编码
        word_dict = {}
        i = 0
        for word in union:
            word_dict[word] = i
            i += 1
        if len(word_dict) == 0:
            return 0.0
        # oneHot编码
        s1_cut_code = self.one_hot(word_dict, keywords1)
        s2_cut_code = self.one_hot(word_dict, keywords2)
        # 余弦相似度计算
        sample = [s1_cut_code, s2_cut_code]
        # 除零处理
        try:
            sim = cosine_similarity(sample)
            return sim[1][0]
        except Exception as e:
            print(e)
            return 0.0


class JaccardSimilarity(object):
    """
    jaccard相似度
    """

    def __init__(self, content_x1, content_y2):
        self.s1 = content_x1
        self.s2 = content_y2

    @staticmethod
    def extract_keyword(content):  # 提取关键词
        # 切割
        seg = [i for i in jieba.cut(content, cut_all=True) if i != '']
        # 提取关键词
        keywords = jieba.analyse.extract_tags("|".join(seg), topK=4, withWeight=False)
        return keywords

    def main(self):
        # 去除停用词
        jieba.analyse.set_stop_words('data/stopwords.txt')

        # 分词与关键词提取
        keywords_x = self.extract_keyword(self.s1)
        keywords_y = self.extract_keyword(self.s2)

        # jaccard相似度计算
        intersection = len(list(set(keywords_x).intersection(set(keywords_y))))
        union = len(list(set(keywords_x).union(set(keywords_y))))
        # 除零处理
        sim = float(intersection) / union if union != 0 else 0
        return sim


class LevenshteinSimilarity(object):
    """
    编辑距离
    """

    def __init__(self, content_x1, content_y2):
        self.s1 = content_x1
        self.s2 = content_y2

    @staticmethod
    def extract_keyword(content):  # 提取关键词
        # 切割
        seg = [i for i in jieba.cut(content, cut_all=True) if i != '']
        # 提取关键词
        keywords = jieba.analyse.extract_tags("|".join(seg), topK=4, withWeight=False)
        return keywords

    def main(self):
        # 去除停用词
        jieba.analyse.set_stop_words('data/stopwords.txt')

        # 提取关键词
        keywords1 = ', '.join(self.extract_keyword(self.s1))
        keywords2 = ', '.join(self.extract_keyword(self.s2))

        # ratio计算2个字符串的相似度，它是基于最小编辑距离
        distances = Levenshtein.ratio(keywords1, keywords2)
        return distances


class MinHashSimilarity(object):
    """
    MinHash
    """

    def __init__(self, content_x1, content_y2):
        self.s1 = content_x1
        self.s2 = content_y2

    @staticmethod
    def extract_keyword(content):  # 提取关键词
        # 切割
        seg = [i for i in jieba.cut(content, cut_all=True) if i != '']
        # 提取关键词
        keywords = jieba.analyse.extract_tags("|".join(seg), topK=4, withWeight=False)
        return keywords

    def main(self):
        # 去除停用词
        jieba.analyse.set_stop_words('data/stopwords.txt')

        # MinHash计算
        m1, m2 = MinHash(), MinHash()
        # 提取关键词
        s1 = self.extract_keyword(self.s1)
        s2 = self.extract_keyword(self.s2)

        for data in s1:
            m1.update(data.encode('utf8'))
        for data in s2:
            m2.update(data.encode('utf8'))

        return m1.jaccard(m2)


class SimHashSimilarity(object):
    """
    SimHash
    """

    def __init__(self, content_x1, content_y2):
        self.s1 = content_x1
        self.s2 = content_y2

    @staticmethod
    def get_bin_str(source):  # 字符串转二进制
        if source == "":
            return 0
        else:
            t = ord(source[0]) << 7
            m = 1000003
            mask = 2 ** 128 - 1
            for c in source:
                t = ((t * m) ^ ord(c)) & mask
            t ^= len(source)
            if t == -1:
                t = -2
            t = bin(t).replace('0b', '').zfill(64)[-64:]
            return str(t)

    @staticmethod
    def extract_keyword(content):  # 提取关键词
        # 切割
        seg = [i for i in jieba.cut(content, cut_all=True) if i != '']
        # 提取关键词
        keywords = jieba.analyse.extract_tags("|".join(seg), topK=4, withWeight=True)
        return keywords

    def run(self, keywords):
        ret = []
        for keyword, weight in keywords:
            bin_str = self.get_bin_str(keyword)
            key_list = []
            for c in bin_str:
                weight = math.ceil(weight)
                if c == "1":
                    key_list.append(int(weight))
                else:
                    key_list.append(-int(weight))
            ret.append(key_list)
        # 对列表进行"降维"
        rows = len(ret)
        cols = len(ret[0])
        result = []
        for i in range(cols):
            tmp = 0
            for j in range(rows):
                tmp += int(ret[j][i])
            if tmp > 0:
                tmp = "1"
            elif tmp <= 0:
                tmp = "0"
            result.append(tmp)
        return "".join(result)

    def main(self):
        # 去除停用词
        jieba.analyse.set_stop_words('data/stopwords.txt')

        # 提取关键词
        s1 = self.extract_keyword(self.s1)
        s2 = self.extract_keyword(self.s2)

        sim_hash1 = self.run(s1)
        sim_hash2 = self.run(s2)
        # print(f'相似哈希指纹1: {sim_hash1}\n相似哈希指纹2: {sim_hash2}')
        length = 0
        for index, char in enumerate(sim_hash1):
            if char == sim_hash2[index]:
                continue
            else:
                length += 1
        return length
