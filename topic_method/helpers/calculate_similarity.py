from text2vec import cos_sim
from scipy.sparse import dok_matrix
from genutility.math import minmax
import numpy as np
import requests


def cut_sentences(content):
    # 结束符号，包含中文和英文的
    end_flag = ['？', '！', '。', '…']
    content_len = len(content)
    sentences = []
    tmp_char = ''
    for idx, char in enumerate(content):
        # 拼接字符
        tmp_char += char
        # 判断是否已经到了最后一位
        if (idx + 1) == content_len:
            sentences.append(tmp_char)
            break
        # 判断此字符是否为结束符号
        if char in end_flag:
            next_idx = idx + 1
            if content[next_idx]=='”':  #如果是前面一个位置是结束符号，后一个位置是”，则分句时，将”放在前一句话
                tmp_char += content[next_idx]
            if content[next_idx] not in end_flag:
            # 再判断下一个字符是否为结束符号，如果不是结束符号，则切分句子
                if tmp_char[0]=='”':  #去掉第一个位置的”
                    tmp_char = tmp_char[1:]
                sentences.append(tmp_char)
                tmp_char = ''
    return sentences


def fetch_embedding(full_text, segment_model):
    """
    获取embedding的向量
    """
    matrix = segment_model.encode(full_text)
    return matrix


def calculate_similarity_matrix(snippets, segment_model, localization_size=100, pool_mean=False):
    """
    通过embedding获得每个文本片段的
    :param snippets: 文本片段
    :param t2v_model: embedding模型
    :param localization_size: 计算上下文长度
    :param pool_mean: 是否采用pool_mean方法，通过平均值计算embedding
    :return: 相似度矩阵，相似度的均值
    """

    similarity_matrix = dok_matrix((len(snippets), len(snippets)), dtype=np.float)

    if pool_mean:

        embeddings = list()
        for item in snippets:
            cut_items = cut_sentences(item)
            mean_embedding = fetch_embedding(cut_items, segment_model).mean(axis=0)
            embeddings.append(mean_embedding)

    else:
        embeddings = fetch_embedding(snippets, segment_model)

    val = list()
    for i in range(0, len(snippets) - 1):
        for j in range(i + 1, min(len(snippets), i + localization_size)):
            cos_res = cos_sim(embeddings[i], embeddings[j])
            similarity_matrix[i, j] = cos_res
            val.append(cos_res.item())

    return similarity_matrix, np.asarray(val).mean()


def calculate_similarity_matrix_array(snippets, segment_model, localization_size=100, pool_mean=False):
    """
    通过embedding获得每个文本片段的
    :param snippets: 文本片段
    :param t2v_model: embedding模型
    :param localization_size: 计算上下文长度
    :param pool_mean: 是否采用pool_mean方法，通过平均值计算embedding
    :return: 相似度矩阵，相似度的均值
    """

    similarity_matrix = dok_matrix((len(snippets), len(snippets)), dtype=np.float64)

    if pool_mean:
        embeddings = list()
        for item in snippets:
            cut_items = cut_sentences(item)
            mean_embedding = fetch_embedding(cut_items, segment_model).mean(axis=0)
            embeddings.append(mean_embedding)

    else:
        embeddings = fetch_embedding([item["sentence"] for item in snippets], segment_model)

    val = list()
    for i in range(0, len(snippets) - 1):
        for j in range(i + 1, min(len(snippets), i + localization_size)):
            cos_res = cos_sim(embeddings[i], embeddings[j])
            similarity_matrix[i, j] = cos_res
            val.append(cos_res.item())

    return similarity_matrix, np.asarray(val).mean()


def average_cluster_similarity(first_list, second_list, similarity_matrix):
    """
    该函数计算两个团之间的相似度 = 所有句子间两两相似度之和 / (团1个数）* (团2个数）
    :param first_list: 第一个团元素列表
    :param second_list: 第二个团元素列表
    :param similarity_matrix: 相似度矩阵
    :return: 两个团之间的相似度
    """
    sum_val = 0.0
    for first_item in first_list:
        for second_item in second_list:
            x, y = minmax(first_item, second_item)
            sum_val += similarity_matrix[x, y]

    return sum_val / (len(first_list) * len(second_list))


def get_closet_two_decimal(value):

    integer = int(value * 100 + 0.5)
    if integer % 10 == 0:
        return value

    else:
        return (integer + 1) / 100
