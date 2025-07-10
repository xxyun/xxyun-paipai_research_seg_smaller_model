from topic_method.helpers.calculate_similarity import average_cluster_similarity


def fill_first_and_last_gap(lst, similarity_matrix,  snippet_max_val,  gap_threshold):
    """
    检测开头和结尾 是否有空档，根据相似度的条件，进行合并
    :param lst: 含有连续数列的数组
    :param snippet_max_val: 文本片段最大值，用于检测末尾是否有间隙
    :param similarity_matrix: 相似度矩阵
     :param gap_threshold: 合并的相似度阈值
    :return:
    """
    result = []
    snippet_min_val = 0

    first_val = lst[0][0]
    last_val = lst[-1][-1]

    if first_val > snippet_min_val:

        sublist1 = list(range(0, first_val))
        sublist2 = lst[0]

        similarity = average_cluster_similarity(sublist1,sublist2, similarity_matrix)

        if similarity >= gap_threshold:
            result.append(sublist1 + sublist2)

        else:
            result.append(sublist1)
            result.append(sublist2)
    else:
        result.append(lst[0])

    result.extend(lst[1:-1])

    if last_val < snippet_max_val:

        sublist1 = lst[-1]
        sublist2 = list(range(last_val+1, snippet_max_val+1))

        similarity = average_cluster_similarity(sublist1,sublist2,similarity_matrix)

        if similarity >= gap_threshold:
            result.append(sublist1 + sublist2)

        else:
            result.append(sublist1)
            result.append(sublist2)
    else:
        result.append(lst[-1])

    return result


def gap_detection(list_one, list_two):
    """
    根据两个排好序的有序列表，检测中间是否存在可插入的空隙
    :param list_one: 第一个有序数序列
    :param list_two: 第二个有序数序列
    :return:
    """
    # 如果a或b为空，或者a的最大值大于等于b的最小值，返回空列表
    if not list_one or not list_two or list_one[-1] >= list_two[0]:
        return []
    # 否则，创建一个空列表gap，从a的最大值加一开始，到b的最小值减一结束，依次添加到gap中
    gap = []
    for i in range(list_one[-1]+1, list_two[0]):
        gap.append(i)
    # 返回gap
    return gap


# ============= 合并区间中有gap的信息 ================

# 定义一个函数 split_list ，用于对一个列表进行分割
def merge_gap_list(lst, similarity_matrix, gap_threshold):
    """
    对于列表中元素，检查是否有可以插入的空隙，如果有则插入。
    之后与左右两边元素计算相似度，如果大于阈值则合并，否则单独处理。
    :param lst: 含有连续数值的数组
    :param similarity_matrix: 相似度矩阵
    :param gap_threshold: 合并的相似度阈值
    :return:
    """
    # 创建一个空列表 result ，用于存放分割后的子列表
    result = []
    # 遍历原始列表，每次取出两个相邻的子列表

    for i in range(len (lst) - 1):

        if len(result) > 0:
            next_one = result[-1]
            del result[-1]
        else:
            next_one = lst[0]
        next_two = lst[i + 1]

        sublist1 = next_one
        sublist2 = next_two
        gap = gap_detection(sublist1, sublist2)

        # 如果它们有重叠部分，则将重叠部分取出，把第一、第二个子列表进行拆分
        if len(gap) > 0:
            # 调用 compute_gain 函数，分别计算重叠部分与第一、第二个子列表的相似度
            sim_one = average_cluster_similarity(gap, sublist1, similarity_matrix)
            sim_two = average_cluster_similarity(gap, sublist2, similarity_matrix)
            # 相似度高的则保留该子列表，相似度低的则从该子列表中将重叠部分剔除

            if max(sim_one, sim_two) < gap_threshold:
                # 如果相似度都小于阈值，则直接插入进去
                result.append(sublist1)
                result.append(gap)
                result.append(sublist2)

            elif sim_one >= sim_two:
                result.append(sublist1 + gap)
                result.append(sublist2)

            elif sim_one < sim_two:
                result.append(sublist1)
                result.append(gap + sublist2)

        else:
            # 如果没有重叠部分，则直接将子列表加入到结果中
            result.append (sublist1)
            result.append(sublist2)

    # 返回 result 列表
    return sorted(result)


def merge_clusters(lst, similarity_matrix, cluster_combine_threshold):
    """
    对最后的列表进行检查，如果超过阈值，则前后进行合并
    :param lst: 含有连续数值的数组
    :param similarity_matrix: 相似度矩阵
    :param cluster_combine_threshold: 合并的相似度阈值
    :return:
    """
    # 创建一个空列表 result ，用于存放分割后的子列表
    result = []
    # 遍历原始列表，每次取出两个相邻的子列表
    i = 0
    while i < len (lst) - 1:
        sublist1 = lst [i]
        sublist2 = lst [i + 1]
        # 调用 compute_gain 函数，分别计算重叠部分与第一、第二个子列表的相似度
        sim = average_cluster_similarity(sublist1,sublist2, similarity_matrix)

        # 相似度高的则保留该子列表，相似度低的则从该子列表中将重叠部分剔除
        if sim > cluster_combine_threshold:
            # 如果块相似度大于阈值，则直接合并
            result.append(sublist1+sublist2)

        else:
            result.append(sublist1)
            result.append(sublist2)

        i += 2

    if i == len (lst) - 1:
        result.append(list(lst[i]))
    # 返回 result 列表
    return sorted(result)


def merge_with_speakers(cluster_lst, speaker_list, loaded_txt):
    """
    当前segment的speaker与前面segment的speaker一致的文本，则予以合并
    :param cluster_lst: 基于语义已经合并成团后的index序列
    :param speaker_list: 演讲人身份的列表信息
    :return:
    """

    if len(speaker_list) == 0:
        return cluster_lst

    combine_res = list()

    for i in range(len(cluster_lst) - 1):

        if len(combine_res) > 0:
            next_one = combine_res[-1]
            del combine_res[-1]
        else:
            next_one = cluster_lst[0]

        next_two = cluster_lst[i + 1]
        last_element_idx = next_one[-1]
        next_first_element_idx = next_two[0]
        if speaker_list[last_element_idx] == speaker_list[next_first_element_idx] and (
                len(next_one) <= 3 or len(next_two) <= 3):
            combine_res.append(next_one+next_two)
        elif loaded_txt[next_one[-1]].endswith('?'):
            combine_res.append(next_one+next_two)
        else:
            combine_res.append(next_one)
            combine_res.append(next_two)

    return combine_res
