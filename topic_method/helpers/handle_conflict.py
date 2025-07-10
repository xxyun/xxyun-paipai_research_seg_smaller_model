from topic_method.helpers.calculate_similarity import average_cluster_similarity


def compute_overlap(first_list, second_list):
    """
    该函数，接受两个有序数列作为参数，
    计算重叠部分，最后得到：重叠部分，左侧剩余部分和右侧剩余部分
    :param first_list: 第一个有序数列表
    :param second_list: 第二个有序数列表
    :return:
    """
    common, left, right = [], [], []
    i, j = 0, 0
    # 循环遍历两个数列，直到其中一个指针超出范围
    while i < len(first_list) and j < len(second_list):
        # 如果两个数列的当前元素相等，则将其添加到重叠部分，并更新两个指针
        if first_list[i] == second_list[j]:
            common.append(first_list[i])
            i += 1
            j += 1
        # 如果两个数列的当前元素不等，则比较它们的大小
        else:
            # 如果a的当前元素小于b的当前元素，则将其添加到左侧剩余部分，并更新a的指针
            if first_list[i] < second_list[j]:
                left.append(first_list[i])
                i += 1
            # 如果a的当前元素大于b的当前元素，则将其添加到右侧剩余部分，并更新b的指针
            else:
                right.append(second_list[j])
                j += 1
    # 循环结束后，将a和b中剩余的元素分别添加到左侧和右侧剩余部分
    left.extend(first_list[i:])
    right.extend(second_list[j:])
    # 返回三个列表作为结果
    return common, left, right


def handle_overlap(lst, similarity_matrix, overlap_threshold):
    """
    对于列表lst，基于相似度，对于前后有重叠的连续数组进行处理。
    :param lst:  含有多个连续数组的list
    :param similarity_matrix:  相似度矩阵
    :param overlap_threshold:  相似度阈值，超过该阈值则予以合并
    :return:
    """
    # 创建一个空列表 result ，用于存放分割后的子列表
    result = []
    # 遍历原始列表，每次取出两个相邻的子列表
    for i in range(len(lst)-1):

        if len(result) > 0:
            next_one = result[-1]
            del result[-1]
        else:
            next_one = lst[0]

        next_two = lst[i + 1]
        next_two = [item for item in next_two if item >= next_one[0]]

        common, sublist1, sublist2 = compute_overlap(next_one, next_two)

        if len(common) > 0:

            if len(sublist1) == 0:
                sim = average_cluster_similarity(common, sublist2, similarity_matrix)

                if sim >= overlap_threshold:
                    result.append(next_two)
                else:
                    result.append(common)
                    result.append(sublist2)

            elif len(sublist2) == 0:
                sim = average_cluster_similarity(common, sublist1, similarity_matrix)
                if sim >= overlap_threshold:
                    result.append(next_one)
                else:
                    result.append(sublist1)
                    result.append(common)

            # 如果它们有重叠部分，则将重叠部分取出，把第一、第二个子列表进行拆分
            else:
                # 调用 compute_gain 函数，分别计算重叠部分与第一、第二个子列表的相似度
                sim_one = average_cluster_similarity(common, sublist1, similarity_matrix)
                sim_two = average_cluster_similarity(common, sublist2, similarity_matrix)
                # 相似度高的部分保留该子列表，相似度低那一侧则从子列表中将重叠部分剔除

                if sim_one > sim_two:
                    result.append(sublist1 + common)
                    result.append(sublist2)

                else:
                    result.append(sublist1)
                    result.append(common + sublist2)

        else:
            # 如果没有重叠部分，则直接将子列表加入到结果中
            result.append(sublist1)
            result.append(sublist2)

    return result
