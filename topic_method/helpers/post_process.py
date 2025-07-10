def detect_consec_for_each_clique(nums):
    """
    该函数用来找到一个列表中的所有连续数字组合，并返回一个新的列表
    :param nums:
    :return:
    """
    # 如果列表为空或只有一个元素，直接返回列表本身
    if not nums or len(nums) == 1:
        return [nums]
    # 对列表进行排序
    nums.sort()
    # 定义一个空列表，用来存储连续数字组合
    result = []
    # 定义一个临时列表，用来存储当前的连续数字组合
    temp = []
    # 遍历列表中的每个元素
    for num in nums:
        # 如果临时列表为空，或者当前元素和临时列表中的最后一个元素之间的差是1，将当前元素添加到临时列表中
        if not temp or num - temp[-1] == 1:
            temp.append(num)
            # 否则，说明当前元素和临时列表中的最后一个元素之间不连续，将临时列表添加到结果列表中，并清空临时列表，然后将当前元素添加到临时列表中
        else:
            result.append(temp)
            temp = [num]

    # 遍历完毕后，将最后一个临时列表添加到结果列表中
    result.append(temp)
    # 返回结果列表
    return result


def is_insertable(first_list, second_list):
    """
    该函数，用来判断两个列表是否可以插入一个值，使它们成为连续值
    :param first_list: 第一个有序数列
    :param second_list:第二个有序数列
    :return:
    """
    # 如果两个列表都只有1个元素，返回False，不插入
    if len(first_list) <= 1 and len(second_list) <= 1:
        return False
    # 如果两个列表中有一个为空，返回False
    if not first_list or not second_list:
        return False
    # 取出两个列表中的第一个元素和最后一个元素
    first = first_list[-1]
    last = second_list[0]
    # 如果第一个元素和最后一个元素相差1，返回True
    if abs(first - last) == 2:
        return True
    # 否则，返回False
    return False


def fill_consec_for_each_clique(lst_item):
    """
    对于每个团元素，中间相差一个元素的团填充差值并予以合并（左右至少有一个有2个以上元素）
    :param lst_item:
    :return:
    """
    # 定义一个变量，存储结果数组
    result = []

    i = 0
    while i < len(lst_item)-1:

        # 取出当前元素和下一个元素
        current_element = lst_item[i]
        next_element = lst_item[i+1]

        # 判断当前元素和下一个元素是否可以插入一个值，使它们成为连续值
        if is_insertable(current_element, next_element):
            # 如果可以，将当前元素和下一个元素合并为一个列表，并添加到结果数组中
            merged = current_element + [current_element[-1]+1] + next_element
            result.append(merged)
            # 跳过下一个元素的遍历
            i += 1
        else:
            # 如果不可以，将当前元素添加到结果数组中
            result.append(current_element)

        i += 1

    if i == len(lst_item)-1:
        result.append(lst_item[i])

    # 打印结果数组
    return result


def find_all_consecutive(lst):
    """
    从所有最大团元素中，获取所有连续元素，组成数列
    :param lst:
    :return:
    """
    # 创建一个空的set和list
    result_set = set()
    result_list = []

    # 遍历列表中的每个数组
    for arr in lst:
        for item in arr:
            # 如果数组的长度大于1
            if len(item) > 1:
                # 把数组转换为元组，因为set只能存储不可变的对象
                arr_tuple = tuple(item)
                # 如果元组不在结果set中
                if arr_tuple not in result_set:
                    # 把元组加入结果set和结果list
                    result_set.add(arr_tuple)
                    result_list.append(item)

    # 把结果set转换为list
    result_list = list(result_set)

    # 对结果list进行排序
    result_list.sort()

    # 打印结果list
    return result_list


def find_max_contained(array):
    """
    将相互包含的团进行合并，获得长度最大的团
    :param array:
    :return:
    """
    # initialize an empty list to store the output
    output = []

    # loop through the array
    for i in range(len(array)):
        # assume the current element is the longest one that contains others
        longest = True
        # loop through the rest of the array
        for j in range(len(array)):
            # skip the same element
            if i == j:
                continue
            # check if the current element is contained by another element
            if set(array[i]).issubset(set(array[j])):
                # if yes, then it is not the longest one
                longest = False
                # break the inner loop
                break
        # if the current element is the longest one that contains others
        if longest:
            # append it to the output list
            output.append(array[i])

    # print the output list
    sorted_res = sorted(output)
    return [tuple(item) for item in sorted_res]


