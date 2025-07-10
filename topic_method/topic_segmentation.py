import re
from typing import List

from topic_method.helpers.pre_process import extract_speaker_and_content
from topic_method.helpers.calculate_similarity import calculate_similarity_matrix, get_closet_two_decimal, \
    calculate_similarity_matrix_array
from util.bbox_util import merge_bboxes
from networkx import find_cliques_recursive, from_scipy_sparse_matrix
from topic_method.helpers.post_process import detect_consec_for_each_clique, \
    fill_consec_for_each_clique, find_all_consecutive, find_max_contained
from topic_method.helpers.handle_conflict import handle_overlap
from topic_method.helpers.merge_list import fill_first_and_last_gap, merge_gap_list, merge_clusters, merge_with_speakers
from pdf_parser.pdf_parsing import detect_heading_line
from functools import reduce


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
            if content[next_idx] == '”':
                tmp_char += content[next_idx]
            if content[next_idx] not in end_flag:
                # 再判断下一个字符是否为结束符号，如果不是结束符号，则切分句子
                if tmp_char[0] == '”':
                    tmp_char = tmp_char[1:]
                sentences.append(tmp_char)
                tmp_char = ''
    return sentences


def maximal_clique_seg_text(loaded_txt, segment_model):
    """
    大文本语义分割的入口
    :param loaded_txt: 转入的列表
    :return: 进行语义合并后的文本信息
    """
    loaded_txt = cut_sentences(loaded_txt)
    # 读取文件目录对应的文本信息
    # loaded_txt = load_file(file_dir)
    # 根据规则，抽取出演讲人列表，以及内容列表
    speaker_list, content_list = extract_speaker_and_content(loaded_txt)
    max_length = len(content_list) - 1

    # 对于获取到的文本段落，计算相似度矩阵
    # t2v_model = SentenceModel(embedding_model_path, max_seq_length=500)
    similarity_matrix, similarity_mean = calculate_similarity_matrix(content_list, segment_model, 100, False)

    # 基于相似度矩阵，动态获取后续的相似度阈值
    graph_threshold = get_closet_two_decimal(similarity_mean)  # 阈值选取比相似度均值大的最接近的两位小数
    gap_combine_threshold = graph_threshold - 0.05
    cluster_combine_threshold = graph_threshold + 0.05

    #  将大于阈值的相似文本，通过边连接，并构造子图，利用Bron-Kerbosch算法获取最大图列表
    graph = from_scipy_sparse_matrix(similarity_matrix > graph_threshold)
    cliques = list(find_cliques_recursive(graph))

    # 接下来对于最大团列表进行后处理，抽取出整个最大团列表中最大的连续数列
    # 此处对于每个最大团元素，识别出其中的连续元素
    clique_with_consecutive = [detect_consec_for_each_clique(item) for item in cliques]
    # 同一个团中差1的元素进行补充：对于每个团元素，中间相差一个元素的团填充差值并予以合并（左右至少有一个有2个以上元素）
    filled_cliques = [fill_consec_for_each_clique(arr) for arr in clique_with_consecutive]

    # 从所有最大团元素中，获取所有连续元素，组成数列
    all_consecs = find_all_consecutive(filled_cliques)

    # 将相互包含的团进行合并，获得长度最大的团
    merged_list = find_max_contained(all_consecs)
    # 接下来对于存在重叠的团进行合并，基于左右的相似度，像泡泡一样进行粘合处理
    list_without_overlap = handle_overlap(merged_list, similarity_matrix, gap_combine_threshold)
    # 接下来对元素逐个进行扫描，进行合并
    # 第一步，查着开头和结尾是否有空隙，并且基于相似度进行合并
    lst_without_first_last_gap = fill_first_and_last_gap(list_without_overlap, similarity_matrix,
                                                         max_length, gap_combine_threshold)
    # 第二步，扫描所有元素，对于中间有空隙的数列，填充空隙，并且基于相似度阈值进行合并
    lst_without_gap = merge_gap_list(lst_without_first_last_gap, similarity_matrix, gap_combine_threshold)
    # 最后一步，再检查一遍所有的文本团，看看是否可以基于阈值再进行一次合并
    combined_clusters = merge_clusters(lst_without_gap, similarity_matrix, cluster_combine_threshold)
    # merged_result = merge_with_speakers(combined_clusters, speaker_list, loaded_txt)

    # 将团序列与原始文本映射，获得最终的文本切分结果
    res = list()
    for seg in combined_clusters:
        res.append([loaded_txt[ind] for ind in seg])

    return res


def maximal_clique_seg_array(loaded_array, segment_model):
    """
    大文本语义分割的入口
    :param loaded_txt: 转入的列表
    :return: 进行语义合并后的文本信息
    """

    # 读取文件目录对应的文本信息
    # loaded_txt = load_file(file_dir)
    # 根据规则，抽取出演讲人列表，以及内容列表
    max_length = len(loaded_array) - 1

    # 对于获取到的文本段落，计算相似度矩阵
    # t2v_model = SentenceModel(embedding_model_path, max_seq_length=500)
    similarity_matrix, similarity_mean = calculate_similarity_matrix_array(loaded_array, segment_model, 100, False)

    # 基于相似度矩阵，动态获取后续的相似度阈值
    graph_threshold = get_closet_two_decimal(similarity_mean)  # 阈值选取比相似度均值大的最接近的两位小数
    gap_combine_threshold = graph_threshold - 0.05
    cluster_combine_threshold = graph_threshold + 0.05

    #  将大于阈值的相似文本，通过边连接，并构造子图，利用Bron-Kerbosch算法获取最大图列表
    graph = from_scipy_sparse_matrix(similarity_matrix > graph_threshold)
    cliques = list(find_cliques_recursive(graph))

    # 接下来对于最大团列表进行后处理，抽取出整个最大团列表中最大的连续数列
    # 此处对于每个最大团元素，识别出其中的连续元素
    clique_with_consecutive = [detect_consec_for_each_clique(item) for item in cliques]
    # 同一个团中差1的元素进行补充：对于每个团元素，中间相差一个元素的团填充差值并予以合并（左右至少有一个有2个以上元素）
    filled_cliques = [fill_consec_for_each_clique(arr) for arr in clique_with_consecutive]

    # 从所有最大团元素中，获取所有连续元素，组成数列
    all_consecs = find_all_consecutive(filled_cliques)

    # 将相互包含的团进行合并，获得长度最大的团
    merged_list = find_max_contained(all_consecs)
    # 接下来对于存在重叠的团进行合并，基于左右的相似度，像泡泡一样进行粘合处理
    list_without_overlap = handle_overlap(merged_list, similarity_matrix, gap_combine_threshold)
    # 接下来对元素逐个进行扫描，进行合并
    # 第一步，查着开头和结尾是否有空隙，并且基于相似度进行合并
    lst_without_first_last_gap = fill_first_and_last_gap(list_without_overlap, similarity_matrix,
                                                         max_length, gap_combine_threshold)
    # 第二步，扫描所有元素，对于中间有空隙的数列，填充空隙，并且基于相似度阈值进行合并
    lst_without_gap = merge_gap_list(lst_without_first_last_gap, similarity_matrix, gap_combine_threshold)
    # 最后一步，再检查一遍所有的文本团，看看是否可以基于阈值再进行一次合并
    combined_clusters = merge_clusters(lst_without_gap, similarity_matrix, cluster_combine_threshold)
    # merged_result = merge_with_speakers(combined_clusters, speaker_list, loaded_txt)

    # 将团序列与原始文本映射，获得最终的文本切分结果
    res = list()
    for seg in combined_clusters:
        res.append([loaded_array[ind] for ind in seg])

    return res


def simple_segment_text(full_text, max_length_limit, min_length):
    combine_res = '\n'.join(full_text)
    if len(combine_res) <= max_length_limit:
        return [full_text]
    return greedy_merge_by_tokens(full_text, max_length_limit, min_length)


def simple_segment_array(seg_array, max_length_limit, min_length):
    total_len = sum(len(s["sentence"]) for s in seg_array)
    if total_len <= max_length_limit:
        merged_items = {
            "sentence": ' '.join(it["sentence"] for it in seg_array),
            "bbox": merge_bboxes([it["bbox"] for it in seg_array]),
        }
        return [merged_items]
    grouped = greedy_merge_by_tokens_for_array(seg_array, max_length_limit, min_length)  # List[List[dict]]
    merged_list = [{
        "sentence": ' '.join([it["sentence"] for it in grp]),
        "bbox": merge_bboxes([it["bbox"] for it in grp]),
    } for grp in grouped]
    return merged_list


def _check_length_limit_and_re_seg(seg_text_list, max_length_limit, min_length):
    combine_res = list()
    for item in seg_text_list:
        combine_res.append(' '.join(item))
    merged_res = simple_segment_text(combine_res, max_length_limit, min_length)
    return merged_res


def _check_length_limit_and_re_seg_array(seg_items, max_length_limit, min_length):
    combine_res = list()
    for items in seg_items:
        merged_items = {"sentence": ' '.join([item["sentence"] for item in items]),
                        "bbox": merge_bboxes([item["bbox"] for item in items])}
        combine_res.append(merged_items)
    merged_res = simple_segment_array(combine_res, max_length_limit, min_length)
    return merged_res


def pattern_segment(full_text):
    qa_seg_trigger_pattern: str = r'((下面).*有请.*(电话(尾号)).*)|(如需提问.*)' \
                                  r'(进入(到)?(交流)?提问(互动)?(的)?环节|开放投资者提问|播报(一下)?提问方式|' \
                                  r'我这(边|面)(还)?(有|问)|(我)?(先|再|追)?问(一)?个((小)?问题)?|' \
                                  r'向您请教(一下|几个)|想再请教一下|大家好，如需提问|关于.*(的(一个)?)?问题' \
                                  r'(线上|下面)(由|请|有|是)?.*提问|有请.*提问|线上有.*举手提问|' \
                                  r'请发言，进入互动问答(的环节)?|在线的领导有什么问题|' \
                                  r'首先我先请教.*(问题)?|我想(跟|向)您请教|请教个(小)?问题|' \
                                  r'我(也)?想(问|请教)(一下)?|下面来自网络.*(文字)?提问)|' \
                                  r'想(再次|后面)?(问|请教)(一下|您)?|(来)?总结一下|' \
                                  r'再帮(我|我们)[^。，！？；：.,?;:!]*?说(一下)?|' \
                                  r'想(再|再次|后面|稍微|跟您)?(问|请教|追问)(一下|两个问题|您)?|(再|再次|后面|接着)(稍微|跟您)?(问|请教|追问)(一下|两个问题)?|(来)?(总结|补充|介绍)一下|(还有|最后|再)一个(小)?(问题|关于)|能不能再帮我们详细的就是说一下|请教一下'

    seg_pattern = qa_seg_trigger_pattern

    raw_sentences = full_text
    segments = []
    for sentences in raw_sentences:
        current_idx = 0
        sent_str = []
        for idx, sent_item in enumerate(sentences):
            if len(sentences) <= 3:
                break
            if (re.search(seg_pattern, sent_item) or detect_heading_line(sent_item)) and idx > 0:
                sent_str.append(sentences[current_idx:idx])
                current_idx = idx
        sent_str.append(sentences[current_idx:])
        segments.extend(sent_str)

    return segments


def pattern_segment_array(content_items_list):
    qa_seg_trigger_pattern: str = r'((下面).*有请.*(电话(尾号)).*)|(如需提问.*)' \
                                  r'(进入(到)?(交流)?提问(互动)?(的)?环节|开放投资者提问|播报(一下)?提问方式|' \
                                  r'我这(边|面)(还)?(有|问)|(我)?(先|再|追)?问(一)?个((小)?问题)?|' \
                                  r'向您请教(一下|几个)|想再请教一下|大家好，如需提问|关于.*(的(一个)?)?问题' \
                                  r'(线上|下面)(由|请|有|是)?.*提问|有请.*提问|线上有.*举手提问|' \
                                  r'请发言，进入互动问答(的环节)?|在线的领导有什么问题|' \
                                  r'首先我先请教.*(问题)?|我想(跟|向)您请教|请教个(小)?问题|' \
                                  r'我(也)?想(问|请教)(一下)?|下面来自网络.*(文字)?提问)|' \
                                  r'想(再次|后面)?(问|请教)(一下|您)?|(来)?总结一下|' \
                                  r'再帮(我|我们)[^。，！？；：.,?;:!]*?说(一下)?|' \
                                  r'想(再|再次|后面|稍微|跟您)?(问|请教|追问)(一下|两个问题|您)?|(再|再次|后面|接着)(稍微|跟您)?(问|请教|追问)(一下|两个问题)?|(来)?(总结|补充|介绍)一下|(还有|最后|再)一个(小)?(问题|关于)|能不能再帮我们详细的就是说一下|请教一下'

    seg_pattern = qa_seg_trigger_pattern

    raw_sentences = content_items_list
    segments = []
    for sentences in raw_sentences:
        current_idx = 0
        sent_str = []
        for idx, sent_item in enumerate(sentences):
            if len(sentences) <= 3:
                break
            if (re.search(seg_pattern, sent_item["sentence"]) or detect_heading_line(sent_item["sentence"])) and idx > 0:
                sent_str.append(sentences[current_idx:idx])
                current_idx = idx
        sent_str.append(sentences[current_idx:])
        segments.extend(sent_str)

    return segments


def split_text_trunk(content_list, max_tokens=300, min_length=100):
    """
    对于长度超的文本进行拆解
    """
    final_res = list()

    for content_item in content_list:
        if len(content_item) > max_tokens:
            cut_content = cut_sentences(content_item)
            merged_res = greedy_merge_by_tokens(cut_content, max_tokens, min_length)
            final_res.extend([' '.join(item) for item in merged_res])
        else:
            final_res.append(content_item)

    return final_res


def split_text_trunk_array(content_items_list, max_tokens: int = 300, min_length: int = 100):
    """
    再按字符长度（≈token 数）对片段做二次拆分。

    参数
    ----
    content_items_list : Union[List[dict], List[List[dict]]]
        - 每个 dict 至少包含 "sentence"、"bbox" 两字段
        - 既可以是 “多段的扁平列表”，也可以是 “段落嵌套列表”
    max_tokens : int
        拆分阈值，单条合并后若字符数 > max_tokens 就再拆
    min_length : int
        传给 greedy_merge_by_tokens_for_array 的最小段长
    返回
    ----
    List[dict]  # 每条 dict = 一段合并好的文本
    """
    final_res: List[dict] = []

    # ---------- 1. 形态归一化 ----------
    if not content_items_list:
        return final_res

    # 若传进来的是 List[dict]，包装成 List[List[dict]]
    if isinstance(content_items_list[0], dict):
        content_items_list = [content_items_list]

    # ---------- 2. 逐段处理 ----------
    for content_items in content_items_list:  # content_items 一定是 List[dict]
        total_chars = sum(len(it["sentence"]) for it in content_items)

        if total_chars <= max_tokens:
            # 不超限，整段直接合并
            merged = {
                "sentence": ' '.join([it["sentence"] for it in content_items]),
                "bbox": merge_bboxes([it["bbox"] for it in content_items]),
            }
            final_res.append(merged)

        else:
            # 超限，再用 greedy 策略拆
            grouped = greedy_merge_by_tokens_for_array(
                content_items, max_tokens, min_length)  # List[List[dict]]

            for grp in grouped:
                merged = {
                    "sentence": ' '.join([it["sentence"] for it in grp]),
                    "bbox": merge_bboxes([it["bbox"] for it in grp]),
                }
                final_res.append(merged)

    return final_res


def greedy_merge_by_tokens(full_text, max_length_limit, min_length):
    """
    这个函数的作用是为了把一些过长或过短的文本片段进行合理的划分，使得每个划分后的文本都接近max_token_limit。这样可以提高后续处理的效率和质量。
    """
    results = []
    seg_tokens = [len(i) for i in full_text]
    cum_token = 0
    current_idx = 0
    for idx, n_token in enumerate(seg_tokens):
        if cum_token != 0 and cum_token + n_token > max_length_limit and cum_token > min_length:
            results.append(full_text[current_idx:idx])
            cum_token = n_token
            current_idx = idx
        else:
            cum_token += n_token

    if len('\n'.join(full_text[current_idx:])) < min_length:
        results[-1].extend(full_text[current_idx:])
    else:
        results.append(full_text[current_idx:])

    return results

def greedy_merge_by_tokens_for_array(content_items, max_length_limit, min_length):
    """
    这个函数的作用是为了把一些过长或过短的文本片段进行合理的划分，
    使得每个划分后的文本都接近max_token_limit。这样可以提高后续处理的效率和质量。
    """
    results = []
    seg_tokens = [len(i["sentence"]) for i in content_items]
    cum_token = 0
    current_idx = 0
    for idx, n_token in enumerate(seg_tokens):
        if cum_token != 0 and cum_token + n_token > max_length_limit and cum_token > min_length:
            results.append(content_items[current_idx:idx])
            cum_token = n_token
            current_idx = idx
        else:
            cum_token += n_token


    if len(results) == 0:
        results.append(content_items[current_idx:])
    elif len('\n'.join(item["sentence"] for item in content_items[current_idx:])) < min_length:
        results[-1].extend(content_items[current_idx:])
    else:
        results.append(content_items[current_idx:])
    return results

def merge_qa(list_a):
    list_b = []
    i = 0 # 创建一个索引变量
    while i < len(list_a): # 遍历列表a
        item = list_a[i] # 取出当前元素
        if 'Q：' in item and len(item) < 100: # 判断是否含有'Q:'且长度小于100
            if i + 1 < len(list_a): # 判断是否有下一个元素
                item += list_a[i + 1] # 如果有，就合并
                i += 1 # 索引加一，跳过下一个元素
        list_b.append(item) # 将处理后的元素加入列表b
        i += 1
    return list_b


def merge_qa_array(list_a):
    list_b = []
    i = 0 # 创建一个索引变量
    while i < len(list_a): # 遍历列表a
        item = list_a[i]["sentence"] # 取出当前元素
        if 'Q：' in item and len(item) < 100: # 判断是否含有'Q:'且长度小于100
            if i + 1 < len(list_a): # 判断是否有下一个元素
                item += list_a[i + 1]["sentence"] # 如果有，就合并
                i += 1 # 索引加一，跳过下一个元素
        list_a[i]["sentence"] = item
        list_b.append(list_a[i]) # 将处理后的元素加入列表b
        i += 1
    return list_b


def adjust_qa_format(list_res):

    for idx, line_item in enumerate(list_res):
        first_key = -1
        if idx > 0: # 加上这个判断条件
            line_item = line_item.replace(":", "：")
            first_key = line_item.find('Q：', 1)
            if first_key > 0 and len(line_item[first_key:]) > 50:
                list_res[idx-1] = list_res[idx-1] + line_item[:first_key]
                list_res[idx] = line_item[first_key:]
            elif first_key == -1 and list_res[idx-1].rfind('Q：') != -1:
                key = list_res[idx-1].rfind('Q：', 1)
                if len(list_res[idx-1][key:]) <= 50:
                    list_res[idx] = list_res[idx-1][key:] + list_res[idx]
                    list_res[idx-1] = list_res[idx-1][:key]
    list_res = [item.replace(' ', '') for item in list_res if len(item.replace(' ', '')) > 0]
    merged_res = merge_qa(list_res)

    return merged_res


def adjust_qa_format_array(list_res):
    for idx, line_item in enumerate(list_res):
        if idx > 0: # 加上这个判断条件
            line = line_item["sentence"].replace(":", "：")
            prev_sentence = list_res[idx - 1]["sentence"]
            first_key = line.find('Q：', 1)
            if first_key > 0 and len(line[first_key:]) > 50:
                list_res[idx-1]["sentence"] = list_res[idx-1]["sentence"] + line[:first_key]
                list_res[idx]["sentence"] = line[first_key:]
            elif first_key == -1 and prev_sentence.rfind('Q：') != -1:
                key = list_res[idx-1]["sentence"].rfind('Q：', 1)
                if len(list_res[idx-1]["sentence"][key:]) <= 50:
                    list_res[idx]["sentence"] = list_res[idx-1]["sentence"][key:] + list_res[idx]["sentence"]
                    list_res[idx-1]["sentence"] = list_res[idx-1]["sentence"][:key]
    list_res = [
        {**item, "sentence": item["sentence"].replace(' ', '')}
        for item in list_res
        if len(item["sentence"].replace(' ', '')) > 0
    ]
    merged_res = merge_qa_array(list_res)
    return merged_res


def topic_segment_text(content, segment_model, max_length_limit, min_length):
    # 利用算法进行大文本拆分
    input_text = content.replace('\n', '').replace(' ', '').replace(':', "：")
    for i in range(0, 100):
        input_text = input_text.replace(f'Q{i}：', 'Q：')  # 此处后续需要优化
    try:
        seg_text = maximal_clique_seg_text(input_text, segment_model)
    except Exception as e:
        seg_text = cut_sentences(input_text)
    # 根据特定pattern再进行一次拆解
    pattern_res = pattern_segment(seg_text)
    # 在之前的基础上，对于长度较大的连续文本片段，使用max_token_limit进行限制，对长片段再进行拆分，以保证特定的长度。
    final_seg = _check_length_limit_and_re_seg(pattern_res, max_length_limit, min_length)
    # 先按照长短进行合并
    combined_res = [''.join(line_item) for line_item in final_seg]
    # 对于超过长度文本再进行截断
    split_res = split_text_trunk(combined_res, max_length_limit, min_length)
    adjusted_res = adjust_qa_format(split_res)
    return adjusted_res


def topic_segment_array(content_array, segment_model, max_length_limit, min_length):
    # 利用算法进行大文本拆分
    for item in content_array:
        text = item["sentence"].replace('\n', '').replace(' ', '').replace(':', '：')
        text = reduce(lambda t, i: t.replace(f'Q{i}：', 'Q：'), range(100), text)
        item["sentence"] = text
    try:
        seg_text = maximal_clique_seg_array(content_array, segment_model)
    except Exception as e:
        seg_text = content_array
    # 根据特定pattern再进行一次拆解
    pattern_res = pattern_segment_array(seg_text)
    # 在之前的基础上，对于长度较大的连续文本片段，使用max_token_limit进行限制，对长片段再进行拆分，以保证特定的长度。
    final_seg = _check_length_limit_and_re_seg_array(pattern_res, max_length_limit, min_length)

    # 对于超过长度文本再进行截断
    split_res = split_text_trunk_array(final_seg, max_length_limit, min_length)
    adjusted_res = adjust_qa_format_array(split_res)
    return adjusted_res

