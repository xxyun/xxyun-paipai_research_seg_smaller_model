import re
from topic_method.topic_segmentation import pattern_segment, pattern_segment_array, split_text_trunk, adjust_qa_format
from util.bbox_util import merge_bboxes


def split_text_into_sentences(text):
    """
    将文本拆分为句子
    """
    sentences = re.split('([。？！!?])', text)
    sentences = [sentences[i] + (
        sentences[i + 1] if i + 1 < len(sentences) else '') for i in
                 range(0, len(sentences), 2)]
    return sentences


def greedy_merge_by_tokens(full_text, max_length_limit, min_length):
    """
    这个函数的作用是为了把一些过长或过短的文本片段进行合理的划分，
    使得每个划分后的文本都接近max_token_limit。这样可以提高后续处理的效率和质量。
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

    if len(results) == 0:
        results.append(full_text[current_idx:])

    elif len('\n'.join(full_text[current_idx:])) < min_length:
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


def simple_segment_text(full_text, max_length_limit, min_length):
    combine_res = '\n'.join(full_text)
    if len(combine_res) <= max_length_limit:
        return [full_text]
    return greedy_merge_by_tokens(full_text, max_length_limit, min_length)


def simple_segment_array(seg_array, max_length_limit, min_length):
    total_len = sum(len(s["sentence"]) for s in seg_array)
    if total_len <= max_length_limit:
        return [seg_array]
    return greedy_merge_by_tokens_for_array(seg_array, max_length_limit, min_length)  # List[List[dict]]


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


def length_segment_text(content, max_length_limit, min_length):
    """
    按长度进行截取，主入口
    """
    split_content = split_text_into_sentences(content)
    res_list = greedy_merge_by_tokens(split_content, max_length_limit, min_length)

    pattern_res = pattern_segment(res_list)
    # 在之前的基础上，对于长度较大的连续文本片段，使用max_token_limit进行限制，对长片段再进行拆分，以保证特定的长度。
    final_seg = _check_length_limit_and_re_seg(pattern_res, max_length_limit, min_length)
    concat_res = list()
    for item in final_seg:
        line_res = ' '.join(item).replace('\n', ' ')
        if len(line_res) < 1500: # 之后根据实验统计后的阈值进行判断
            concat_res.append(line_res)
    return concat_res


def length_segment_array(content_items, max_length_limit, min_length):
    """
    按长度进行截取，主入口
    """
    res_list = greedy_merge_by_tokens_for_array(content_items, max_length_limit, min_length)
    pattern_res = pattern_segment_array(res_list)
    # 在之前的基础上，对于长度较大的连续文本片段，使用max_token_limit进行限制，对长片段再进行拆分，以保证特定的长度。
    final_seg = _check_length_limit_and_re_seg_array(pattern_res, max_length_limit, min_length)
    concat_res = list()
    for items in final_seg:  # it 是单条 dict
        # 按照你的业务，这里其实什么也不用再合并，直接判断长度即可
        merged_items = {"sentence": ' '.join([item["sentence"] for item in items]),
                        "bbox": merge_bboxes([item["bbox"] for item in items])}
        if len(merged_items["sentence"]) < 1500:
            concat_res.append(merged_items)
    return concat_res