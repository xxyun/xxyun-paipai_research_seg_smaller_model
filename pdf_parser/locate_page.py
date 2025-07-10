import jieba
import regex


def _ngrams(input_str, n, tokenize_flag=False):
    """
     为给定的字符串生成 n-grams（N元组）。
     参数：
     input_str: 输入字符串。
     n: grams的数量（N的大小）。
     tokenize_flag: 是否采用分词方法
     返回：
     set: 包含 n-grams 的集合。
     """
    if tokenize_flag:
        input_str = list(jieba.cut(input_str))
        return set([''.join(input_str[i:i + n]) for i in range(len(input_str) - n + 1)])
    else:
        return set([input_str[i:i + n] for i in range(len(input_str) - n + 1)])


def _calculate_overlap_ratio(source, target, N):
    """
    使用N-gram方法计算source和target之间的重叠比率。
    参数：
    source: 源字符串。
    target: 目标字符串。
    overlap_threshold: 重叠比率的阈值。
    N: grams的数量。
    返回：
    bool: 如果overlap_ratio >= overlap_threshold，则返回True，否则返回False。
    """
    # 这里需要按照我们讨论的过滤，保留中文、英文、数字。
    source = source.replace(" ", '').replace("|", "").replace("｜", "").replace("-", "")
    target = target.replace(" ", '').replace("|", "").replace("｜", "").replace("-", "")
    source_ngrams = _ngrams(source, N, False)
    target_ngrams = _ngrams(target, N, False)
    overlap_num = len(source_ngrams.intersection(target_ngrams))
    overlap_ratio = overlap_num / len(source_ngrams)  # 更正此处
    return overlap_ratio


def locate_page(chunk: str, pdf_parse_page: list[tuple[int, str]], N):
    """
    定位给定文本块在PDF页面中的位置。
    参数：
    - chunk: 要定位的文本块。
    - pdf_parse_page: 一个元组列表，每个元组包含页码和该页的文本。
    - overlap_threshold: 用于比较的重叠比率阈值。
    - N: 使用的N-gram大小。
    返回：
    - tuple: 包含页码和文本块的元组。如果没有找到匹配的页码，则返回None。
    """
    max_val = -float('inf')
    max_page = 0
    try:
        for page_num, page_text in pdf_parse_page:
            ratio = _calculate_overlap_ratio(chunk, page_text, N)
            if ratio >= max_val:
                max_page = page_num
                max_val = ratio
        return max_page
    except Exception as e:
        # 可以添加更具体的异常处理
        print(f"Error in locating page: {e}")
    return -1
