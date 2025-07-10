import re


def load_file(file_dir):
    """
    加载文件，剔除换行符及空格
    :param file_dir: 文件地址
    :return: 读取到的文本列表
    """

    text = list()
    with open(file_dir, 'r') as rf:

        for line in rf.readlines():
            if len(line.replace('\n', '').replace(' ', '')) == 0:
                continue
            text.append(line.replace('\n', '').replace(' ', ''))

    return text


def extract_speaker_and_content(text_segments):
    """
    本函数用于从对话序列中抽取讲话人身份，作为一个数列。同时抽取剔除讲话人身份后的文本作为一个单独的数列。
    :param text_segments:获取到的文本片段
    :return:
    """

    # 定义两个空列表，用于存储讲话人身份和讲话内容
    speakers = []
    contents = []
    # 定义一个正则表达式，匹配文本开头的“讲话人”和后面的数字或汉字，以及后面的冒号和讲话内容
    pattern = re.compile(r'^(讲话人([1-9一二三四五六七八九]))[:：](.*)')
    # 遍历文本列表
    for text_line in text_segments:
        # 使用正则表达式在文本中搜索匹配
        match = pattern.search(text_line)
        # 如果找到匹配
        if match:
            # 提取匹配的第一个分组，即完整的讲话人身份
            speaker = match.group(1)
            # 提取匹配的第三个分组，即讲话内容
            content = match.group(3)
            # 对于满足长度的文本，加入到列表中
            speakers.append(speaker)
            contents.append(content)
        # 如果没有找到匹配
        else:
            # 将None添加到两个列表中，表示无法识别讲话人身份和讲话内容
            speakers.append(None)
            contents.append(text_line)
    # 返回讲话人身份的列表和讲话内容的列表
    return speakers, contents

