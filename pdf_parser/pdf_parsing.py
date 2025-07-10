import fitz
import re
import os
import requests
import collections
import numpy as np


def is_directory_page(page_text):
    """
    根据自定义规则判断是否为目录页,这里可以根据实际情况进行定制,
    示例规则：检查文本中是否包含目录特定的关键词或特征
    :param page_text:
    :return:
    """
    page_text = page_text.replace(' ', '')
    keywords = ["目录", "........."]
    flag_one = all(keyword in page_text for keyword in keywords)
    count_num = max(page_text.count('..............'),
                    page_text.count("………………………………"),
                    page_text.count("··············"))
    flag_two = False
    if count_num >= 1:
        flag_two = True
    # 两者满足一个即可
    return flag_one or flag_two


def pdf_to_txt(url):
    """
    读取pdf文件
    """
    response = requests.get(url)
    doc = fitz.open(stream=response.content, filetype="pdf")
    res = list()
    p = list()

    for i in range(doc.page_count):
        page = doc[i]
        text = page.get_text("")
        if is_directory_page(text):
            continue
        if i == 0:
            # 此处过滤掉相关报告、其他报告等呢绒
            text = text.replace(' ', '')
            if len(text) == 0:
                continue

            split_text = text.split('\n')
            split_text = [item for item in split_text if len(item.replace('\n', '').replace(' ', '')) > 0]
            keywords = [
                '近期研究',
                '相关研究',
                '相关报告',
                '相关研究报告'
            ]
            clean_res = list()
            for idx, line in enumerate(split_text):
                left_text = ''.join(split_text[idx:])
                if any([kwd in line for kwd in keywords]) and len(left_text) <= 500:
                    break
                else:
                    clean_res.append(line)

            temp_str = '\n'.join(clean_res)
            p.append((i, text))
            res.append(temp_str)
        elif len(text.replace('\n', '').replace(' ', '')) > 0:
            temp_str = text.replace(' ', '')
            p.append((i, temp_str))
            res.append(temp_str)

    line_res = list()
    for line in res:
        line_res.extend(line.split('\n'))
    line_res = [item for item in line_res if len(item.replace(' ', '')) > 0
                and '[table' not in item.lower()]

    combined_text = list()
    for item in line_res:
        item = item.replace('\t', "")
        if not item:
            continue
        combined_text.append(item.replace("\n", ""))

    return combined_text, p


def calculate_alpha_ratio(sentence):
    """
    检测文本中的中英文字符占比
    """
    # 匹配非中英文符号的正则表达式
    pattern = r"[^\u4e00-\u9fa5a-zA-Z]"
    # 使用re.findall()函数找出所有符号
    symbols = re.findall(pattern, sentence)
    # 计算符号的个数
    symbol_count = len(symbols)
    # 计算总字符数
    total_count = len(sentence)
    percentage = round(symbol_count / total_count * 100, 2)

    return 100 - percentage


def graph_check(text):
    """
    检测是否和图片以及数据来源信息有关
    """

    import re
    pattern_one = r'^(注[：:]|注释[：:]|图[0-9一二三四五六七八九十]*[：:.]|表[0-9一二三四五六七八九十]*[：:.]|数据来源[：:.]|资料来源[：:.]|信息来源[：:.]).+'
    pattern_two = r'^(图表[0-9一二三四五六七八九十]*[：:.]|附录[0-9一二三四五六七八九十]*[：:.]).+'
    match_one = re.search(pattern_one, text)
    match_two = re.search(pattern_two, text)

    cut_res = cut_sent(text)

    for line_item in cut_res:
        if '分析师' in line_item and '简介' in line_item:
            return True
        elif '团队' in line_item and '简介' in line_item:
            return True
        elif '评级' in line_item and '说明' in line_item:
            return True
        elif '重要声明' in line_item:
            return True

    if match_one or match_two:
        return True
    else:
        return False


def remove_duplicate_text(combined_text):
    """
    去除高频出现的文字串
    """
    count = collections.Counter(combined_text)
    most_common_list = [item for item, freq in count.items() if freq >= 4]
    filter_text = [line for line in combined_text
                   if line not in most_common_list]

    return filter_text


def remove_disclaimer(text):
    """
    免责条款/公司其他声明
    """
    if not text:
        return ""
    pattern = r'(风险提示与免责条款|行业评级与免责声明|法律主体声明|重要免责声明|与公司有关的信息披露' \
              r'证券投资咨询业务的说明|本公司具备证券投资咨询业务资格的说明|公司业务资格说明|分发和地区通知|免责及评级说明部分|' \
              r'投资评级说明及重要声明|华西证券免责声明|研究所分析师列表|宏观策略研究团队).*?$'
    result = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    keywords = [
        '免责声明\n',
        '免责声明'
        '免责申明\n',
        '免责申明'
        '披露声明\n',
        '披露声明'
        '资质声明\n',
        '资质声明',
        '评级说明\n',
        '评级说明'
        '分析师承诺\n',
        '分析师承诺',
        '特别声明\n',
        '特别声明',
        '分析师声明\n',
        '分析师声明',
        '分析师申明\n',
        '分析师申明',
        '分析师简介\n',
        '分析师简介',
        '分析师认证',
        '背景经历',
        'MSCI ESG评级免责声明条款\n',
        '分析师承诺及风险提示\n',
        '分析师承诺及风险提示'
        '分析师简介及承诺\n',
        '分析师简介及承诺',
        '分析师与研究助理简介\n',
        '分析师与研究助理简介',
        "研究团队简介\n",
        "研究团队简介",
        "附录"
    ]

    for keyword in keywords:
        index = result.rfind(keyword)
        if index != -1:
            result = result[:index]

    return result.strip()


def remove_page_footer(text):
    """
    去掉页脚
    """
    if not text:
        return ""
    pattern = r'请务必阅读正文最后的.*?公司免责声明'
    result = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    if not result:
        return text
    keywords = [
        '请阅读最后一页重要免责声明',
        '请务必参阅正文后面的信息披露和法律声明',
        '敬请参阅末页重要声明及评级说明',
        '敬请阅读本报告正文后各项声明',
        '请务必仔细阅读正文后的所有说明和声明',
        '请务必阅读最后一页股票评级说明和免责声明',
        '请阅读最后一页各项声明',
        '信息披露和法律声明',
        '请务必参阅正文后面的信息披露和法律声明',
        '请务必阅读正文之后的信息披露和免责申明',
        '敬请参阅最后一页特别声明',
        '请务必参阅最后一页的法律申明',
        '请务必阅读报告末页',
        '请务必阅读末页的重要说明',
        '请仔细阅读本报告末页声明',
        '本报告的信息均来自已公开信息，关于信息的准确性与完整性，建议投资者谨慎判断，据此入市，风险自担',
        '有关分析师的申明，见本报告最后部分。其他重要信息披露见分析师申明之后部分，或请与您的投资代表联系。并请阅读本证券研究报告最后一页的免责申明',
        '本公司具备证券投资咨询业务资格，请务必阅读最后一页免责声明',
        '此报告仅供内部客户参考',
        '请通过合法途径获取本公司研究报告，如经由未经许可的渠道获得研究报告，请慎重使用并注意阅读研究报告尾页的声明内容',
        '敬请阅读本报告正文后各项声明',
        '文章纯属个人观点,仅供参考,文责自负。读者据此入市,风险自担',
        '请务必阅读正文之后的免责条款部分',
        '请仔细阅读在本报告尾部的重要法律声明',
        '请阅读最后一页免责声明及信息披露',
        '请务必阅读最后特别声明与免责条款',
        '请务必阅读正文之后的免责声明部分',
        '请务必阅读正文之后的',
        '请务必阅读尾页重要声明',
        '敬请参阅尾页之免责声明',
    ]
    for keyword in keywords:
        result = result.replace(keyword, '')
    return result.strip()


def remove_contact_info(text):
    """
    去除联系信息
    """
    if not text:
        return ""
        # 正则表达式模式，匹配要删除的内容
    patterns = [
        r'姓名：\S+',
        r'作者：\S+',
        r'研究员：\S+',
        r'分析师：\S+',
        r'证券分析师：\S+',
        r'首席分析师：\S+',
        r'\(\d+\)\d+',
        r'\S+@\S+',
        r'证券投资咨询业务证书编号：\s*\S+',
        r'证书编号：\s*\S+',
        r'投资咨询资格编号：\s*\S+',
        r'资格证书：\s*\S+',
        r'执业编号：\s*\S+',
        r'执业证号：\s*\S+',
        r'执业登记编码：\s*\S+',
        r'登记编码：\s*\S+',
        r'登记编码：\s*\S+',
        r'SAC编号：\s*\S+',
        r'执业证书：\s*\S+',
        r'执业证书号：\s*\S+',
        r'SAC执业证书：\s*\S+',
        r'SAC执业编号：\s*\S+',
        r'SAC编号：\s*\S+',
        r'SAC执业证书编号：\s*\S+',
        r'执业证书编码：\s*\S+',
        r'执业证书编号：\s*\S+',
        r'分析师登记编码：\s*\S+',
        r'分析师：\S+',
        r'邮箱：\S+',
        r'邮箱地址：\S+',
        r'Email：\S+',
        r'E-MAIL：\S+',
        r'联系邮箱：\S+',
        r'SAC NO：\S+',
        r'研究助理：\S+',
        r'联系方式：\S+',
        r'联系方式：\S+',
        r'地址：\S+',
        r'联系人\S+',
        r'联系电话\S+',
        r'电话：\s*\d{3}-\d{8}',
    ]
    # 依次匹配并删除指定内容
    for pattern in patterns:
        text = re.sub(pattern, '', text)

    return text


def clean_report(text):
    """
    对于文档内容进行清洗
    """
    if not text:
        return ""
    text = '\n'.join(text)
    text = remove_disclaimer(text)
    text = remove_page_footer(text)
    text = remove_contact_info(text)
    return text


def detect_heading_line(text):
    """
    根据特征，检测是否为标题
    """
    if len(text) >= 30:
        return False

    pattern_title_1 = re.compile(r'^[1-9一二三四五六七八九十][.、]+')
    pattern_title_2 = re.compile(r'^[1-9一二三四五六七八九十][.、]+[1-9一二三四五六七八九十][.、]*')
    pattern_title_3 = re.compile(r"^[（(]*[0-9一二三四五六七八九十]+[)）]+[.、]*")
    # 检查段落是否符合“数字.数字. ”格式
    heading_flag = False
    if pattern_title_1.search(text):
        heading_flag = True

    # 检查段落是否符合“数字.数字. ”格式
    elif pattern_title_2.search(text):
        heading_flag = True
    elif pattern_title_3.search(text):
        heading_flag = True

    if heading_flag and calculate_alpha_ratio(text) <= 40:
        heading_flag = False

    return heading_flag


def fill_gaps(doc_length, num_array):
    """
    定义一个函数，接受文档全长和数值数组作为参数, 补全填充获取起始位置的列表
    """
    if len(num_array) == 0:
        return []
    if num_array[0] != 0:
        num_array.insert(0, 0)
    if num_array[-1] != doc_length:
        num_array.append(doc_length)
    idx = 0
    res = list()
    while idx < len(num_array) - 1:
        res.append([num_array[idx], num_array[idx + 1]])
        idx += 1
    return res


def load_text_with_seq(sort_res, line_res):
    """
    对于通过算法切割的片段，根据原文，转换成文本列表
    """
    final_text = list()

    for start, end in sort_res:

        segment_res = line_res[start:end]
        new_res = list()
        for item in segment_res:
            if detect_heading_line(item):
                new_res.append("#:" + item + '\n')
            else:
                new_res.append(item)
        segment_res = new_res
        mean_length = np.mean([len(item) for item in segment_res])

        if mean_length > 14 and ('。' in ''.join(segment_res) or '，' in ''.join(segment_res)):
                final_text.append(segment_res)
        elif (line_res[start].endswith('。') or line_res[start].endswith('！')) and (len(final_text) > 0 and not final_text[-1][-1].endswith('。')):
            final_text.append([line_res[start]])
        else:
            for item in segment_res:
                if detect_heading_line(item.replace("#:", '')):
                    final_text.append([item])

    return final_text


def remove_table_from_text(line_res):
    """
    从文本中剔除表格内容
    """
    len_data = np.array([len(item) for item in line_res])
    segment_idx = list()
    for idx, value in enumerate(len_data):

        if idx <= 4:
            continue

        avg_value = len_data[idx - 5:idx].mean()
        if value / avg_value >= 3 or avg_value / value >= 3 or detect_heading_line(line_res[idx]):
            segment_idx.append(idx)

    sort_res = fill_gaps(len(line_res), segment_idx)
    text_res = load_text_with_seq(sort_res, line_res)
    return text_res


def remove_all_symbol_line(check_res):
    """
    剔除掉全部是数字和符号的列
    """
    clean_res = list()

    for item in check_res:
        if calculate_alpha_ratio(item.replace(' ', '')) <= 5 and not item.endswith('。') and not item.endswith('！') and not item.endswith('；'):
            continue
        else:
            clean_res.append(item.replace('\n', ''))

    return clean_res


def cut_sent(para):
    """
    对文本进行分割，分句
    """
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    para_split = para.split("\n")
    final_split_res = list()
    for item in para_split:
        final_split_res.extend(item.split('#:'))

    return final_split_res


def get_clean_pdf_file(pdf_path):

    # 获取pdf文件
    try:
        text_content, page = pdf_to_txt(pdf_path)
    except Exception as e:
        print(f"error happened{e}")
        text_content = []
        page = []
    filter_text = remove_duplicate_text(text_content)
    final_txt = clean_report(filter_text)
    final_txt = final_txt.split('\n')

    line_res = list()
    for line in final_txt:
        line_res.extend(line.split('\n'))

    line_res = [item for item in line_res if len(item) > 0]
    final_text = remove_table_from_text(line_res)
    check_res = list()
    for item in final_text:
        check_res.extend(item)
    clean_res = remove_all_symbol_line(check_res)
    final_res = ' '.join(clean_res)
    cut_res = cut_sent(final_res)

    filter_res = list()

    for idx, item in enumerate(cut_res):
        test_item = item.replace(' ', '')
        if graph_check(test_item) or len(test_item) == 0:
            continue
        if idx == 0:
            filter_res.append(item)
        else:
            filter_res.append(test_item)

    return '\n'.join(filter_res), page


if __name__ == "__main__":

    working_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(working_dir,'../test_input/report_0808')
    output_path = os.path.join(working_dir,'../test_input/output')

    for item in os.listdir(input_path):
        pdf_val = os.path.join(input_path, item)
        value = item.split('.')[0]
        output_file_path = os.path.join(output_path, value+'.txt')
        values, page = get_clean_pdf_file(pdf_val)
        with open(output_file_path,'w') as wf:
            wf.write(values+'\n')
