from typing import List
from pydantic import BaseModel
from fastapi import FastAPI
from topic_method.topic_segmentation import topic_segment_text, topic_segment_array
from token_length_method.length_segmentation import length_segment_text, length_segment_array
from pdf_parser.pdf_parsing import get_clean_pdf_file
from pdf_parser.locate_page import locate_page
from text2vec import SentenceModel
import uvicorn
import os

# Initialize API Server
app = FastAPI(
    title="Text Segmentation",
    description="Text Segmentation",
)


class PdfInputValue(BaseModel):
    pdf_url: str


class PdfTextOutput(BaseModel):
    pdf_text: str


class InputValue(BaseModel):
    context_info: str
    input_text: str
    seg_method: int
    trunk_max_len: int
    trunk_min_len: int
    sentence_max_len: int
    sentence_min_len: int


class OutputList(BaseModel):
    result: List[tuple]


class ReportInputValue(BaseModel):
    pdf_url: str
    context_info: str
    threshold: int
    trunk_max_len: int
    trunk_min_len: int
    sentence_max_len: int
    sentence_min_len: int


class InputArrayValue(BaseModel):
    content_items: List[dict]
    context_info: str
    threshold: int
    trunk_max_len: int
    trunk_min_len: int
    sentence_max_len: int
    sentence_min_len: int


class ReportOutputList(BaseModel):
    result: List[tuple]


class ReportArrayOutputList(BaseModel):
    result: List[dict]


@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add local model
    """
    global segment_model
    working_dir = os.path.dirname(os.path.abspath(__file__))
    embedding_model_path = os.path.join(working_dir, 'segment_model/model')
    segment_model = SentenceModel(embedding_model_path, max_seq_length=510)


@app.post("/parse_pdf", response_model=PdfTextOutput)
def parse_pdf(pdf_input_value: PdfInputValue):
    pdf_url = pdf_input_value.pdf_url
    text, page = get_clean_pdf_file(pdf_url)
    return PdfTextOutput(pdf_text=text)


@app.post("/segment_text", response_model=OutputList)
def split_text_2_chunk_sentence_pdf(input_value: InputValue):
    """
    context_info: str  上下文信息
    input_text:str   输入的文本
    seg_method:int  token_Length/topic, 1为token_length方法，2为topic拆分
    trunk_max_len: int trunk的最大长度
    trunk_min_len: int trunk的最小长度
    sentence_max_len: int sentence的最大长度
    """

    context_info = input_value.context_info
    input_text = input_value.input_text
    seg_method = input_value.seg_method
    trunk_max_len = input_value.trunk_max_len # 默认为450
    trunk_min_len = input_value.trunk_min_len  # 默认为200
    sentence_max_len = input_value.sentence_max_len  # 默认为200
    sentence_min_len = input_value.sentence_min_len  # 默认为80

    # 切分trunk片段
    if seg_method == 1:
        trunk_segments = length_segment_text(input_text, trunk_max_len, trunk_min_len)
    else:
        trunk_segments = topic_segment_text(input_text, segment_model, trunk_max_len, trunk_min_len)

    if len(trunk_segments) == 0 or trunk_segments == ['']:
        return OutputList(result=[])

    # 切分sentence片段
    return_res = list()
    for trunk_idx, trunk_item in enumerate(trunk_segments):
        sentence_segments = length_segment_text(
            trunk_item, sentence_max_len, sentence_min_len)
        for sentence_idx, sentence_item in enumerate(sentence_segments):
            return_res.append((trunk_idx, sentence_idx, context_info+sentence_item, context_info+trunk_item))

    return OutputList(result=return_res)


@app.post("/split_research_report", response_model=ReportOutputList)
def split_research_report(input_value: ReportInputValue):
    """
    context_info: str  上下文信息
    input_text:str   输入的文本
    seg_method:int  token_Length/topic, 1为token_length方法，2为topic拆分
    trunk_max_len: int trunk的最大长度
    trunk_min_len: int trunk的最小长度
    sentence_max_len: int sentence的最大长度
    """
    pdf_url = input_value.pdf_url
    context_info = input_value.context_info
    threshold = input_value.threshold
    trunk_max_len = input_value.trunk_max_len # 默认为450
    trunk_min_len = input_value.trunk_min_len  # 默认为200
    sentence_max_len = input_value.sentence_max_len  # 默认为200
    sentence_min_len = input_value.sentence_min_len  # 默认为80
    # pdf_url = pdf_input_value.pdf_url
    clean_text, pdf_parse_page = get_clean_pdf_file(pdf_url)
    if len(clean_text) == 0:
        return ReportOutputList(result=[])
    if len(clean_text) > threshold:
        seg_method = 1
    else:
        seg_method = 2
    print(f"采用第{seg_method}种分割方法")
    # 切分trunk片段
    if seg_method == 1:
        trunk_segments = length_segment_text(clean_text, trunk_max_len, trunk_min_len)
    else:
        trunk_segments = topic_segment_text(clean_text, segment_model, trunk_max_len, trunk_min_len)
    if len(trunk_segments) == 0 or trunk_segments == ['']:
        return ReportOutputList(result=[])
    # 切分sentence片段
    return_res = list()
    for trunk_idx, trunk_item in enumerate(trunk_segments):
        from collections import Counter
        # 假设 page_three, page_four, page_five, page_six 已经定义并且有值
        sentence_segments = length_segment_text(
            trunk_item, sentence_max_len, sentence_min_len)
        for sentence_idx, sentence_item in enumerate(sentence_segments):
            pages = [locate_page(sentence_item, pdf_parse_page, N) for N in range(3, 7)]
            # 直接找出最常见的页面
            most_common_page, _ = Counter(pages).most_common(1)[0]
            return_res.append((trunk_idx, sentence_idx, context_info+sentence_item, context_info+trunk_item, most_common_page + 1))
    return ReportOutputList(result=return_res)

@app.post("/split_report_array", response_model=ReportArrayOutputList)
def split_report_array(input_value: InputArrayValue):
    """
    context_info: str  上下文信息
    input_text:str   输入的文本
    seg_method:int  token_Length/topic, 1为token_length方法，2为topic拆分
    trunk_max_len: int trunk的最大长度
    trunk_min_len: int trunk的最小长度
    sentence_max_len: int sentence的最大长度
    """
    content_items = input_value.content_items
    context_info = input_value.context_info
    threshold = input_value.threshold
    trunk_max_len = input_value.trunk_max_len # 默认为450
    trunk_min_len = input_value.trunk_min_len  # 默认为200
    sentence_max_len = input_value.sentence_max_len  # 默认为200
    sentence_min_len = input_value.sentence_min_len  # 默认为80
    # pdf_url = pdf_input_value.pdf_url

    total_chars = sum(len(s["sentence"]) for s in content_items)

    if total_chars == 0:
        return ReportOutputList(result=[])
    if total_chars > threshold:
        seg_method = 1
    else:
        seg_method = 2
    print(f"采用第{seg_method}种分割方法")
    # 切分trunk片段
    if seg_method == 1:
        trunk_segments = length_segment_array(content_items, trunk_max_len, trunk_min_len)
    else:
        trunk_segments = topic_segment_array(content_items, segment_model, trunk_max_len, trunk_min_len)
    if len(trunk_segments) == 0 or trunk_segments[0]["sentence"] == '':
        return ReportOutputList(result=[])
    # 切分sentence片段
    return_res: List[dict] = []
    print("[PaiPai]trunk_segments:", trunk_segments)
    if not trunk_segments or len(trunk_segments) == 0:
        return ReportOutputList(result=[])
    for trunk_idx, trunk_item in enumerate(trunk_segments):
        # trunk_item 是 dict -> 再次基于长度拆分
        sentence_segments = length_segment_array(
            trunk_item, sentence_max_len, sentence_min_len)
        for sent_idx, sent_dict in enumerate(sentence_segments):
            merged = {
                "sentence_v": context_info + sent_dict["sentence"],
                "chunk": context_info + trunk_item["sentence"],
                "sentence_chunk_index": sent_idx,
                "chunk_index": trunk_idx,
                "bbox": sent_dict["bbox"],
            }
            return_res.append(merged)

    return ReportArrayOutputList(result=return_res)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8169)
