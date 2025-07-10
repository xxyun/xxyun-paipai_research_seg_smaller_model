def merge_bboxes(boxes: list[list[int]]) -> list[int]:
    boxes = list(boxes)               # <--- 关键修复
    if not boxes:
        return [0, 0, 0, 0]
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [x1, y1, x2, y2]