import difflib

def diff_highlight(original: str, corrected: str) -> str:
    """
    按字符比较original和corrected，只把修改的字符标红，
    其余字符原样保留。结果用HTML span标红。
    """
    matcher = difflib.SequenceMatcher(None, original, corrected)
    result = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # 相同部分直接添加原文
            result.append(original[i1:i2])
        elif tag in ('replace', 'insert'):
            # 新文本用红色标出
            result.append(f"<span style='color:red'>{corrected[j1:j2]}</span>")
        elif tag == 'delete':
            # 删除的文本不显示
            pass
    return "".join(result)
