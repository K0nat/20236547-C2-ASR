# code/text_normalize.py
import re
import string

# 英文数字词 -> 阿拉伯数字（TESS 单词级足够）
WORD_TO_DIGIT = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def remove_punctuation(text: str) -> str:
    return text.translate(_PUNCT_TABLE)


def words_to_digits(text: str) -> str:
    tokens = text.split()
    out = []
    for tok in tokens:
        out.append(WORD_TO_DIGIT.get(tok, tok))
    return " ".join(out)


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_tess_prefix(text: str) -> str:
    """TESS 参考里常见的 'say the word xxx' 前缀"""
    t = text.lower().strip()
    prefix = "say the word "
    if t.startswith(prefix):
        return t[len(prefix):].strip()
    return t


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower().strip()
    text = strip_tess_prefix(text)
    text = remove_punctuation(text)
    text = words_to_digits(text)
    text = collapse_spaces(text)
    return text