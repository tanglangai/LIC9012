"""
梁家熙
数据之前的预处理函数，例如
连续的时间可以连在一起

我们最后是想用pipeline将这些函数连载一起
"""

from typing import List, Dict

# 这里的替换是否正确？
#放在这里当做是全局变量，方便修改与其他的引用。
#注意，predict_func 文件中引用了这里，故修改一定要同步修改

NAME = "人1名"
TIME = "时1间"
BOOKNAME = "书1书"
NUMBER = "数1字"

def merge_adjoint_postag(words: List[str], postags: List[str]) -> [List[str], List[str]]:
    "将相邻的postags合并，比如t,t都代表时间且相邻，应该合并。"
    ret_words = []
    ret_postags = []
    for word, tag in zip(words, postags):
        if not ret_postags:
            ret_words.append(word)
            ret_postags.append(tag)
        else:
            # 暂时只融合t
            if ret_postags[-1] == tag and tag in ['t']:
                ret_words[-1] = ret_words[-1] + word
            else:
                ret_words.append(word)
                ret_postags.append(tag)
    
    assert len(ret_postags) == len(ret_words)
    return ret_words, ret_postags


def merge_name_punctuation(words: List[str], postags: List[str]) -> [List[str], List[str]]:
    "爱德华  ·   彼得   这种应该合并为一个词。"
    ret_words = []
    ret_postags = []
    for word, tag in zip(words, postags):
        if not ret_postags:
            ret_words.append(word)
            ret_postags.append(tag)
        else:
            if word.strip() == '·' or ret_words[-1].endswith('·'):
                ret_words[-1] = ret_words[-1] + word
            else:
                ret_words.append(word)
                ret_postags.append(tag)
    
    assert len(ret_postags) == len(ret_words)
    return ret_words, ret_postags


def merge_quotation_marks(words: List[str], postags: List[str]) -> [List[str], List[str]]:
    """
    排除"",'',《》里面过多分词的情况,将太多分词的连在一起。
    """
    ret_words = []
    ret_postags = []
    starts = ["'", "‘", "“", '"', '《', '<']
    ends = ["'", "’", "”", '"', "》", ">"]
    flag = False
    for word, tag in zip(words, postags):
        if not ret_postags:
            ret_words.append(word)
            ret_postags.append(tag)
            if word in starts:
                flag = True
        else:
            if word in starts:
                flag = True
            elif word in ends:
                flag = False
            
            if flag and word not in starts and ret_words[-1] not in starts:
                ret_words[-1] = ret_words[-1] + word
            else:
                ret_words.append(word)
                ret_postags.append(tag)
    return ret_words, ret_postags


def replace_some_nourns(words: List[str], postags: List[str]) -> [List[str], List[str], Dict[int, str]]:
    "替换名词，时间和图书名"
    ret_words = []
    ret_postags = []
    
    position_dict = {}
    i = -1

    # flag 用来指示是否是 《》里的内容
    flag = False
    for word, tag in zip(words, postags):
        i += 1
        if tag == 'nr':
            ret_words.append(NAME)
            ret_postags.append(tag)
            position_dict[i] = word
            continue
        if tag == 'm':
            ret_words.append(NUMBER)
            ret_postags.append(tag)
            position_dict[i] = word
            continue
        if tag == 't':
            ret_words.append(TIME)
            ret_postags.append(tag)
            position_dict[i] = word
            continue
        if word == '《':
            ret_words.append(word)
            ret_postags.append(tag)
            flag = True
            continue
        if word == '》':
            ret_words.append(word)
            ret_postags.append(tag)
            flag = False
            continue
        if flag:
            ret_words.append(BOOKNAME)
            ret_postags.append(tag)
            position_dict[i] = word
            continue
        # 以上都不满足，则
        ret_words.append(word)
        ret_postags.append(tag)
    
    return ret_words, ret_postags, position_dict


