"""
梁家熙
这里是一些过滤的函数。
比如单独的一个标点符号不可能是目标
"""

from typing import List, Dict
from copy import deepcopy

def filter_only_one_character(spo_list: List[Dict])->List[Dict]:
    "如果主体和课题只有单独的一个字（此时很有可能是一个标点符号）时，去除"
    ret_lst = []
    for tdict in spo_list:
        obj = tdict['object']
        sub = tdict['subject']
        if len(obj) <= 1 or len(sub) <= 1:
            continue
        
        ret_lst.append(deepcopy(tdict))
    return ret_lst