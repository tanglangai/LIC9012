from typing import List, Dict
import json
from copy import deepcopy

from common_utils.preprocess_merge_something import NAME, TIME, BOOKNAME, NUMBER

def tojsonfile(file_pth: str,lst: List[Dict])->None:
    "传入一个列表，然后写到制定的目录中。"
    with open(file_pth, 'w') as file:
        for line in lst:
            line=json.dumps(line,ensure_ascii=False)
            line+='\n'
            file.write(line)
    print('end')
    
def convert_ans_to_original_word(ans: Dict, words: List[str], position_dict: [Dict]):
    "我们之前对人名等进行了替换，这里在替换回来"
    spo_lst = ans['spo_list']
    ret_spo_lst = []
    
    for tdict in spo_lst:

        obj = tdict['object']
        sub = tdict['subject']
        
        if obj in [NAME, TIME, BOOKNAME, NUMBER]:
            index = words.index(obj)
            real_word = position_dict[index]
            tdict['object'] = real_word
            
        if sub in [NAME, TIME, BOOKNAME, NUMBER]:
            index = words.index(sub)
            real_word = position_dict[index]
            tdict['subject'] = real_word
            
        ret_spo_lst.append(deepcopy(tdict))
    ans['spo_list'] = ret_spo_lst
    return deepcopy(ans)