"""
@author 梁家熙
一些函数
关于试图映射
与label的生成。
如果看不懂，请找我，我会详细注释的。
"""

from typing import Dict, List
import json
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from dataUtils.make_label_from_raw import wsearch

def generaterows(file_pth: str) ->dict:
    """
    简单读取文件，抛出每一行
    使用如下所示
        for line in (generaterows('../data/processed_data/faketrain.json')):
            print(line)
    """
    with open(file_pth) as file:
        while True:
            line = file.readline()
            if not line:
                break
            yield line


def postag2wordpos(postag: List[Dict[str, str]])->[List, List]:
    """   传入
        [{"word": "如何", "pos": "r"},
        {"word": "演", "pos": "v"},
        {"word": "好", "pos": "a"}]字典，
        转换成两个单独的列表
    """
    words = []
    poses = []
    for dct in postag:
        words.append(dct['word'])
        poses.append(dct['pos'])
    return words, poses


def is_match_scheme(scheme_a: Dict[str, str], scheme_b: Dict[str, str])->bool:
    "判断两个scheme是否匹配，为  match_  函数服务的子模块"
    if scheme_a['object_type'] == scheme_b['object_type'] and \
    scheme_a['subject_type'] == scheme_b['subject_type'] and \
    scheme_a['predicate'] == scheme_b['predicate']:
        return True
    return False

# def match_(words: List[str], spo: Dict[str,str], label_scheme_dict: Dict[str, dict]):
#     """
#     判断spo中的客体主体这对词，是不是对应spo的object，subject，是的话，再找是label_scheme_dict的哪一个
#     返回 1~50 之间的值，
#     不是，返回0
#     """
#     # spo 是一条答案
#     obj = spo['object']
#     sub = spo['subject']
#     predicate = spo['predicate']
#
#     # 如果词列表中有词包含了obj 且 sub，那么再去寻找是哪一个视图
#     # 注意，用简单的in来判断是不是，容易犯 哥伦比亚 将 单独"比"这个字算入entity 的错误。
#     i = -1
#     j = -1
#     for index, w in enumerate(words):
#
#     if any([w in obj for w in words]) and any([w in sub for w in words]):
#         #肯定能找到一个视图的，找不到
#         if predicate not in label_scheme_dict.keys():
#             raise Exception("答案应该能够找到对应的标签")
#         height = label_scheme_dict[predicate]['position']
#     else:
#         height = 0
#
        
        
        
    

def scheme2index(pth: str) -> Dict[str, int]:
    """
    将scheme对应成1-50!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    把0 空出来！！！！！！！！
    返回五十种视图的字典。
    修改一下，key 是str形式的字典，而value 则是第几个值。
    """
    index_dict = {}
    i = 1
    with open(pth, 'r') as file:
        for line in file.readlines():
            if not line.strip() or line.strip() == 'EOF':
                continue
            data = json.loads(line)
            # index_dict[i] = json.loads(line)
            predicate = data['predicate']
            obj_type = data['object_type']
            sub_type = data['subject_type']
            index_dict[predicate] = {'position': i,
                                     'object_type': obj_type,
                                     'subject_type': sub_type}
            i += 1
            
    assert len(index_dict) == 50
    return index_dict


def convert_spolist2tensor(words: List[str], label_scheme_dict: Dict[str, dict],
                           spo_list: List[Dict[str, str]]) -> torch.Tensor:
    """
    将spo_list转成n*n*51，其中n是句子长度。
    """
    n = len(words)
    temp_tensor = torch.zeros((n, n, 51))

    for spo in spo_list:
        assert isinstance(spo, dict)
        
        obj_locate = wsearch(words, spo['object'])
        obj_loc = []
        for i, bool_ in enumerate(obj_locate):
            if bool_:
                obj_loc.append(i)

        sbj_locate = wsearch(words, spo['subject'])
        sbj_loc = []
        for i, bool_ in enumerate(sbj_locate):
            if bool_:
                sbj_loc.append(i)
        
        

    #按理说，每一对词，51列中有且只能有一个1，因此这条语句永远是真的
    #若不通过，说明一对词被标注成了多个scheme，检查！

    if temp_tensor.sum().item() != n * n:
        m = temp_tensor.shape[0]
        for i in range(m):
            for j in range(m):
                a = temp_tensor[i, j]
                if a.sum().item() >1:
                    print(i,j)
                    print(words[i])
                    print(words[j])
                    print(words)
                    print(a)
                    
                    i = -1
                    for d in a:
                        i+=1
                        if d  == 1:
                            print(label_scheme_dict[i])
                    print(spo_list)
                    print()
                    
                    # print(label_scheme_dict)
                    assert False
                    
        print("temp_tensor.sum() is {}",format(temp_tensor.sum()))
        print("n * n is {}".format(n * n))
        raise Exception("一对词被标注成了多个scheme？")
    return temp_tensor
    

    
if __name__ == '__main__':
    #能成功打印
    for line in (generaterows('../data/processed_data/faketrain.json')):
        print(line)