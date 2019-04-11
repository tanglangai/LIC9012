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

def match_(obj: str, sub: str, spo: Dict[str, str], label_scheme_dict: Dict[int, dict]):
    """
    判断这对词，是不是对应spo的object，subject，是的话，再找是label_scheme_dict的哪一个
    返回0~49之间的值，
    不是，返回0
    """
    # 注意，用简单的in来判断是不是，容易犯 哥伦比亚 将 单独"比"这个字算入entity 的错误。
    if obj in spo['object'] and sub in spo['subject']:
        for key, scheme in label_scheme_dict.items():
            if is_match_scheme(spo, scheme):
                return key
        
        raise Exception("没有找到对应的label_scheme,这不正常。")
    else:
        return None
    

def scheme2index(pth: str) -> Dict[int, dict]:
    """
    将scheme对应成1-50!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    把0 空出来！！！！！！！！
    返回五十种视图的字典。
    """
    index_dict = {}
    i = 1
    with open(pth, 'r') as file:
        for line in file.readlines():
            if not line.strip() or line.strip() == 'EOF':
                continue
            
            index_dict[i] = json.loads(line)
            
            i += 1
    assert len(index_dict) == 50
    return index_dict


def convert_spolist2tensor(words: List[str], label_scheme_dict: Dict[int, dict],
                           spo_list: List[Dict[str, str]]) -> torch.Tensor:
    """
    将spo_list转成n*n*51，其中n是句子长度。
    """
    n = len(words)
    temp_tensor = torch.zeros((n, n, 51))

    for i, obj in enumerate(words):
        for j, sub in enumerate(words):
            
            for spo in spo_list:
                
                assert isinstance(spo, dict)
    
                height = match_(obj, sub, spo, label_scheme_dict)
                if height is not None:

                    temp_tensor[i, j, height] = 1
                # 之前的一个重大的bug！！！！！！！！！！！！！！
                # 我们至少要有一个1 ，如果没有匹配的话，就将第 0 位置置为 1
            if temp_tensor[i, j, :].sum().item() == 0:
                temp_tensor[i,j,0] = 1
                    
    # return temp_tensor.cpu().numpy().tolist()
    #之前将数据放入了cpu中
    #CPU和GPU之间互换数据是一件十分浪费时间的事情。
    #尝试直接返回tensor的形式。
    
    #{"postag": [{"word": "丝角蝗科", "pos": "nz"}, {"word": "，", "pos": "w"}, {"word": "Oedipodidae", "pos": "nz"}, {"word": "，", "pos": "w"},
    # {"word": "昆虫纲直翅目蝗总科", "pos": "nz"}, {"word": "的", "pos": "u"}, {"word": "一个科", "pos": "n"}],
    # "text": "丝角蝗科，Oedipodidae，昆虫纲直翅目蝗总科的一个科",
    # "spo_list": [{"predicate": "目", "object_type": "目", "subject_type": "生物", "object": "直翅目", "subject": "丝角蝗科"}]}
    
    "会有问题，如上那一条，因为错误分词的结果"
    # if temp_tensor[:,:,0].sum() == n*n:
    #     print("ALERT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #     print("起码有一个标注了的吧，我之前是因为预处理在这个函数之前调用了，导致“周星驰”不会出现在“NAME”之中")
        
        
    #按理说，每一对词，51列中有且只能有一个1，因此这条语句永远是真的
    #若不通过，说明一对词被标注成了多个scheme，检查！
    
    
    
    if  temp_tensor.sum().item() != n * n:
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
                    assert False
        print("temp_tensor.sum() is {}",format(temp_tensor.sum()))
        print("n * n is {}".format(n * n))
        raise Exception("一对词被标注成了多个scheme？")
    return temp_tensor
    

    
if __name__ == '__main__':
    #能成功打印
    for line in (generaterows('../data/processed_data/faketrain.json')):
        print(line)