"""
最初那一版的预测文件、

"""
from typing import List,Dict,Any,Set
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance, Token
from  allennlp.predictors.predictor import Predictor
from allennlp.models import Model
from overrides import overrides
from copy import deepcopy
import torch
import json
import demjson
from members.liangjiaxi.mylibrary.Config import Config
from common_utils.scheme_mapping import scheme2index
from common_utils.scheme_mapping import postag2wordpos

@Predictor.register('Bert51baseline_predictor')
class ner51_predictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super(ner51_predictor, self).__init__(model,dataset_reader)
        self.index_scheme=scheme2index(Config.scheme_pth)
        
    @overrides
    def predict_json(self, inputs: JsonDict):
        postag: List = inputs['postag']
        
        #text是为了之后的返回
        text: str = inputs['text']
        
        words, poses = postag2wordpos(postag)

        words_poses_tokens = []
        reserved_pos = []
        for word, pos in zip(words, poses):
            if pos in Config.RESERVED_LSIT:
                tmp_id = 1
            else:
                tmp_id = 0
            reserved_pos.append(tmp_id)
            
            # 关键的一步，将tag标签赋予Token，而且要token_indexers变成 pos_tag
            tmp_token = Token(word,tag_ = pos)
            words_poses_tokens.append(tmp_token)
            
        instance = self._dataset_reader.text_to_instance(words_poses_tokens, reserved_pos, None)
        
        output = self.predict_instance(instance)
        attention_logits = output['attention_logits']
        attention_logits = torch.Tensor(attention_logits)
        
        # #注意这一步之后，维度变成了51 * n *n
        # attention_logits = attention_logits.permute(2, 0, 1)
        #
        # attention_label=torch.round(attention_logits)
        #
        # if not attention_label.sum().item():
        #     assert attention_logits.shape[0] == 51
        #     b = torch.argmax(attention_logits)
        #     geshu, chang, kuan = attention_logits.shape
        #     i, j, k = b.item() // (chang * kuan), (b.item() % (chang * kuan)) // kuan, (
        #                 b.item() % (chang * kuan)) % kuan
        #     attention_label[i,j,k] = 1
        
        assert  attention_logits.shape[-1] == 51
        #这里是n*n的矩阵,类似
        # tensor([[1, 3, 1],
        #        [3, 1, 1],
        #        [2, 3, 1]])
        attention_map = torch.argmax(attention_logits, dim=-1)
        rows = attention_logits.shape[0]
        cols = attention_logits.shape[1]

        spo_list = []
        for i in range(rows):
            for j in range(cols):
                #如果预测的是0，则不是50种视图的任意一种
                if attention_map[i][j] == 0:
                    continue
                else:
                    #找到对应的视图
                    key = attention_map[i][j].item()
                    scheme = self.index_scheme[key]
                    tmp_scheme = {}
                    tmp_scheme.update(scheme)
                    obj = words[i]
                    sub = words[j]
                    tmp_scheme['object'] = obj
                    tmp_scheme['subject'] = sub
                    spo_list.append(deepcopy(tmp_scheme))
        
        
        #我们应该对预测结果进行一些处理，比如爱德华*迈特等等应该连在一起。这是事后合并操作。
        #遍历所有的可能性。
        # spo_list=[]
        # for k in range(1, 51):
        #     temp_matrix: torch.Tensor = attention_label[k, :, :]
        #     scheme = self.index_scheme[k]
        #     tmp_scheme = {}
        #     tmp_scheme.update(scheme)
        #     t_lst = merge_list(words, temp_matrix, scheme)
        #     spo_list.extend(t_lst)
            
        #去重操作，这一步不知道可以不可以省略
        tmpset = set([str(w) for w in spo_list])
        spo_list=[demjson.decode(w) for w in tmpset]
        
        #过滤操作，将一些不可能的值过滤掉，如只有一个字符的预测值等等
        post_process = Config.post_pipeline
        spo_list = post_process(spo_list)
        
        return {
            "text": text,
            'spo_list': spo_list
        }
    
#
# def merge_list(words: List[str], temp_matrix: torch.Tensor, scheme):
#     "合并类似爱德华·某某·某某 这种连续存在的情况"
#     temp_list=temp_matrix.cpu().numpy().tolist()
#     assert temp_list
#     ret_list=[]
#
#     for i in range(len(temp_list)):
#         for j in range(len(temp_list[0])):
#             #如果这个位置是 1
#             if temp_list[i][j]:
#                 obj = words[i]
#                 sub = words[j]
#                 ret_list=match(ret_list, obj, sub, i, j, scheme, temp_list)
#
#     for dct in ret_list:
#         dct.pop('position')
#
#     return ret_list
#
# def match(ret_list: List,obj: str,sub: str,i: int,j: int, scheme: dict, temp_list:List[List[float]]):
#     """
#     因为它是一行一行进行扫描的，所以我们
#     """
#     #去掉空格对嘛？？
#     sub=sub.strip()
#     obj=obj.strip()
#
#     if not ret_list:
#         t_dict={}
#         t_dict.update(scheme)
#         t_dict['position'] = (i, j)
#         t_dict['object'] = obj
#         t_dict['subject'] = sub
#
#
#         ret_list.append(deepcopy(t_dict))
#     else:
#         temp_dict=deepcopy(ret_list)
#         for index,dct in enumerate(temp_dict):
#
#
#             #先看能合并的，剩下的都认为是无法合并，另起新行
#             #左边能否合并
#             if j-1 >=0 and temp_list[i][j-1] == 1 :
#                 #合并左边
#                 if dct['position'] == (i, j-1):
#                     dct = ret_list[index]
#                     dct['subject'] = "".join([dct['subject'], sub])
#                     dct['position'] = (i, j)
#                     break
#             #再看一下上边能否合并
#             elif i-1 >=0 and temp_list[i-1][j] == 1 :
#                 if dct['position'] == (i-1, j):
#                     dct = ret_list[index]
#                     dct['object'] = "".join([dct['object'], obj])
#                     dct['position'] = (i, j)
#                     break
#             else:
#                 t_dict = {}
#                 t_dict.update(scheme)
#                 t_dict['position'] = (i, j)
#                 t_dict['object'] = obj
#                 t_dict['subject'] = sub
#
#                 #如果没有出现过
#                 if t_dict not in ret_list:
#                     ret_list.append(deepcopy(t_dict))
#
#     return deepcopy(ret_list)