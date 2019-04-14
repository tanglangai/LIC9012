"""
梁家熙

写的数据读取类
结构：
bert读取，51种label
"""

from typing import Iterator, List, Dict, Callable

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.fields import TextField, ArrayField, SequenceLabelField, MetadataField, ListField
import torch


import json
import numpy as np

from members.liangjiaxi.mylibrary.Config import Config
from common_utils.scheme_mapping import generaterows, scheme2index, postag2wordpos, convert_spolist2tensor, \
    convert2tensor
from dataUtils.make_label_from_raw import analysis_one_sample

torch.manual_seed(1)

@DatasetReader.register('bert_51')
class Bert51_DataReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 lazy: bool = True) -> None:
        super().__init__(lazy=lazy)
        self.token_indexers = token_indexers
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        """
        如果预处理，提前做好labels，训练集就占了42G大小的空间，
        这必然是应该放弃的。那么,我们就要在这里引入函数。
        一遍处理，一边读取，并不需要将一个非常稀疏的矩阵存起来，再读取！
        """
        for line in generaterows(file_path):
            jsondata = json.loads(line)
            if len(jsondata['postag']) > 50:
                continue
                
            #加载{1，视图一} 这种字典
            scheme_pth = Config.scheme_pth
            label_scheme_dict = scheme2index(scheme_pth)
            
            #读取postag，空则说明数据有问题，忽视之
            postag = jsondata['postag']
            if not postag:
                continue
            try:
                data_info = analysis_one_sample(jsondata)
            except ValueError as e:
                print(e)
                continue
            n = len(jsondata['postag'])
            relations = data_info['relations']
            object_locates = data_info['object_locates']
            subject_locates = data_info['subject_locates']
            object_type = data_info['object_type']
            subject_type = data_info['subject_type']

            labels_tensor = convert2tensor(relations, object_locates, subject_locates,
                                           object_type, subject_type, label_scheme_dict, n)
            
        
            # replace_some_nourns将一些名词替换成其他的
            words = [ins['word'] for ins in jsondata['postag']]
            poses = [ins['pos'] for ins in jsondata['postag']]
            from common_utils.preprocess_merge_something import replace_some_nourns
            words, poses, position_dict = replace_some_nourns(words, poses)
            # 注意这里转换成了Token
            words_poses_tokens = []
            # # 注意新添了一个列表,表示是不是名词啊之类的
            # #是名词则意味着可能是答案，至于动词形容词那肯定就不是答案了。
            # reserved_pos = []
            for word, pos in zip(words, poses):
                # if pos in Config.RESERVED_LSIT:
                #     tmp_id = 1
                # else:
                #     tmp_id = 0
            #
            #     reserved_pos.append(tmp_id)

                # 关键的一步，将tag标签赋予Token，而且配置文件里的token_indexers参数要添加一个 pos_tag
                tmp_token = Token(word, tag_=pos)
                words_poses_tokens.append(tmp_token)

            # yield self.text_to_instance(words_poses_tokens, reserved_pos, labels_tensor)
            yield self.text_to_instance(words_poses_tokens, labels_tensor)
        
    def text_to_instance(self, words_poses_tokens: List[Token],
                         # reserved_pos: List[int],
                         labels_tensor: torch.Tensor = None) -> Instance:
        
        words_poses_field = TextField(words_poses_tokens, self.token_indexers)
        # reserved_pos_field = ArrayField(np.array(reserved_pos))
        
        # ner_field = SequenceLabelField(labels=ner_labels, sequence_field=words_poses_field)
        if labels_tensor is not None:
            labels_np = labels_tensor
        else:
            n = len(words_poses_field)
            labels_np = torch.zeros((n, n, 51))
            
        ner_field = ArrayField(labels_np, padding_value=-1)
        
        # ner_field=MetadataField(ner_labels)
        
        fields = {'words_poses_field': words_poses_field,
                  # "reserved_pos_field": reserved_pos_field,
                  "ner_labels": ner_field}
        
        return Instance(fields)
