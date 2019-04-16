"""
解码阶段，
对于制造的假数据，我们想要找到好的解码方式来将数据解码。
"""
from typing import Iterator, List, Dict, Callable

import demjson
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.fields import TextField, ArrayField, SequenceLabelField, MetadataField, ListField
import torch

from common_utils.scheme_mapping import generaterows

"""数据格式
words: List[str]
poses: List[str]  这两部分假设代表的是我们从上层获得的信息。

entity_pair: List[int, int] 代表的是位置
label: int，代表这对词属于51种视图中的哪一个
"""

@DatasetReader.register('decoder_datareader')
class Decoder_DataReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 lazy: bool = True) -> None:
        super().__init__(lazy=lazy)
        self.token_indexers = token_indexers

    def _read(self, file_path: str) -> Iterator[Instance]:
        for line in generaterows(
            file_path
        ):
            data = demjson.decode(line)
            words = data['words']
            poses = data['poses']
            entity_pair = data['entity_pair']
            label = data['label']
            words_poses_tokens = []
            for word, pos in zip(words, poses):
                tmp_token = Token(word, tag_=pos)
                words_poses_tokens.append(tmp_token)
                
            label = torch.LongTensor([label])
            entity_pair = torch.LongTensor(entity_pair)
            yield self.text_to_instance(words_poses_tokens, entity_pair, label)
        
    def text_to_instance(self,
                         words_poses_tokens: List,
                         entity_pair: torch.Tensor,
                         labels_tensor: torch.Tensor = None
                         ) -> Instance:
        words_poses_field = TextField(words_poses_tokens, self.token_indexers)
        
        if labels_tensor is not None:
            labels_np = labels_tensor
        else:

            labels_np = torch.zeros(1)
        entity_pair_field = ArrayField(entity_pair, padding_value=-1)
        ner_field = ArrayField(labels_np, padding_value=-1)
        
        fields = {'words_poses_field': words_poses_field,
                  "entity_pair": entity_pair_field,
                  "ner_labels": ner_field}

        return Instance(fields)