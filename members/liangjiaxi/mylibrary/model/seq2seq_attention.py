"""
梁家熙

embedding + seq2seq + matrix attention 解码

"""

from typing import Iterator, List, Dict, Optional
import torch
import json
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from allennlp.commands.train import train_model
from allennlp.common.params import Params
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder, MatrixAttention
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.data.token_indexers import PretrainedBertIndexer
from members.liangjiaxi.mylibrary.modified.modified_F1 import Modified_F1


@Model.register('seq2seq_attention')
class Seq2seq_attention(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                
                 matrix_attention: MatrixAttention,
                 linear_layer: FeedForward = None
                 ) -> None:
        super(Seq2seq_attention, self).__init__(vocab)
        
        self.text_field_embedder = text_field_embedder
        
        self.linear_layer = linear_layer
        
        self.matrix_attention = matrix_attention
        # 如果tensor1是J×1×N×M张量和tensor2是K×M×P张量，将一个J×K×N×P张量。
        # 如果tensor1是J×1×N×M张量和tensor2是K×M×P张量，将一个J×K×N×P张量。

        # 损失函数的选择？
        weights = torch.ones(51)
        weights[0] = 1.0 / 600
        
        self.loss = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
        
        # from allennlp.training.metrics.f1_measure import F1Measure
        # self.metric = F1Measure(positive_label=0)
        self.metric = Modified_F1()
        
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, words_poses_field: Dict[str, torch.Tensor],
                # reserved_pos_field: np.array,
                ner_labels: np.array = None):
        # 没有了encoder，mask也没啥用了
        # words_poses_mask = get_text_field_mask(words_poses_field)
        words_poses_embeddings = self.text_field_embedder(words_poses_field)
        ner_labels = ner_labels.long()
        assert len(words_poses_embeddings.shape) == 3
        
        if self.linear_layer:
            embeddings = self.linear_layer(words_poses_embeddings)
        else:
            embeddings = words_poses_embeddings
        
        # TODO(梁家熙)过滤掉v词的一些行
        # 将一些embedding 置为0，可以加速运算（？）
        # reserved_pos_field = reserved_pos_field.byte()
        # embeddings[1 - reserved_pos_field] = 0
        
        attention = self.matrix_attention(embeddings, embeddings)
        
        # 输出是batchsize * 51 * seqlen *seqlen,这一条变换成batchsize * seqlen *seqlen * 51
        attention = attention.permute(0, 2, 3, 1)
        assert attention.shape[-1] == 51
        
        # #将不可能出现的赋予一个很大的负值，这样经过softmax之后就是0了
        # attention[reserved_pos_field, :, :] = -9999
        # attention = attention.permute(0, 2, 1, 3)
        # attention[reserved_pos_field, :, :] = -9999
        # attention = attention.permute(0, 2, 1, 3)
        
        attention = self.softmax(attention)
        output = {'attention_logits': attention}
        
        if ner_labels is not None and ner_labels.sum().item() != 0:
            
            self.metric(attention, ner_labels)
            
            all_loss = 0
            batch_size, n, _, _ = ner_labels.shape
            # for batch in range(batch_size):
            #     for i in range(n):
            #         for j in range(n):
            #             a = attention[batch, i, j, :].sum()
            #             b = ner_labels[batch, i, j, :].sum()
            #             c = ner_labels[batch, i, j, :]
            #             if b.item() > 1:
            #                 print()
            #             t = self.loss(attention[batch, i, j, :], ner_labels[batch, i, j, :])
            for i in range(n):
                for j in range(n):
                    a = ner_labels[:, i, j, :]
                    a = torch.argmax(a, dim=-1)
                    t = self.loss(attention[:, i, j, :], a)
                    all_loss += t
            output["loss"] = all_loss
            
            # print(torch.argmax(attention, dim=-1))
        
        return output
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        acc, recall, f1 = self.metric.get_metric(reset)
        return {"precision": acc, "recall": recall, 'f1_score': f1}

