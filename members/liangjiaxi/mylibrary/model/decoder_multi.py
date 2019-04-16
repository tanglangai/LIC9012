"""
尝试不同的方式来对数据进行解码
去掉attention，换成feedforward
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
from allennlp.modules import Seq2SeqEncoder, MatrixAttention, LayerNorm
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.data.token_indexers import PretrainedBertIndexer
from members.liangjiaxi.mylibrary.modified.modified_F1 import Modified_F1
from members.liangjiaxi.mylibrary.modified.modified_metric import Mymetric


@Model.register('decoder_multi_model')
class Decoder_Multi_model(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,

                 linear_layer: FeedForward ,
    
                 sentence_encoder: Seq2VecEncoder = None,
                 seq2seq_encoder: Seq2SeqEncoder = None,
                 initializer: InitializerApplicator = InitializerApplicator()
                 
                 ) -> None:
        super(Decoder_Multi_model, self).__init__(vocab)
        
        self.sentence_encoder = sentence_encoder
        self.seq2seq_encoder = seq2seq_encoder
        self.linear_layer = linear_layer

        # 如果tensor1是J×1×N×M张量和tensor2是K×M×P张量，将一个J×K×N×P张量。
        # 如果tensor1是J×1×N×M张量和tensor2是K×M×P张量，将一个J×K×N×P张量。

        # 损失函数的选择？
        # self.loss = F.poisson_nll_loss
        
        weights = torch.ones(51)
        
        weights[0] = 1.0 / 100
        self.loss = torch.nn.CrossEntropyLoss(weight=weights)
  
        # self.loss = torch.nn.CrossEntropyLoss()
        
        # from allennlp.training.metrics.f1_measure import F1Measure
        # self.metric = F1Measure(positive_label=1)
        self.metric = Mymetric()
    
        self.softmax = torch.nn.Softmax(dim=-1)
        self.text_field_embedder = text_field_embedder
  
        # initializer(self)
        
    
    def forward(self, words_poses_field: Dict[str, torch.Tensor],
                entity_pair: torch.Tensor,
                ner_labels: torch.Tensor = None):
        ner_labels = ner_labels.long()
        words_poses_embeddings_mask = get_text_field_mask(words_poses_field)
        words_poses_embeddings = self.text_field_embedder(words_poses_field)
        
        assert len(words_poses_embeddings.shape) == 3
        
        # 如果设置了seq2seq的话
        if self.seq2seq_encoder:
            words_poses_embeddings = self.seq2seq_encoder(words_poses_embeddings, words_poses_embeddings_mask)
            
            
   
        embeddings = words_poses_embeddings
        
        # 如果存在句子级别的embedding的话
        if self.sentence_encoder:
            sentence_embedding = self.sentence_encoder(embeddings, words_poses_embeddings_mask)
            # 为了维度好拼接
            sentence_embedding = sentence_embedding.unsqueeze(1)
            sentence_embedding = sentence_embedding.expand_as(embeddings)
            embeddings = torch.cat([embeddings, sentence_embedding], dim = -1) # torch.Size([batchsize, seq_len, dim])
            

        obj_idx = entity_pair[:, 0].long()
        sub_idx = entity_pair[:, 1].long()
        
        #准确找到实体所对应的向量
        obj_tensor_lst = []
        for i, o in enumerate(obj_idx):
            obj_tensor_lst.append(embeddings[i, o, :].unsqueeze(0))
        sub_tensor_lst = []
        for j,s in enumerate(sub_idx):
            sub_tensor_lst.append(embeddings[j, s, :].unsqueeze(0))
        obj = torch.cat(obj_tensor_lst, 0)
        sub = torch.cat(sub_tensor_lst, 0)
        sub = sub[:,:(sub.shape[-1] // 2)]
        attention = self.linear_layer(torch.cat([obj,sub], -1))
        
        
        assert attention.shape[-1] == 51
        attention = attention.squeeze()
        
        # attention = self.softmax(attention)
        output = {'attention_logits': attention}
        
        # assert ner_labels is not None
        
        if ner_labels is not None:
            # ####################################################################
            # print(ner_labels.view(-1).shape)
            # # print(attention.shape)
            # self.metric(attention, ner_labels)
            # # 肯定会经常报错
            # a = torch.zeros(ner_labels.shape[0],51).cuda()
            # for i, o in enumerate(ner_labels.view(-1).long()):
            #     a[i,o] = 1
            # # a[:,ner_labels.long()] = 1
            # ner_labels = a
            # loss = torch.abs(ner_labels.view(-1) - attention.view(-1)).sum()
            #
            # # output["loss"] = self.loss(attention, ner_labels.long().view(-1))
            # output["loss"] = loss
            # ######################################################################
            # self.metric(attention, ner_labels)
            # # 肯定会经常报错
            #
            # output["loss"] = self.loss(attention, ner_labels.long().view(-1))

            ####################################################################
   
   
            weights = torch.zeros(51).cuda()
            for i in ner_labels.long().flatten():
                weights[i] += 1
            weights[weights == 0] = 1e+14
            weights = 1 / weights
            self.metric(attention, ner_labels)
            # 肯定会经常报错
            a = torch.zeros(ner_labels.shape[0], 51).cuda()
            for i, o in enumerate(ner_labels.view(-1).long()):
                a[i, o] = 1
            # a[:,ner_labels.long()] = 1
            ner_labels = a
            loss = (torch.abs(ner_labels - attention) * weights).sum()

            # output["loss"] = self.loss(attention, ner_labels.long().view(-1))
            output["loss"] = loss
            ######################################################################
        return output
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        acc= self.metric.get_metric(reset)
        return {"acc ratio": acc}

