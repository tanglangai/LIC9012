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
from allennlp.modules.seq2seq_encoders.gated_cnn_encoder import GatedCnnEncoder

@Model.register('seq2seq_attention')
class Seq2seq_attention(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                seq2seq_encoder: Seq2SeqEncoder,
                 matrix_attention: MatrixAttention,
                 linear_layer: FeedForward = None
                 ) -> None:
        super(Seq2seq_attention, self).__init__(vocab)
        
        self.text_field_embedder = text_field_embedder
        self.seq2seq_encoder = seq2seq_encoder
        self.linear_layer = linear_layer
        
        self.matrix_attention = matrix_attention
        # 如果tensor1是J×1×N×M张量和tensor2是K×M×P张量，将一个J×K×N×P张量。
        # 如果tensor1是J×1×N×M张量和tensor2是K×M×P张量，将一个J×K×N×P张量。

        # # 损失函数的选择？
        # weights = torch.ones(51)
        # weights[0] = 0
        
        self.loss = torch.nn.L1Loss()
    
        
        # from allennlp.training.metrics.f1_measure import F1Measure
        # self.metric = F1Measure(positive_label=0)
        self.metric = Modified_F1()
        
        # self.activation = torch.nn.Softmax(dim=-1)
        self.activation = torch.nn.ReLU()
    
    def forward(self, words_poses_field: Dict[str, torch.Tensor],
                # reserved_pos_field: np.array,
                ner_labels: np.array = None):

        words_poses_mask = get_text_field_mask(words_poses_field)
        words_poses_embeddings = self.text_field_embedder(words_poses_field)
        ner_labels = ner_labels.long()
        assert len(words_poses_embeddings.shape) == 3
        
        if self.linear_layer:
            embeddings = self.linear_layer(words_poses_embeddings)
        else:
            embeddings = words_poses_embeddings
        
        # 经过了seq 2 seq的  编码的结果
        embeddings = self.seq2seq_encoder(embeddings, words_poses_mask)
        
        attention = self.matrix_attention(embeddings, embeddings)
        
        # 输出是batchsize * 50 * seqlen *seqlen,这一条变换成batchsize * seqlen *seqlen * 50
        attention = attention.permute(0, 2, 3, 1)
        
        assert attention.shape[-1] == 50
        
        # 有必要用softmax嘛
        attention = self.activation (attention)
        
        # #我们对最大值放缩一下试一试
        # 失败了
        #RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
        # b = attention.max()
        # attention /= b
        
        output = {'attention_logits': attention}
        
        if ner_labels is not None and ner_labels.sum().item() != 0:
            
            self.metric(attention, ner_labels)
            
            # all_loss = 0
            # batch_size, n, _, _ = ner_labels.shape
            
            
            # for i in range(n):
            #     for j in range(n):
            #         for batch in range(attention.shape[0]):
            #             a = ner_labels[batch, i, j, :]
            #             a = torch.argmax(a)
            #             t = self.loss(attention[batch, i, j, :], a)
            #             all_loss += t
            
            # 让我们看一看只针对那些有标签的位置的预测行不行
            # 其实，我们应该可以随机50%的，50%的
            pos_attention = attention[ner_labels == 1]
            pos_ner_labels = ner_labels[ner_labels == 1]
            
            neg_ner_labels = ner_labels[ner_labels == 0]
            num_of_pos = len(pos_attention)
            neg_attention = attention[ner_labels == 0]
            num_of_neg = len(neg_attention)
            all_num = num_of_pos + num_of_neg
     
            #这个比例可以调整，例如50%  50%还是其他的
            bernoulli_distribution = torch.bernoulli(neg_attention, num_of_neg/(all_num*1e+5))
            neg_attention = neg_attention[bernoulli_distribution.long() == 1]
            neg_ner_labels = neg_ner_labels[bernoulli_distribution.long() == 1]
            
            attention = torch.cat([pos_attention,neg_attention], dim = -1)
            ner_labels = torch.cat([pos_ner_labels,
                                    neg_ner_labels
            ], dim = -1)
            
            # output["loss"] = all_loss
            output["loss"] = self.loss(attention, ner_labels.float())
            # output["loss"] = self.loss(attention, ner_labels.long())
            
            # print(torch.argmax(attention, dim=-1))
        
        return output
    
    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        acc, recall, f1, hs_label_f1 = self.metric.get_metric(reset)
        return {"precision": acc, "recall": recall, 'f1_score': f1, "has_label_f1": hs_label_f1}

