from typing import Optional

import torch

from allennlp.training.metrics.metric import Metric
from allennlp.training.metrics.f1_measure import F1Measure
from allennlp.common.checks import ConfigurationError

from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support
@Metric.register("modified_f1")
class Modified_F1(F1Measure):
    """
    Computes Precision, Recall and F1 with respect to a given ``positive_label``.
    For example, for a BIO tagging scheme, you would pass the classification index of
    the tag you are interested in, resulting in the Precision, Recall and F1 score being
    calculated for this tag only.
    """

    def __init__(self):
        self.pre_score = 0
        self.recall_score = 0
        self.f1_score = 0
        
        self.has_label_f1 = 0
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)
        
        assert predictions.shape[-1] == 50
        
        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to F1Measure contains an id >= {}, "
                                     "the number of classes.".format(num_classes))
        
        if mask is None:
            mask = (gold_labels != -1).float()
        
        yuzhi = 0.5
        predictions[predictions > yuzhi] = 1
        predictions[predictions <= yuzhi] = 0
        
        # gold_labels = torch.argmax(gold_labels, dim=-1)
        # mask = mask.sum(-1).ne(0)
        
        a = predictions[mask.long() == 1].long()
        b = gold_labels[mask.long() == 1].long()
        
        numel = a.numel()
        print()
        
        # 包含0的位置
        print((((a == b).sum()).item() / numel))
        
        #只看 有关系的部分到底能够预测对多少
        a = a[b == 1]
        b = b[b == 1]
        print('-'*10)
        print((((a == b).sum()).item() / numel))
        print()
        
        mask = mask.long()
        gold_labels = gold_labels.long()
        
        #通过mask将有效的值提取出来
        all_predictions = predictions[mask == 1]
        all_gold_labels = gold_labels[mask == 1]
        a = all_predictions
        b = all_gold_labels
        # 这一步是为了只看在 有标签的目标上面，到底预测了多少有标签的值
        # all_predictions = all_predictions[all_gold_labels == 1]
        # all_gold_labels = all_gold_labels[all_gold_labels == 1]
        
        assert all_predictions.shape == all_gold_labels.shape
        
        # [None, 'binary' (default), 'micro', 'macro', 'samples', \
        #                'weighted']
        
        average = 'micro'
        
        # labels = list(range(1, 51))
        
        self.pre_score, self.recall_score, self.f1_score, _ = precision_recall_fscore_support(all_gold_labels,
                                                                                                  all_predictions,
                                                                                                  average=average,
                                                                                                  # labels = labels
                                                                                                  )
        
        
        all_predictions = all_predictions[all_gold_labels == 1]
        all_gold_labels = all_gold_labels[all_gold_labels == 1]
        self.has_label_f1 = f1_score(all_gold_labels, all_predictions)
        
        # if self.f1_score > 0.95 and self.has_label_f1 <0.01:
        #     assert len(a) != len(all_predictions)
        #     print('#'*20)
        #     print(a)
        #     print((a == 1).sum())
        #     print((a == 0).sum())
        #     print('-'*10)
        #     print(b)
        #     print((b==1).sum())
        #     print((b==0).sum())
        #     print('#'*20)
            
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """

        return self.pre_score, self.recall_score, self.f1_score, self.has_label_f1
    
    def reset(self):
        self.pre_score = 0
        self.recall_scorel = 0
        self.f1_score = 0
        self.has_label_f1 = 0