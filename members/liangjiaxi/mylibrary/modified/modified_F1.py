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
        
        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to F1Measure contains an id >= {}, "
                                     "the number of classes.".format(num_classes))
        
        if mask is None:
            mask = (gold_labels != -1).float()
        
        predictions = torch.argmax(predictions, dim=-1)
        gold_labels = torch.argmax(gold_labels, dim=-1)
        mask = mask.sum(-1).ne(0)
        
        mask = mask.long()
        gold_labels = gold_labels.long()
        
        #通过mask将有效的值提取出来
        all_predictions = predictions[mask == 1]
        all_gold_labels = gold_labels[mask == 1]
        assert all_predictions.shape == all_gold_labels.shape
        
        # [None, 'binary' (default), 'micro', 'macro', 'samples', \
        #                'weighted']
        
        average = 'weighted'
        
        labels = list(range(1, 51))
        self.pre_score, self.recall_score, self.f1_score, _ = precision_recall_fscore_support(all_gold_labels,
                                                                                           all_predictions,
                                                                                           average = average,
                                                                                           labels = labels)

        

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """

        return self.pre_score, self.recall_score, self.f1_score
    
    def reset(self):
        self.pre_score = 0
        self.recall_scorel = 0
        self.f1_score = 0
