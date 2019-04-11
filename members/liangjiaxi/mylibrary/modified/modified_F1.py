from typing import Optional

import torch

from allennlp.training.metrics.metric import Metric
from allennlp.training.metrics.f1_measure import F1Measure
from allennlp.common.checks import ConfigurationError


@Metric.register("modified_f1")
class Modified_F1(F1Measure):
    """
    Computes Precision, Recall and F1 with respect to a given ``positive_label``.
    For example, for a BIO tagging scheme, you would pass the classification index of
    the tag you are interested in, resulting in the Precision, Recall and F1 score being
    calculated for this tag only.
    """
    def __init__(self) -> None:
        self._positive_label =1
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0

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
        
        predictions = predictions[:, :, :, 1:]
        gold_labels = gold_labels[:, :, :, 1:]
        
        assert gold_labels.shape[3] == 50
        
        if mask is None:
            mask = (gold_labels != -1).float()
        
        predictions = torch.argmax(predictions, dim=-1)
        gold_labels = torch.argmax(gold_labels, dim=-1)
        mask = mask.sum(-1).ne(0)
        
        mask = mask.float()
        gold_labels = gold_labels.long()
        positive_label_mask = gold_labels.ne(49).float()
        negative_label_mask = 1.0 - positive_label_mask
        

        # True Negatives: correct non-positive predictions.
        correct_null_predictions = (predictions !=
                                    gold_labels).float() * negative_label_mask
        self._true_negatives += (correct_null_predictions.float() * mask).sum()

        # True Positives: correct positively labeled predictions.
        correct_non_null_predictions = (predictions ==
                                        gold_labels).float() * positive_label_mask
        self._true_positives += (correct_non_null_predictions * mask).sum()

        # False Negatives: incorrect negatively labeled predictions.
        incorrect_null_predictions = (predictions !=
                                      gold_labels).float() * positive_label_mask
        self._false_negatives += (incorrect_null_predictions * mask).sum()

        # False Positives: incorrect positively labeled predictions
        incorrect_non_null_predictions = (predictions ==
                                          gold_labels).float() * negative_label_mask
        self._false_positives += (incorrect_non_null_predictions * mask).sum()

