from typing import Optional

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("my_metric")
class Mymetric(Metric):
    """
    This ``Metric`` calculates the mean absolute error (MAE) between two tensors.
    """
    def __init__(self) -> None:
        self.acc_num = 0
        self.all_item = 0

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ...).
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predictions``.
        mask: ``torch.Tensor``, optional (default = None).
            A tensor of the same shape as ``predictions``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)
        gold_labels = gold_labels.squeeze()
        gold_labels = gold_labels.long()
        predictions = torch.argmax(predictions, -1)
        
        assert predictions.shape == gold_labels.shape
        print()
        print("predictions are")
        print(predictions)
        print("gold_labels are")
        print(gold_labels)
        print()
        acc = (predictions == gold_labels).sum().item()
       
        
        self.acc_num +=acc
        self.all_item += gold_labels.numel()
    
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated mean absolute error.
        """
        v = self.acc_num/(self.all_item + 1e-14)
        if reset:
            self.reset()
        return  v

    @overrides
    def reset(self):

        self.acc_num = 0
        self.all_item = 0