from torchmetrics import Metric
import torch
import numpy as np
import sklearn.metrics

class CrossEntropyLossMetric(Metric):
    """
    Wrapper around the torch.nn.CrossEntropyLoss for use
    with torchmetrics.
    """

    def __init__(self):
        super().__init__()
        self.add_state("summed_loss", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("total_num", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum")
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.size(0) != target.size(0):
            raise ValueError("Both tensors must have the same number of batches.")
        self.summed_loss += self.loss_fn(preds, target)
        self.total_num += preds.shape[0]

    def compute(self) -> torch.Tensor:
        return self.summed_loss.float() / self.total_num
    
class BinaryF1Score(Metric):
    """
    Computes the Binary F1 Score.
    """

    def __init__(self):
        super().__init__()
        self.add_state("tp_total", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("fn_total", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("fp_total", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("tn_total", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.size(0) != target.size(0):
            raise ValueError("Both tensors must have the same number of batches.")

        # Update TP, FN, FP, TN
        tn, fp, fn, tp = np.ravel(sklearn.metrics.confusion_matrix(
                                  target.cpu(), preds.cpu(), labels=[0, 1], normalize=None), "C")
        self.tp_total += tp
        self.fn_total += fn
        self.fp_total += fp
        self.tn_total += tn

    def compute(self) -> torch.Tensor:
        precision = self.tp_total / (self.tp_total + self.fp_total) 
        recall = self.tp_total / (self.tp_total + self.fn_total)
        return torch.nan_to_num(2 * (precision * recall) / (precision + recall))

class CosineSimilarityMetric(Metric):
    """
    Metric for calculating and accumulating cosine similarity loss
    """
    def __init__(self):
        super().__init__()
        # Add state variables
        self.add_state("total_similarity", 
                      default=torch.tensor(0.0, dtype=torch.float64), 
                      dist_reduce_fx="sum")
        self.add_state("total_num", 
                      default=torch.tensor(0, dtype=torch.float64), 
                      dist_reduce_fx="sum")
        # Cosine similarity loss function
        self.cos_sim = torch.nn.CosineSimilarity(dim=1)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update metric states
        Args:
            preds: Predicted values [batch_size, 24]
            target: Ground truth values [batch_size, 24]
        """
        if preds.size() != target.size():
            raise ValueError("Prediction and target tensors must have the same shape")
            
        # Calculate cosine similarity
        similarity = self.cos_sim(preds, target)
        
        # Update states
        self.total_similarity += similarity.sum()
        self.total_num += preds.shape[0]

    def compute(self) -> torch.Tensor:
        """Calculate average similarity"""
        return self.total_similarity / self.total_num