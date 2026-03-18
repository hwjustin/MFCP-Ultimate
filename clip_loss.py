from typing import Tuple, List

import torch
from torch import nn
from torch.nn import functional as F


class CLIPLoss(torch.nn.Module):
    """
    Loss function for multimodal contrastive learning based on CLIP.

    Embeddings are L2 normalized and cosine similarity is computed between all
    cross-modal pairs. The loss encourages matching embeddings on the diagonal
    (same subject across modalities).

    Args:
        temperature: Temperature parameter for scaling logits (lower = sharper)
        lambda_0: Weight for modality0->modality1 direction (default 0.5)
    """

    def __init__(self,
                 temperature: float,
                 lambda_0: float = 0.5) -> None:
        super(CLIPLoss, self).__init__()

        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        if lambda_0 > 1 or lambda_0 < 0:
            raise ValueError('lambda_0 must be a float between 0 and 1.')
        self.lambda_0 = lambda_0
        self.lambda_1 = 1 - lambda_0

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, indices: List[int] = None) -> Tuple:
        """
        Compute CLIP contrastive loss.

        Args:
            out0: Embeddings from modality 0 (batch_size, embed_dim)
            out1: Embeddings from modality 1 (batch_size, embed_dim)
            indices: Optional indices (unused, for compatibility)

        Returns:
            loss: Contrastive loss value
            logits: Similarity matrix (batch_size, batch_size)
            labels: Ground truth labels (diagonal)
        """
        # Normalize embeddings onto unit hypersphere
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        # Compute cosine similarity matrix
        logits = torch.matmul(out0, out1.T) / self.temperature
        labels = torch.arange(len(out0), device=out0.device)

        # Bidirectional loss
        loss_0 = self.lambda_0 * self.cross_entropy(logits, labels)
        loss_1 = self.lambda_1 * self.cross_entropy(logits.T, labels)
        loss = loss_0 + loss_1

        return loss, logits, labels


class SupervisedCLIPLoss(CLIPLoss):
    """
    CLIP loss that treats samples with the same compound label as positives.

    This extends the standard CLIPLoss to handle replicates (multiple samples
    from the same compound) by treating them as positive pairs in the contrastive
    loss, rather than only the diagonal.

    Args:
        temperature: Temperature parameter for scaling logits (lower = sharper)
        lambda_0: Weight for modality0->modality1 direction (default 0.5)
    """

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, labels: torch.Tensor) -> Tuple:
        """
        Compute supervised CLIP contrastive loss with compound labels.

        Args:
            out0: Embeddings from modality 0 (batch_size, embed_dim)
            out1: Embeddings from modality 1 (batch_size, embed_dim)
            labels: Compound labels (batch_size,) - samples with same label are positive pairs

        Returns:
            loss: Contrastive loss value
            logits: Similarity matrix (batch_size, batch_size)
            labels: Ground truth labels (for compatibility)
        """
        # Normalize embeddings onto unit hypersphere
        out0 = F.normalize(out0, dim=1)
        out1 = F.normalize(out1, dim=1)

        # Compute cosine similarity matrix
        logits = torch.matmul(out0, out1.T) / self.temperature

        # Build positive mask: same compound -> positive pair
        # Shape: (batch_size, batch_size)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # Normalize each row to sum to 1 (distribute loss across all positives)
        pos_mask = pos_mask / pos_mask.sum(dim=1, keepdim=True).clamp(min=1)

        # Compute log softmax
        log_probs_0 = F.log_softmax(logits, dim=1)
        log_probs_1 = F.log_softmax(logits.T, dim=1)

        # Compute loss: negative log likelihood for positive pairs
        loss_0 = -(pos_mask * log_probs_0).sum(dim=1).mean()
        loss_1 = -(pos_mask.T * log_probs_1).sum(dim=1).mean()

        # Combine bidirectional losses
        loss = self.lambda_0 * loss_0 + self.lambda_1 * loss_1

        return loss, logits, labels
