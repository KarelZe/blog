import torch


def SoftBoundaryLoss(emb: torch.Tensor, r: float, c: torch.Tensor, nu: float) -> float:
    """Soft-boundary loss.

    Args:
        emb (torch.Tensor): embedding
        r (float): radius
        c (torch.Tensor): centroid
        nu (float): weight term

    Returns:
        float: loss
    """
    dist = torch.sum((emb - c) ** 2, dim=1)
    scores = dist - r**2
    loss = r**2 + (1 / nu) * torch.mean(torch.relu(scores))

    return loss
