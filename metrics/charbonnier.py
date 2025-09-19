import torch


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    # smooth |x| ≈ sqrt(x^2 + eps^2); robust like L1 but smooth near 0
    diff = pred - target
    loss = torch.sqrt(diff * diff + eps * eps)
    return loss.mean()
