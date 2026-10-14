"""
losses.py
"""
import torch
import torch.nn.functional as F


def focal_cox_loss(risk, times, events, gamma=1.0):
    """Focal Cox loss: down-weight easy non-events to focus on hard cases."""
    risk = risk.squeeze(-1)
    _, idx = torch.sort(times, descending=True)
    risk_sorted = risk[idx]
    events_sorted = events[idx]

    log_cumsum = torch.logcumsumexp(risk_sorted, dim=0)
    # Breslow handling: every subject tied at a time uses the complete risk set
    # at that time (including competing events/censoring at the same time).
    _, counts = torch.unique_consecutive(times[idx], return_counts=True)
    ends = torch.cumsum(counts, dim=0) - 1
    log_cumsum = torch.repeat_interleave(log_cumsum[ends], counts)
    neg_ll = -(risk_sorted - log_cumsum) * events_sorted

    # Focal weight: higher for samples with lower predicted risk (harder)
    with torch.no_grad():
        p_event = torch.sigmoid(risk_sorted)
        focal_w = (1 - p_event) ** gamma * events_sorted + \
                  (1 - (1 - p_event)) ** gamma * (1 - events_sorted)
    return (neg_ll * focal_w).sum() / (events_sorted.sum() + 1e-8)


def cindex_ranking_loss(risk, times, events):
    """Differentiable C-index ranking loss (concordance index surrogate)."""
    risk = risk.squeeze(-1)
    n = risk.size(0)
    if n < 2:
        return torch.tensor(0.0, device=risk.device, requires_grad=True)

    # Comparable pairs: i has event, j has longer time
    risk_i = risk.unsqueeze(1)    # (n, 1)
    risk_j = risk.unsqueeze(0)    # (1, n)
    time_i = times.unsqueeze(1)
    time_j = times.unsqueeze(0)
    event_i = events.unsqueeze(1)

    comparable = (time_i < time_j) & (event_i == 1)
    if comparable.sum() == 0:
        return torch.tensor(0.0, device=risk.device, requires_grad=True)

    diff = risk_i - risk_j
    loss = (1.0 - torch.sigmoid(diff)) * comparable.float()
    return loss.sum() / (comparable.sum() + 1e-8)


def total_loss(risk, times, events, cox_weight=1.0, rank_weight=0.5,
               focal_gamma=1.5, device=None):
    """Combined Cox + ranking loss (focal Cox)."""
    l_cox = focal_cox_loss(risk, times, events, gamma=focal_gamma)
    l_rank = cindex_ranking_loss(risk, times, events)
    loss = cox_weight * l_cox + rank_weight * l_rank
    return loss, l_cox.item(), l_rank.item()
