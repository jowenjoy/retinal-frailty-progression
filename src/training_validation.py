"""
Validation used within training for checkpoint selection.
"""
import numpy as np
import torch


def evaluate_model(model, dataloader, device, use_covar=True):
    """Run model on a DataLoader, return C-index."""
    model.eval()
    all_risks, all_times, all_events = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            img_feats, covar, times, events, _ = [x.to(device) for x in batch]
            if not use_covar:
                covar = torch.zeros_like(covar)
            risk = model(img_feats, covar)
            all_risks.extend(risk.squeeze(-1).cpu().numpy().tolist())
            all_times.extend(times.cpu().numpy().tolist())
            all_events.extend(events.cpu().numpy().tolist())

    risks = np.array(all_risks)
    times = np.array(all_times)
    events = np.array(all_events, dtype=int)

    try:
        from lifelines.utils import concordance_index
        cindex = concordance_index(times, -risks, events)
    except Exception:
        cindex = 0.5

    return {'cindex': cindex}
