"""Five-round masked reconstruction on the frozen E18 comparison panel."""

import torch


def masked_training_input(native, valid, stage, generator=None):
    """Uniformly reveal the starting quota for round stage (1 through 5)."""
    lengths = valid.sum(-1)
    known = ((stage - 1) * lengths + 4) // 5
    known = torch.minimum(known, (lengths - 1).clamp_min(0))
    priority = torch.rand(native.shape, device=native.device, generator=generator)
    ranks = priority.masked_fill(~valid, 2).argsort(dim=-1).argsort(dim=-1)
    reveal = valid & (ranks < known[:, None])
    return native.masked_fill(~reveal, 20), known / lengths.clamp_min(1), valid & ~reveal


def quota_trajectory(model, encoded, mask, design, temperature, generator):
    state = torch.full_like(design, 20, dtype=torch.long)
    lengths = design.sum(-1)
    history = [state]
    for round_no in range(1, 6):
        known = design & state.ne(20)
        clock = known.sum(-1) / lengths.clamp_min(1)
        logits = model.decode(encoded, mask, state, clock)[..., :20].float()
        confidence = logits.softmax(-1).amax(-1)
        pending = design & ~known
        ranks = confidence.masked_fill(~pending, -1).argsort(dim=-1, descending=True, stable=True).argsort(dim=-1)
        quota = (round_no * lengths + 4) // 5 - known.sum(-1)
        fill = pending & (ranks < quota[:, None])
        state = state.clone()
        if fill.any():
            weights = (logits / temperature).softmax(-1)
            state[fill] = torch.multinomial(weights[fill], 1, generator=generator).flatten()
        history.append(state)
    return torch.stack(history)
