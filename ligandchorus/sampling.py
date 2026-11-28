"""Regional-temperature sampler used in the E18 benchmark."""

import math

import torch


def regional_temperature(features, inside=0.29, outside=0.24):
    """Use Cα distance to context atoms to choose an inference temperature."""
    distances = torch.cdist(features["X"][:, :, 1, :].unsqueeze(2), features["Y"]).squeeze(2)
    distances = distances + (1.0 - features["Y_m"]) * 1e6
    pocket = distances.min(dim=-1).values[0] < 5.0
    return torch.where(pocket, inside, outside)[None, :, None]


@torch.inference_mode()
def sample(model, features, *, samples=10, steps=20, seed=0, inside=0.29, outside=0.24):
    """Sample in parallel for one backbone and its supplied atomic context."""
    if samples < 1 or steps < 1:
        raise ValueError("samples and steps must be positive")
    if not all(math.isfinite(t) and t > 0 for t in (inside, outside)):
        raise ValueError("temperatures must be finite and positive")
    if features["mask"].shape[0] != 1:
        raise ValueError("Expected features for one structure")
    model.eval()
    encoded = model.ligandmpnn.encode(features)
    context = tuple(tensor.expand(samples, *tensor.shape[1:]) for tensor in encoded)
    device = encoded[0].device
    generator = torch.Generator(device=device).manual_seed(seed)
    length = features["mask"].shape[-1]
    sequence = torch.full((samples, length), model.mask_token, dtype=torch.long, device=device)
    mask = features["mask"].expand(samples, -1)
    temperature = regional_temperature(features, inside, outside)
    for step in range(steps):
        unfilled = sequence == model.mask_token
        if not bool(unfilled.any()):
            break
        t = torch.full((samples,), step / steps, device=device)
        logits = model.decode(context, mask, sequence, t)
        probabilities = (logits / temperature).softmax(dim=-1)
        proposed = torch.multinomial(
            probabilities.flatten(0, 1), 1, generator=generator,
        ).reshape_as(sequence)
        sequence = torch.where(unfilled, proposed, sequence)
    return sequence
