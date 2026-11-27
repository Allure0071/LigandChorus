"""Transition tables used by the two fixed-panel recovery benchmarks."""

import hashlib

import torch
import torch.nn.functional as F


def benchmark_schedule(scores, grade_table=None):
    """Return forward draw weights, cumulative kernels, reverse interval kernels and grid."""
    identity = torch.eye(20, dtype=torch.float64)
    if grade_table is None:
        exchange = torch.exp2(scores / 2)
        exchange /= exchange.sum(1, keepdim=True)
        intervals = [(1 - b) * identity + b * exchange for b in (0.10, 0.25, 0.45, 0.70)]
        intervals.append(torch.full_like(identity, 1 / 20))
        products = [identity]
        for transition in intervals:
            products.append(products[-1] @ transition)
        products = torch.stack(products)
        return products, products, torch.stack(intervals), (0, 1, 2, 3, 4, 5)

    # Author release 5626747. Its original_score uses ARNDCQEGHILKMFPSTWYV.
    if hashlib.sha256(grade_table.read_bytes()).hexdigest() != (
        "b24cde8e763eb99f096191a0828c4cd904bf56d0e989bda45b11707ec6643f33"
    ):
        raise ValueError("Expected the published GraDe-IF blosum_substitute.pt")
    table = torch.load(grade_table, weights_only=True, map_location="cpu")
    order = ["ARNDCQEGHILKMFPSTWYV".index(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"]
    raw = table["original_score"][order][:, order]
    torch.testing.assert_close(raw.double(), scores + 4, rtol=0, atol=0)
    # BlosumTransition receives timesteps+1 and interpolates to timestep+1.
    temps = F.interpolate(table["Qtb_temperature"][None, None], size=502,
                          mode="linear", align_corners=True).flatten()
    index = (torch.arange(501, dtype=torch.float32) / 500 * 501).round().long()
    products = (raw[None] / temps[index, None, None]).softmax(-1).clamp_min(1e-6)
    grid = (0, 100, 200, 300, 400, 500)
    ratios = torch.stack([products[s] / products[t] for s, t in zip(grid, grid[1:])])
    intervals = ratios / ratios.sum(-1, keepdim=True)
    # The upstream forward routine draws from Qbar @ one_hot (a column),
    # while the upstream reverse routine indexes Qbar[clean, observed].
    return products.transpose(1, 2), products, intervals, grid
