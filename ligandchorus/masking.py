"""Time-conditioned masking during E18 training."""

import torch


def corrupt_data(sequence, t, mask_token=20):
    """
    按时间步 t 损坏序列（离散流匹配的前向过程）

    Args:
        sequence: [B, L] 真实序列（0-19 为氨基酸，20 为 mask）
        t: [B] 时间步，范围 [0, 1]
        mask_token: mask 的 token ID（默认 20）

    Returns:
        noisy_sequence: [B, L] 损坏后的序列
        mask_positions: [B, L] bool，哪些位置被 mask
    """
    B, L = sequence.shape
    device = sequence.device

    # 每个位置以概率 t 保留真实氨基酸，1-t 替换为 mask
    # 注意：t 越大，保留越多（t=1 时完全保留）
    keep_mask = torch.rand(B, L, device=device) < t.unsqueeze(-1)

    noisy_sequence = torch.where(
        keep_mask,
        sequence,
        torch.full_like(sequence, mask_token)
    )

    mask_positions = ~keep_mask

    return noisy_sequence, mask_positions
