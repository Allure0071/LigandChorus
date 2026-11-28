"""The E18 structure-aware parallel decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._legacy_guidance import LigandGuidedSampler
from ._vendor.model_utils import DecLayer, ProteinMPNN, cat_neighbors_nodes
from .masking import corrupt_data


def build_model(device):
    """Construct the architecture corresponding to existing E18 checkpoints."""
    encoder = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=32,
        device=device,
        atom_context_num=25,
        model_type="ligand_mpnn",
        ligand_mpnn_use_side_chain_context=True,
    )
    return StructAwareFlowModel(encoder).to(device)


class StructAwareFlowModel(nn.Module):
    """结构感知流模型：冻结 LigandMPNN 编码器 + DecLayer 解码器"""

    def __init__(self, ligandmpnn, hidden_dim=128, vocab_size=21,
                 use_guidance=True, guidance_scale=1.0, num_decoder_layers=3):
        super().__init__()
        self.ligandmpnn = ligandmpnn
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.mask_token = vocab_size - 1
        self.use_guidance = use_guidance

        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.seq_embedding = nn.Embedding(vocab_size, hidden_dim)
        # 结构感知解码器：DecLayer 输入维度 = h_V(邻) + h_S(邻) + h_E = 3*hidden
        self.decoder_layers = nn.ModuleList(
            [DecLayer(hidden_dim, hidden_dim * 3, dropout=0.0) for _ in range(num_decoder_layers)])
        self.W_out = nn.Linear(hidden_dim, vocab_size, bias=True)

        if use_guidance:
            self.guided_sampler = LigandGuidedSampler(
                hidden_dim=hidden_dim, vocab_size=vocab_size,
                use_attention=True, use_chemical=True, use_temperature=True,
                guidance_scale=guidance_scale)

        self.ligandmpnn.requires_grad_(False)
        self.ligandmpnn.eval()

    def train(self, mode=True):
        super().train(mode)
        self.ligandmpnn.eval()
        return self

    def forward(self, feature_dict, noisy_sequence, t):
        with torch.no_grad():
            encoded = self.ligandmpnn.encode(feature_dict)
        return self.decode(encoded, feature_dict["mask"], noisy_sequence, t)

    def decode(self, encoded, mask, noisy_sequence, t):
        _, L = noisy_sequence.shape
        h_V, h_E, E_idx = encoded

        t_emb = self.time_embedding(t.unsqueeze(-1)).unsqueeze(1).expand(-1, L, -1)
        h_V = h_V + t_emb

        h_S = self.seq_embedding(noisy_sequence)              # [B,L,H]
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)           # [B,L,K,2H]

        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)     # [B,L,K,3H]
            h_V = layer(h_V, h_ESV, mask)                     # mask_V=mask

        return self.W_out(h_V)

    def compute_loss(self, feature_dict, true_sequence, *, loss_mask=None, generator=None):
        B, L = true_sequence.shape
        device = true_sequence.device
        t = torch.rand(B, device=device, generator=generator)
        noisy_sequence, mask_positions = corrupt_data(
            true_sequence, t, self.mask_token, generator=generator,
        )
        if loss_mask is not None:
            mask_positions = mask_positions & loss_mask
            # Keep short target chains measurable when random masking misses them.
            first_target = loss_mask & (loss_mask.long().cumsum(dim=1) == 1)
            fallback = ~mask_positions.any(dim=1, keepdim=True) & first_target
            mask_positions = mask_positions | fallback
            # Supply the other chains as structure without exposing their native tokens.
            noisy_sequence = noisy_sequence.masked_fill(~loss_mask | fallback, self.mask_token)
            feature_dict = {**feature_dict, "S": true_sequence.masked_fill(~loss_mask, self.mask_token)}
        logits = self.forward(feature_dict, noisy_sequence, t)
        if loss_mask is not None:
            return F.cross_entropy(logits[mask_positions], true_sequence[mask_positions])
        loss_all = F.cross_entropy(logits.view(-1, self.vocab_size),
                                   true_sequence.view(-1), reduction='none').view(B, L)
        loss_masked = loss_all * mask_positions.float()
        return loss_masked.sum() / mask_positions.float().sum().clamp(min=1)

    def compute_substitution_loss(self, feature_dict, clean_sequence, noisy_sequence, t, *, encoded=None):
        """Predict clean residue identities from externally corrupted tokens (plan 2)."""
        valid = (feature_dict["mask"].bool() & feature_dict["chain_mask"].bool()
                 & (clean_sequence >= 0) & (clean_sequence < self.mask_token))
        if not valid.any():
            raise ValueError("Substitution loss requires at least one labeled design position")
        if encoded is None:
            logits = self.forward(feature_dict, noisy_sequence, t)
        else:
            logits = self.decode(encoded, feature_dict["mask"], noisy_sequence, t)
        return F.cross_entropy(logits[valid][:, :self.mask_token].float(), clean_sequence[valid])
