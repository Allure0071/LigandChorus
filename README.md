# LigandChorus

LigandChorus is the E18 research implementation of structure-aware parallel decoding for ligand-conditioned fixed-backbone enzyme sequence design. It freezes the LigandMPNN encoder and trains a three-layer graph decoder to recover masked residues. Inference starts from an all-mask sequence, reuses the structural encoding, and applies region-specific sampling temperatures.

## Setup

```sh
python -m pip install -e .
```

The training and inference entry points are available through `python -m ligandchorus.train --help` and `python -m ligandchorus.infer --help`.

## Structural inputs

Both commands take preprocessed NumPy `.npz` records. Use the following keys.

| Key | Shape | Meaning |
| --- | --- | --- |
| `X` | `[L, 4, 3]` | Backbone coordinates in Å, in N, Cα, C, O order |
| `S` | `[L]` | Native sequence tokens used for training |
| `residue_mask` | `[L]` | Mask for valid backbone positions |
| `chain_labels` | `[L]` | Integer chain identifiers |
| `Y` | `[A, 3]` | Context atom coordinates in Å |
| `Y_t` | `[A]` | Context atom atomic numbers |

Token IDs 0–19 follow `ACDEFGHIKLMNPQRSTVWY`. Token 20 denotes a mask or an unknown residue. During inference, `S` provides only the record shape and may contain placeholder tokens.

## Training and inference

The training manifest is JSON Lines with one `{"path": "structures/example.npz"}` entry per training assembly. Relative paths are resolved against the manifest's directory. Supply the official `ligandmpnn_v_32_010_25.pt` checkpoint to initialize the frozen encoder. An inference run takes the resulting full E18 state dictionary.

```sh
python -m ligandchorus.train --manifest train.jsonl --encoder-weights ligandmpnn_v_32_010_25.pt --output e18.pt
python -m ligandchorus.infer --input structure.npz --weights e18.pt --output samples.npy
```

The default inference settings generate ten sequences with up to 20 parallel passes. The sampler uses temperature 0.29 for positions whose Cα is within 5 Å of a context atom and 0.24 elsewhere. It retains previously sampled residues and stops once no mask tokens remain. The output contains the sampled token IDs at the original backbone positions. Checkpoints and structural datasets are supplied separately.

## Upstream

The vendored encoder and graph layers derive from [LigandMPNN](https://github.com/dauparas/LigandMPNN). Its MIT license and copyright notice are included in [LIGANDMPNN_LICENSE](ligandchorus/_vendor/LIGANDMPNN_LICENSE).
