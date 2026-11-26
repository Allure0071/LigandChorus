# LigandChorus

LigandChorus is the E18 research implementation of structure-aware parallel decoding for ligand-conditioned fixed-backbone enzyme sequence design. It freezes the LigandMPNN encoder and trains a three-layer graph decoder to recover masked residues. Inference starts from an all-mask sequence, reuses the structural encoding, and applies region-specific sampling temperatures.

## Setup

```sh
python -m pip install -e .
```

The training and inference entry points are available through `python -m ligandchorus.train --help` and `python -m ligandchorus.infer --help`.

## Structural inputs

Both commands accept preprocessed structural records as NumPy `.npz` files or in the original E18 PyTorch data shards. Records contain the following arrays.

| Key | Shape | Meaning |
| --- | --- | --- |
| `X` | `[L, 4, 3]` | Backbone coordinates in Å, in N, Cα, C, O order |
| `S` | `[L]` | Native sequence tokens used for training |
| `residue_mask` | `[L]` | Mask for valid backbone positions |
| `chain_labels` | `[L]` | Integer chain identifiers |
| `Y` | `[A, 3]` | Context atom coordinates in Å |
| `Y_t` | `[A]` | Context atom atomic numbers |

Token IDs 0–19 follow `ACDEFGHIKLMNPQRSTVWY`. Token 20 denotes a mask or an unknown residue. During inference, `S` provides only the record shape and may contain placeholder tokens.

## Use the existing server data

The original E18 manifest has JSON Lines entries with `key`, `split`, `shard`, and `index`. Keep the manifest and `shards/` under the same directory, or pass `--shard-root` with the directory that contains `shards/`. The commands read the existing shards directly.

Point the shell variables at your server data directory and the two local checkpoint files.

```sh
DATA_ROOT=/path/to/data
LIGANDMPNN_WEIGHTS=/path/to/ligandmpnn_v_32_010_25.pt
E18_WEIGHTS=/path/to/struct_aware_10kstruct.pt
STRUCTURE_KEY=1a28_1
```

Run the trained E18 checkpoint on a structure selected by its exact manifest key.

```sh
python -m ligandchorus.infer --manifest "$DATA_ROOT/manifest_strict.jsonl" \
  --key "$STRUCTURE_KEY" --weights "$E18_WEIGHTS" --output samples.npy
```

To retrain, initialize from the official LigandMPNN checkpoint. The training command keeps the manifest order, selects the first 10,000 assemblies with `split=train` by default, and trains the decoder for 25,000 optimizer steps.

```sh
python -m ligandchorus.train --manifest "$DATA_ROOT/manifest_strict.jsonl" \
  --encoder-weights "$LIGANDMPNN_WEIGHTS" --output e18_retrained.pt
```

The server workflow was checked with PyTorch 2.5.1 and NumPy 1.23.5. Use data shards from your trusted project storage.

## Use standalone NPZ inputs

The training manifest is JSON Lines with one `{"path": "structures/example.npz"}` entry per training assembly. Relative paths are resolved against the manifest's directory. Supply the official `ligandmpnn_v_32_010_25.pt` checkpoint to initialize the frozen encoder. An inference run takes the resulting full E18 state dictionary.

```sh
python -m ligandchorus.train --manifest train.jsonl --encoder-weights ligandmpnn_v_32_010_25.pt --output e18.pt
python -m ligandchorus.infer --input structure.npz --weights e18.pt --output samples.npy
```

The default inference settings generate ten sequences with up to 20 parallel passes. The sampler uses temperature 0.29 for positions whose Cα is within 5 Å of a context atom and 0.24 elsewhere. It retains previously sampled residues and stops once no mask tokens remain. The output contains the sampled token IDs at the original backbone positions. Checkpoints and structural datasets are supplied separately.

## Upstream

The vendored encoder and graph layers derive from [LigandMPNN](https://github.com/dauparas/LigandMPNN). Its MIT license and copyright notice are included in [LIGANDMPNN_LICENSE](ligandchorus/_vendor/LIGANDMPNN_LICENSE).
