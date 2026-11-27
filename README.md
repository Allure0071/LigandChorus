# LigandChorus

LigandChorus is the E18 research implementation of structure-aware parallel decoding for ligand-conditioned fixed-backbone enzyme sequence design. It freezes the LigandMPNN encoder and trains a three-layer graph decoder to recover masked residues. Inference starts from an all-mask sequence, reuses the structural encoding, and applies region-specific sampling temperatures.

The [2026-11-27 reproducibility package](reproduction/2026-11-27/README.md) records the enzyme-trained E18 checkpoint, its validation-calibrated inference settings, and the completed four-model comparison on 80 enzyme and 60 nonenzyme structures. It includes replay commands, server asset paths and checksums, and the aggregate and per-structure reports.

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

### Validate during training

Prepare a fixed JSON Lines panel with a structural manifest `key`, a numeric `chain_label`, and the zero-based `pocket_positions` in that record. Each key must refer to an explicit `split=valid` record. Use precomputed pocket labels for the chosen ligand and provide only valid, known positions on the target chain. Example panel row:

```json
{"key": "example_1", "chain_label": 0, "pocket_positions": [12, 15, 18]}
```

```sh
python -m ligandchorus.train --manifest "$DATA_ROOT/manifest_strict.jsonl" \
  --encoder-weights "$LIGANDMPNN_WEIGHTS" --validation-manifest validation.jsonl \
  --validate-every 1000 --output last.pt
```

Validation runs at each interval and at the final optimizer step. It reports target-chain masked cross-entropy and structure-level medians for target-chain recovery, pocket recovery, and mean pairwise pocket diversity. It generates ten sequences per structure with the default E18 sampling settings. `--validation-seed` fixes the corruption masks and sampling across checkpoints. The run saves the earliest checkpoint attaining the highest validation pocket recovery to `best.pt` and saves the final weights to `--output`. Both files contain full state dictionaries usable by `ligandchorus.infer`.

### Supervise selected enzyme chains

Use `--train-targets train_targets.jsonl` to select explicitly annotated `split=train` records before applying `--structures`. The file uses structural record keys and numeric `chain_labels` from those records. Prepare it from verified chain annotations. Example row:

```json
{"key": "train_example_1", "chain_labels": [0, 2]}
```

The model retains the complete assembly geometry. Only valid, known residues on the listed chains contribute to the masked training loss. Native tokens outside the target mask are hidden during masked training and validation. The option can be combined with `--validation-manifest`.

## Use standalone NPZ inputs

The training manifest is JSON Lines with one `{"path": "structures/example.npz"}` entry per training assembly. Relative paths are resolved against the manifest's directory. Supply the official `ligandmpnn_v_32_010_25.pt` checkpoint to initialize the frozen encoder. An inference run takes the resulting full E18 state dictionary.

```sh
python -m ligandchorus.train --manifest train.jsonl --encoder-weights ligandmpnn_v_32_010_25.pt --output e18.pt
python -m ligandchorus.infer --input structure.npz --weights e18.pt --output samples.npy
```

The default inference settings generate ten sequences with up to 20 parallel passes. The sampler uses temperature 0.29 for positions whose Cα is within 5 Å of a context atom and 0.24 elsewhere. It retains previously sampled residues and stops once no mask tokens remain. The output contains the sampled token IDs at the original backbone positions. Checkpoints and structural datasets are supplied separately.

## Compare fixed-backbone sequence designs

Evaluate E18 against LigandMPNN, LASErMPNN, and ADFLIP on the general non-enzyme ligand panel and the enzyme panel. Give each model the same backbone, ligand geometry, target residue mapping, and number of draws per case. Hide the native target identities and side chains from every model. Use native side-chain heavy atoms only during evaluation to identify pocket positions within 5 Å of the specified ligand.

| Model | Checkpoint and inference setting |
| --- | --- |
| E18 | Use the checkpoint selected on the enzyme validation panel. |
| [LigandMPNN](https://github.com/dauparas/LigandMPNN) | Use the published `ligandmpnn_v_32_010_25.pt` checkpoint with ligand context. |
| [LASErMPNN](https://github.com/polizzilab/LASErMPNN) | Use the published `laser_weights_0p1A_noise_ligandmpnn_split.pt` checkpoint and record ligand preprocessing. |
| [ADFLIP](https://github.com/ykiiiiii/ADFLIP) | Use the published `ADFLIP_v1.pt` checkpoint in single-structure mode. Record the sampling schedule and any step or confidence threshold. |

Use the official external checkpoints without fine-tuning. Log and set each model's sampling parameters on validation or development examples before testing. Match the draw count across models and record speed, hardware, and inference steps when comparing computational cost.

Start with the [317 official LigandMPNN small-molecule test entries](https://github.com/dauparas/LigandMPNN/blob/main/training/test_small_molecule.json). Before freezing the panels, classify each eligible case by the ligand-contacting target chain and check exact entries and homologs against E18 training and validation and the three baseline checkpoints' published references. Reserve test families before finalizing training and record which references were checked. Report the separate, newer temporal holdout independently.

The scorer reads one reference JSON Lines file and one normalized prediction file per model, with one row per case. Reference rows contain `key`, `residue_ids`, `native_sequence`, and zero-based `pocket_positions`. Prediction rows contain `key`, the same ordered `residue_ids`, and sampled `sequences`. Normalize raw outputs to the target-chain residue order and exclude any input or reference sequence from the sample list. Preserve input files, residue maps, software revisions, checkpoint hashes, settings, and seeds alongside the predictions. Repeated draws count toward the sampling budget and contribute zero distance to identical pairs.

```sh
python -m ligandchorus.benchmark --references references.jsonl \
  --e18 e18.jsonl --ligandmpnn ligandmpnn.jsonl --lasermpnn lasermpnn.jsonl \
  --adflip adflip.jsonl \
  --samples 10 --output results.json
```

The report gives target-chain recovery, pocket recovery, and mean pairwise pocket sequence distance for each case, along with means and medians across cases. Every model must have a complete set of aligned predictions for the panel. Use `--rfd4-proteina` only in a separate supplementary run with its own overlap audit and output file. The historical `benchmark_outputs` converter remains available for reproducing earlier results.

## Upstream

The vendored encoder and graph layers derive from [LigandMPNN](https://github.com/dauparas/LigandMPNN). Its MIT license and copyright notice are included in [LIGANDMPNN_LICENSE](ligandchorus/_vendor/LIGANDMPNN_LICENSE).
