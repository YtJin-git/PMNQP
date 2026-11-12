# PMNQP
## Primitive Memory Networks with Qualified Prompts for Compositional Zero-Shot Learning

Yutang Jin, Boyu Zhao, Shitao Chen, Shiming Chen, Guo-Sen Xie, Yisong Wang, and Xinge You

**Published in IEEE Transactions on Image Processing (TIP), vol. 35, pp. 11628–11641, 2026.**  
**Date of publication:** 3 November 2026.

[Paper](https://ieeexplore.ieee.org/document/11773295) | [DOI](https://doi.org/10.1109/TIP.2026.3748354) | [Citation](#citation)

## Overview

PMNQP addresses **compositional zero-shot learning (CZSL)**, where a model recognizes unseen attribute–object compositions using knowledge learned from seen compositions.

Built on CLIP, PMNQP combines two complementary sources of information:

- **Qualified prompts:** attribute and object descriptions provide additional textual context. The descriptions are generated offline and included in `qualified_words/`; no LLM access is required during training or inference.
- **Primitive memory networks:** attribute and object memory banks provide visual evidence for recognizing primitive concepts and their compositions.

The implementation contains compositional, attribute, and object branches, together with qualified-prompt and primitive-memory readouts. Dataset-specific settings are provided for MIT-States, UT-Zappos, and C-GQA.

## Results

Please refer to the [published paper](https://doi.org/10.1109/TIP.2026.3748354) for the quantitative comparisons, ablation studies, and analysis. The evaluation code reports seen accuracy, unseen accuracy, best harmonic mean (HM), and area under the seen–unseen accuracy curve (AUC).

## Usage

### Setup

Clone the repository and run commands from its root so that relative paths such as `qualified_words/` resolve correctly:

```bash
git clone https://github.com/YtJin-git/PMNQP.git
cd PMNQP
```

Install compatible versions of **PyTorch** and **torchvision** for your Python/CUDA environment using the [official PyTorch instructions](https://pytorch.org/get-started/locally/). The source also imports the following third-party packages:

```bash
pip install numpy scipy Pillow PyYAML tqdm matplotlib ftfy regex
```

This repository does not currently provide a version-pinned environment file or pretrained PMNQP checkpoints. The package list above is derived from source imports, not a validated reproducibility environment. See [implementation notes](#implementation-notes) before running experiments.

The CLIP loader accepts a supported model name (for example, `ViT-L/14`) or a local CLIP checkpoint path through `--clip_arch`. A model name downloads the corresponding CLIP weights on first use. Keep the selected architecture consistent with `clip_model` in the YAML configuration.

### Dataset preparation

Prepare MIT-States, UT-Zappos, or C-GQA with the **compositional-split-natural** split expected by `dataset.py`. Each dataset root must contain:

```text
<dataset_root>/
├── images/
├── metadata_compositional-split-natural.t7
└── compositional-split-natural/
    ├── train_pairs.txt
    ├── val_pairs.txt
    └── test_pairs.txt
```

The image paths referenced by the metadata must resolve beneath `images/`.

`tools/download_datasets.sh` only downloads the original MIT-States ZIP archive. It does **not** extract the archive, provide the compositional split metadata, or prepare UT-Zappos/C-GQA. The complete dataset layout above is required before training.

### Configurations

| Dataset | Configuration | Dataset identifier | Qualified descriptions |
| --- | --- | --- | --- |
| MIT-States | `config/mit-states.yml` | `mit-states` | `qualified_words/mit-states.json` |
| UT-Zappos | `config/ut-zappos.yml` | `ut-zap50k` | `qualified_words/ut-zap50k.json` |
| C-GQA | `config/cgqa.yml` | `cgqa` | `qualified_words/cgqa.json` |

The supplied configurations use CLIP ViT-L/14 and closed-world evaluation (`open_world: False`). Edit the YAML file to change training hyperparameters, GPU index (`cuda_device`), or evaluation settings.

**Configuration precedence:** YAML values are loaded after command-line arguments and override arguments with the same name. The example commands below use `dataset_path`, `save_path`, `clip_arch`, and `load_model`, which are not active entries in the supplied YAML files. If you add these fields to a YAML file, update their values there.

### Training

The training entry point is `train.py`. After preparing the data and addressing the implementation notes below, the MIT-States invocation is:

```bash
python train.py \
  --yml_path config/mit-states.yml \
  --dataset_path /path/to/mit-states \
  --clip_arch "ViT-L/14" \
  --save_path logs/pmnqp/mit-states
```

For UT-Zappos or C-GQA, replace the configuration, dataset root, and output directory with the matching values from the table above.

Training evaluates validation and test splits and writes checkpoints under `--save_path`. The code includes periodic `epoch_<index>.pt` checkpoints (zero-based epoch indices) and `val_best.pt`, as well as configuration exports after training. See the checkpoint-selection caveat below before interpreting `val_best.pt`.

### Testing

The standalone evaluation entry point is `test.py`. Once its dataset-initialization issue described below is corrected, the corresponding closed-world invocation is:

```bash
python test.py \
  --yml_path config/mit-states.yml \
  --dataset_path /path/to/mit-states \
  --clip_arch "ViT-L/14" \
  --load_model logs/pmnqp/mit-states/val_best.pt
```

Use a checkpoint trained with the matching dataset and model configuration. No pretrained PMNQP checkpoint is currently included.

For open-world evaluation, change `open_world` in the YAML file rather than relying on a command-line override. The evaluator uses `data/feasibility_<dataset>.pt` and validation-based threshold selection; ensure the feasibility scores match the dataset's pair ordering.

### Implementation notes

The commands above document the repository's entry points; they are not a claim that the current release has passed end-to-end execution. Source inspection identifies the following issues that need attention before reproduction:

- **Qualified-description initialization:** `models/PMNQP.py` calls `qualified_attr_list.qualified(...)` and `qualified_obj_list.qualified(...)`, but both objects are Python lists, which have no `qualified` method. This blocks model initialization.
- **Standalone evaluation:** `test.py` calls `get_model(...)` without a dataset, while `PMNQP.__init__` requires one and accesses its attributes immediately.
- **UT-Zappos open-world evaluation:** the dataset identifier is `ut-zap50k`, while the supplied feasibility file is named `data/feasibility_ut-zappos.pt`. The evaluator's expected filename and supplied filename need to be aligned, with pair ordering checked.
- **Checkpoint selection:** `train.py` shares its best-metric variables between validation and test evaluation. Consequently, `val_best.pt` should not be interpreted as a checkpoint selected solely by validation performance without correcting this logic.

## Repository structure

```text
PMNQP/
├── clip_modules/          # CLIP implementation and tokenizer
├── config/                # Dataset-specific configurations
├── data/                  # Open-world feasibility scores
├── models/
│   ├── PMNQP.py           # PMNQP model
│   ├── common.py          # Shared model components
│   └── model_factory.py   # Model construction
├── qualified_words/       # Offline attribute/object descriptions
├── tools/                 # Optimization and dataset utilities
├── dataset.py             # Dataset loading and splits
├── parameters.py          # Command-line arguments
├── train.py               # Training entry point
├── test.py                # Evaluation entry point
└── utils.py               # Configuration and utility functions
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{jin2026pmnqp,
  title={Primitive Memory Networks with Qualified Prompts for Compositional Zero-Shot Learning},
  author={Jin, Yutang and Zhao, Boyu and Chen, Shitao and Chen, Shiming and Xie, Guo-Sen and Wang, Yisong and You, Xinge},
  journal={IEEE Transactions on Image Processing},
  volume={35},
  pages={11628--11641},
  year={2026},
  doi={10.1109/TIP.2026.3748354},
  url={https://ieeexplore.ieee.org/document/11773295}
}
```
