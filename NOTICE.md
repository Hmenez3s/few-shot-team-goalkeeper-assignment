# Third-party material

The MIT license in `LICENSE` covers the code in this repository. It does not cover the material below.

## SoccerNet Game State Reconstruction 2025

- Source: <https://huggingface.co/datasets/SoccerNet/SN-GSR-2025>, revision `ce84ee7d9acb2f9fe999ceed2b38e303ffa5efb9`
- License declared on the dataset card: GPL-3.0
- Used for: training, validation and test crops, and every frame shown in `media/`

The dataset is not redistributed here; `download.sh` fetches it from the official source. Images derived from it remain under its GPL-3.0 terms: the GIF, video and images in `media/`, the crops in `dataset-audit/` and the figures in `paper/figures/`. The checkpoints in `models/` were trained on it. The numbers in `results/` were measured on it.

## ResNet-18 ImageNet weights

- Source: torchvision, <https://download.pytorch.org/models/resnet18-f37072fd.pth>
- License: BSD-3-Clause (torchvision)
- Used for: the starting point of the fine-tuned encoder and the frozen baseline

The checkpoints in `models/` were initialised from these weights and trained on the dataset above.

## PRTReID baseline weights

- Source: <https://zenodo.org/records/10653453> (`prtreid-soccernet-baseline.pth.tar`) and <https://zenodo.org/records/10604211> (HRNet-W32)
- Used for: the SoccerNet baseline comparison in `experiments/`; the weights are downloaded separately and not redistributed here

## SoccerTrack

- Source: <https://www.kaggle.com/datasets/atomscott/soccertrack> (version 6, `wide_view`), project <https://github.com/AtomScott/SoccerTrack>
- License declared on the Kaggle dataset: LGPL-3.0
- Used for: the transfer test in `experiments/transfer_datasets.py` and `experiments/transfer_baselines.py`

The dataset is not redistributed here; `transfer_datasets.py --download` fetches the clips from Kaggle. The numbers in `results/transfer-soccertrack.json` were measured on it.

## OSNet and SigLIP/CLIP weights

- OSNet: Market-1501 weights from torchreid (MIT), used frozen as a comparison encoder
- SigLIP (`google/siglip-base-patch16-224`) and CLIP (`openai/clip-vit-base-patch32`) from Hugging Face, used frozen as comparison encoders; revisions are recorded in `results/`
- None of these weights are redistributed here
