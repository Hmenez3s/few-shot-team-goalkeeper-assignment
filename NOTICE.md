# Third-party material

The MIT license in `LICENSE` covers the code in this repository. It does not cover the material below.

## SoccerNet Game State Reconstruction 2025

- Source: <https://huggingface.co/datasets/SoccerNet/SN-GSR-2025>, revision `ce84ee7d9acb2f9fe999ceed2b38e303ffa5efb9`
- License declared on the dataset card: GPL-3.0
- Used for: training, validation and test crops, and every frame shown in `media/`

The dataset is not redistributed here; `download.sh` fetches it from the official source. The images, the GIF and the video in `media/` are derived from its test split and remain under the dataset's terms. The numbers in `results/` were measured on it.

## ResNet-18 ImageNet weights

- Source: torchvision, <https://download.pytorch.org/models/resnet18-f37072fd.pth>
- License: BSD-3-Clause (torchvision)
- Used for: the starting point of the fine-tuned encoder and the frozen baseline

The fine-tuned checkpoint published in the repository's release was initialised from these weights and trained on the dataset above.
