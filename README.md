# LongContextReranker
Official Implementation of LOCORE: Image Re-ranking with Long-Context Sequence Modeling [CVPR 2025]

Zilin Xiao, Pavel Suma, Ayush Sachdeva, Hao-Jen Wang, Giorgos Kordopatis-Zilos, Giorgos Tolias, Vicente Ordonez

[[arxiv]()] [[project page]()]

## Getting Started

Steps marked with `[*]` are optional and can be skipped if you want to use the pretrained model directly.

### Environment Setup

We recommend to create a fresh conda environment with the following command:

```bashc
conda create -n locore python=3.9
conda activate locore
pip install -r requirements.txt
```

### [*] Download Longformer Pretrained Weights
You can download the pretrained weights from here:
- longformer-base-5120: [Google Drive]()
- longformer-tiny-5120: [Google Drive]()

These weights are adapted from the original Longformer weights using positional encoding interpolation with the method described [here](https://github.com/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb).

### [*] Prepare Training Dataset

### [*] Extract Features for Training

### [*] Train LOCORE

### Download LOCORE Checkpoints
You can download the pretrained LOCORE checkpoints from here if you skipped the training step:
- LOCORE-small: [Google Drive]()
- LOCORE-base: [Google Drive]()

### Prepare Testing Dataset

### Prepare Features for Testing

### Run LOCORE
