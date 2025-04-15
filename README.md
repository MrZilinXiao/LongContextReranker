# LongContextReranker
Official Implementation of LOCORE: Image Re-ranking with Long-Context Sequence Modeling [CVPR 2025]

[Zilin Xiao](https://zilin.me/), [Pavel Suma](https://scholar.google.com/citations?user=Ci_CMMEAAAAJ&hl=en), [Ayush Sachdeva](https://www.linkedin.com/in/ayushsachdeva1/), [Hao-Jen Wang](https://www.linkedin.com/in/hao-jen-wang/), [Giorgos Kordopatis-Zilos](https://gkordo.github.io/), [Giorgos Tolias](https://cmp.felk.cvut.cz/~toliageo/), [Vicente Ordonez](https://www.cs.rice.edu/~vo9/)

[[arxiv](https://arxiv.org/abs/2503.21772)] [[project page](https://zilin.me/locore/)] [[huggingface demo (TBD)](https://huggingface.co/spaces/vislang/locore)]

## Getting Started

Steps marked with `[*]` are optional and can be skipped if you want to use the pretrained model directly.

### Environment Setup

We recommend to create a fresh conda environment with the following command:

```bashc
conda create -n locore python=3.9
conda activate locore
pip install -r requirements.txt
```

### Prepare Local Features
Our latest model operates on DINOv2 local features. Please download the DINOv2 detector model provided by AMES authors [here](http://ptak.felk.cvut.cz/personal/sumapave/public/ames/networks/dinov2_detector.pt). 

Prepare the local features for all datasets (replace `/scratch/zx51/ames/` with your own path). Follow similar steps for other datasets:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python extract/extract_descriptors.py \
    --detector /scratch/zx51/ames/networks/dinov2_detector.pt \
    --save_path /scratch/zx51/ames/my_data/ \
    --data_path /scratch/zx51/google-landmark \
    --file_name /scratch/zx51/ames/data/gldv2/train_750k.txt \
    --backbone dinov2 \
    --dataset gldv2_train \
    --topk 100 \
    --imsize 770
```

### Prepare TopK Indices Before Reranking

Replace `data_dir` in `extract/prepare_topk_global.py` with your own path. The script will generate the topk indices for each dataset.

```bash
PYTHONPATH=. python extract/prepare_topk_global.py --dataset gldv2-train --desc_name dinov2
PYTHONPATH=. python extract/prepare_topk_global.py --dataset roxford5k+1m --desc_name dinov2
PYTHONPATH=. python extract/prepare_topk_global.py --dataset rparis6k+1m --desc_name dinov2
PYTHONPATH=. python extract/prepare_topk_global.py --dataset roxford5k --desc_name dinov2
PYTHONPATH=. python extract/prepare_topk_global.py --dataset rparis6k --desc_name dinov2
```

### [*] Download Longformer Pretrained Weights

You can download the pretrained weights from here:
- longformer-base-5120: [Research Data Site](https://nas.mrxiao.net/research_data/longformer_base_5120.zip)
- longformer-tiny-5120: [Research Data Site](https://nas.mrxiao.net/research_data/longformer_tiny_5120.zip)

These weights are adapted from the original Longformer weights using positional encoding interpolation with the method described [here](https://github.com/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb).

### [*] Train LOCORE

Run the training with the following command (replace `/workspace/ames/my_data/` with your own path):

```bash
torchrun --nproc-per-node=8 train_ames.py \
    --output_dir ./checkpoints/dinov2_locore_base \
    --use_pretrained True \
    --language_model_name ./longformer-base-5120 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 99 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --lr_scheduler_type constant \
    --logging_steps 10 \
    --dataloader_num_workers 4 \
    --log_on_each_node False \
    --shuffle_pos True \
    --ddp_find_unused_parameters False \
    --query_global_attention True \
    --gradient_checkpointing True \
    --local_dim 768 \
    --global_dim 768 \
    --local_hdf5_path /workspace/ames/my_data/gldv2-train/dinov2_local.hdf5 \
    --nn_ids_path /workspace/ames/my_data/gldv2-train/nn_dinov2.pkl \
    --sample_txt_path /workspace/ames/my_data/gldv2-train/train_750k.txt \
    --global_pt_path /workspace/ames/my_data/gldv2-train/dinov2_global.pt \
    --num_descriptors 48
```

### Download LOCORE Checkpoints

You can download the pretrained LOCORE checkpoints from here if you skipped the training step:
- `locore_base.safetensors`: [Huggingface Models](https://huggingface.co/vislang/locore_dinov2_base) or [Research Data Site](https://nas.mrxiao.net/research_data/locore_base.safetensors)

### Run LOCORE

The following command will run the LOCORE model on the GLDv2 validation set first. The best alpha and temperature will be selected based on the results. The final reranking on ROxf & RPar will be performed with the selected parameters.

```bash 
# grid search on gldv2-val for best alpha and temp
python eval_revisitop.py \
    --checkpoint_path ./checkpoints/dinov2_locore_base \
    --device cuda:0 --gldv2_val

# run sliding window reranking
python eval_revisitop.py \
    --checkpoint_path ./checkpoints/dinov2_locore_base \
    --device cuda:0 --alpha 0.5 --temp 0.1 --with_hard --with_1m --sliding_reranking_type end-to-start
```

## Acknowledgements

This repository is built on top of the following repositories:
- AMES: [https://github.com/pavelsuma/ames](https://github.com/pavelsuma/ames)
- Longformer: [https://github.com/allenai/longformer](https://github.com/allenai/longformer)
- Reranking Transformer: [https://github.com/uvavision/RerankingTransformer](https://github.com/uvavision/RerankingTransformer)

Please consider citing our paper if you find this repository useful for your research:

```bibtex
@article{xiao2025locore,
  title={LOCORE: Image Re-ranking with Long-Context Sequence Modeling},
  author={Xiao, Zilin and Suma, Pavel and Sachdeva, Ayush and Wang, Hao-Jen and Kordopatis-Zilos, Giorgos and Tolias, Giorgos and Ordonez, Vicente},
  journal={arXiv preprint arXiv:2503.21772},
  year={2025}
}
```
