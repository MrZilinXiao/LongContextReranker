# 11/03 convert to HDF5 dataset loading
from common_utils.gld_data_utils_hdf5 import build_extractive_hdf5_data_modules
from common_utils.trainer_utils import ExtractiveTrainer
from models.longformer_universal_v2 import ExtractiveLongformerForCache
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict
import transformers
from transformers import AutoConfig
import torch
import torch.nn as nn
import pathlib
import os
from safetensors.torch import load_model
import numpy as np

local_rank = None

def rank0_print(*args):  # debug usage only
    if local_rank in [-1, 0]:
        print(*args)
        
def test_nonzero_features(desc):
    print(desc.shape)
    subset = np.sort(np.random.choice(len(desc), 1000, replace=False))
    desc_subset = desc[subset]
    norms = np.linalg.norm(desc_subset[..., 5:], axis=-1)
    failed_ids = np.where((norms == 0) * (1 - desc_subset[..., 3]))[0]
    if len(failed_ids):
        print(f"amount failed: {len(failed_ids)}")
    else:
        print("test_nonzero_features: OK")
    return failed_ids


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


@dataclass
class DataArguments:
    local_hdf5_path: str = field(default="/scratch/zx51/ames/my_data/gldv2-train/dinov2_local.hdf5",)
    nn_ids_path: str = field(default="/scratch/zx51/ames/data/gldv2/nn_dinov2.pkl",)
    sample_txt_path: str = field(default="/scratch/zx51/ames/data/gldv2/train_750k.txt",)
    # added 04/11 for local and global dim
    local_dim: int = field(default=768, metadata={"help": "Local feature dimension"})

    is_training: bool = field(default=True, metadata={"help": "Is training or evaluation"})
    topk: int = field(default=100, metadata={"help": "Number of nearest neighbors for reranking training"})
    min_pos_per_topk: int = field(default=0, metadata={"help": "Minimum number of positive samples per nearest neighbor"})
    max_pos_per_topk: int = field(default=100, metadata={"help": "Number of positive samples per nearest neighbor"})
    shuffle_pos: bool = field(default=True, metadata={"help": "Shuffle samples in candidates"})
    num_descriptors: int = field(default=49, metadata={"help": "Number of top local descriptors to use for training"})
    # 11/04: add optional global feature path. not providing it will disable global features
    global_pt_path: Optional[str] = field(default=None, metadata={"help": "Path to global features"})
    global_dim: int = field(default=768, metadata={"help": "Global feature dimension"})
    shuffle_indices: bool = field(default=False, metadata={"help": "Shuffle indices for training data"})
    num_training_samples: int = field(default=None, metadata={"help": "Number of training samples"})
    # 25/02/05: add bottleneck dim to fairly compare with AMES, 
    # this does not change dataloading, but only model init with a bottleneck layer
    bottleneck_dim: int = field(default=None, metadata={"help": "Bottleneck dimension for the model"})

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None)  # for torch.load checkpoint loading; usually used for unfrozen finetune
    language_model_name: str = field(default="allenai/longformer-base-4096")
    # model_version: str = field(default="v1")   # any model can get different versions
    # since 03/25, language_model_name can be `videomamba_small.safetensors` with use_pretrained=True
    use_pretrained: bool = field(default=False, metadata={"help": "Use pretrained weights for the language model"})
    # 25/02/05: use pretrained positional embedding only
    use_pretrained_pos_only: bool = field(default=False, metadata={"help": "Use pretrained positional embedding only"})
    # below for overriding the language model default config
    num_hidden_layers: int = field(default=6, metadata={"help": "Number of hidden layers"})
    attention_window: int = field(default=512, metadata={"help": "Attention window size"})
    num_attention_heads: int = field(default=4, metadata={"help": "Number of attention heads"})
    max_position_embeddings: int = field(default=5122, metadata={"help": "Maximum position embeddings"})
    # change hidden size can reduce the final model size and insert intermediate projection layers in between
    hidden_size: int = field(default=768, metadata={"help": "Hidden size of Transformer encoder"})
    intermediate_size: int = field(default=3072, metadata={"help": "Intermediate FFN size of Transformer encoder"})
    query_global_attention: bool = field(default=False, metadata={"help": "Use global attention for query"})
    pos_type: str = field(default='absolute', metadata={"help": "Type of position embedding"})
    # on 04/23: add optional L2 norm
    do_norm: bool = field(default=False, metadata={"help": "Whether to L2 normalize features"})
    # dedicated params for recurrent model
    query_attn_layer_ids: List[int] = field(default=None, metadata={"help": "Layer ids for query attention of recurrent models. None for not enabled."})
    # 05/12: num_layers_kept added for utilizing previous layers from pre-trained models
    num_layers_kept: int = field(default=None, metadata={"help": "Number of layers kept from pre-trained models"})
    # 25/02/12: force adding linear even when dimension matches
    force_linear: bool = field(default=False, metadata={"help": "Force adding linear layer even when dimension matches"})
    # 25/02/12: frozen longformer first as a stage-1
    frozen_longformer: bool = field(default=False, metadata={"help": "Freeze longformer layers for stage-1 training"})
    # resume_weight: Optional[str] = field(default=None, metadata={"help": "Path to resume weight"})
    linear_activation: str = field(default=None, metadata={"help": "Activation function for linear layer"})
        
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch_fused")
    remove_unused_columns: bool = field(default=False)   # DO NOT remove unused columns
    lr_decay_epochs: List[int] = field(default=None)  # works only when lr_scheduler_type is `constant`
    lr_decay_ratio: float = field(default=0.1)   # at each decay epoch, lr = lr * lr_decay_ratio
    
    # special seed for reproducibility: added 04/03; turns out not needed: worker seed should always follows the process it forks from.
    base_seed: int = field(default=None, metadata={"help": "Base seed for permutation generation"})

    def __post_init__(self):
        super().__post_init__()
        if self.report_to == "wandb":
            # prepend mmddyyyy format datetime before the run_name
            self.run_name = datetime.now().strftime("%m%d%Y") + "_" + self.run_name
        # assert lr_decay_epochs not exceeding num_train_epochs
        if self.lr_decay_epochs is not None:
            assert max(self.lr_decay_epochs) <= self.num_train_epochs, "lr_decay_epochs should not exceed num_train_epochs"
        rank0_print(f"lr_decay_epochs: {self.lr_decay_epochs}")


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    # compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    training_args.base_seed = training_args.base_seed * 1000 + local_rank * 1000 if training_args.base_seed is not None else None
    # this will be handled by ExtractiveTrainer.get_train_dataloader()
    
    if not model_args.use_pretrained:
        # init model here from scratch / load from checkpoint
        if 'mamba' in model_args.language_model_name.lower() or 'rwkv' in model_args.language_model_name.lower():
            config = None
        else:
            config = AutoConfig.from_pretrained(model_args.language_model_name)
            if 'longformer' in model_args.language_model_name:
                config.num_hidden_layers = model_args.num_hidden_layers
                config.attention_window = [model_args.attention_window] * config.num_hidden_layers
                config.num_attention_heads = model_args.num_attention_heads
                config.hidden_size = model_args.hidden_size
                config.intermediate_size = model_args.intermediate_size
                config.max_position_embeddings = model_args.max_position_embeddings
                # config.sep_only = model_args.sep_only
        # mamba model does not need config for now
            
    else:
        # assert model_args.model_name_or_path is None, f"Cannot set both use_pretrained and model_name_or_path to True"
        rank0_print(f"Will load pretrained weights from {model_args.language_model_name}... Command line args will be ignored.")
        config = model_args.language_model_name   # use str config to ask weight loading

    if False:
        pass
    # TODO: watch out! models other than longformer are not tested    
    elif 'longformer' in model_args.language_model_name.lower():
        global_offset = 1 if data_args.global_pt_path is not None else 0
        model = ExtractiveLongformerForCache(
            language_model=config,
            local_dim=data_args.local_dim, 
            global_dim=data_args.global_dim, 
            query_global_attention=model_args.query_global_attention, 
            pos_type=model_args.pos_type, 
            do_norm=model_args.do_norm, 
            num_layers_kept=model_args.num_layers_kept, 
            bottleneck_dim=data_args.bottleneck_dim, 
            use_pretrained_pos_only=model_args.use_pretrained_pos_only,
            num_features=data_args.num_descriptors + global_offset,
            force_linear=model_args.force_linear, 
            linear_activation=model_args.linear_activation,
        )
    
    if model_args.model_name_or_path is not None:
        if os.path.exists(os.path.join(model_args.model_name_or_path, 'model.safetensors')):
            load_model(model, os.path.join(model_args.model_name_or_path, 'model.safetensors'))
            rank0_print(f"Loaded {model_args.model_name_or_path}.")
        else:
            ckpt = torch.load(os.path.join(model_args.model_name_or_path, 'pytorch_model.bin'), map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
            rank0_print(f"Loaded {model_args.model_name_or_path}. Missing keys: {missing_keys}; Unexpected keys: {unexpected_keys}")
            assert missing_keys == [] and unexpected_keys == [], "Missing or unexpected keys in the checkpoint."
    
    if model_args.frozen_longformer:
        # trainable parameter keys: ['language_model.classifier.weight', 'language_model.classifier.bias', 'scale_encoder.*', 
        # '*to_lm',]
        # freeze the entire model first
        for param in model.parameters():
            param.requires_grad = False
        # unfreeze trainable parameters
        trainable_params = []
        for name, param in model.named_parameters():
            if 'language_model.classifier' in name or 'scale_encoder' in name or 'to_lm' in name:
                param.requires_grad = True
                trainable_params.append(name)
                
        rank0_print(f"Trainable parameters: {trainable_params}")
        # correct echo: Trainable parameters: ['language_model.classifier.weight', 'language_model.classifier.bias', 'scale_encoder.weight', 'local_to_lm.0.weight', 'local_to_lm.0.bias', 'local_to_lm.1.weight', 'local_to_lm.1.bias', 'global_to_lm.0.weight', 'global_to_lm.0.bias', 'global_to_lm.1.weight', 'global_to_lm.1.bias']
        # Trainable parameters: 398850
            
    # print model parameter statistics
    rank0_print("Model parameter statistics:")
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Total parameters: {num_params}")
    rank0_print(f"Trainable parameters: {trainable_params}")
    
    # move to bf16 / fp16
    # model = model.to(dtype=compute_dtype)
    # rank0_print(f"Model dtype: {compute_dtype}")
    
    data_module = build_extractive_hdf5_data_modules(
        local_hdf5_path=data_args.local_hdf5_path,
        nn_ids_path=data_args.nn_ids_path,
        sample_txt_path=data_args.sample_txt_path,
        is_training=data_args.is_training,
        topk=data_args.topk,
        max_pos_per_topk=data_args.max_pos_per_topk,
        min_pos_per_topk=data_args.min_pos_per_topk,
        shuffle_pos=data_args.shuffle_pos,
        num_descriptors=data_args.num_descriptors,
        global_pt_path=data_args.global_pt_path,
        global_dim=data_args.global_dim,
        num_samples=data_args.num_training_samples,
        shuffle_indices=data_args.shuffle_indices,
    )
    
    # build trainer
    trainer = ExtractiveTrainer(
        model=model,
        args=training_args,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir
    )

if __name__ == "__main__":
    train()