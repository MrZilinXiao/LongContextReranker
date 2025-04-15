# a regular ring transformer with hf call signature as follows:
# input_ids, (not needed, just for compability)
# attention_mask=attention_mask,  (needed, 1 -> attended, 0 -> not attended)
# global_attention_mask=global_attention_mask,  (not needed, as ring-attention is global by default)
# head_mask=head_mask,   (None by default)
# token_type_ids=token_type_ids,   (might be needed to inject token_type in future)
# position_ids=position_ids,    (needed to inject absolute position embeddings)
# inputs_embeds=inputs_embeds,    (needed)
# output_attentions=output_attentions, (not needed, just for compability)
# output_hidden_states=output_hidden_states,  (not needed, just for compability)
# return_dict=return_dict,  (not needed, just for compability)
# so that it can replace `LongformerModelWithoutEmbedding` in extractive_longformer_multispan.py

# 1) we first try to train from scratch and compare with longformer-base
# 2) try to migrate bert or longformer to ring-attention

from ring_attention_pytorch.ring_attention import (
    RingAttention, RingRotaryEmbedding, FeedForward, RMSNorm, 
    # helper function
    default, maybe_pad_seq_and_mask, divisible_by, cast_tuple, exists, 
    sharded_batch_to_sharded_seq, sharded_seq_to_sharded_batch
)

from ring_attention_pytorch.ring import (
    is_distributed,
    get_world_size
)


from typing import Optional, Tuple, Union, Dict, List

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import Module, ModuleList

from einops import rearrange, repeat

from beartype import beartype
from math import ceil
from dataclasses import dataclass

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput

def build_ring_transformer_extractive(
    seq_len=5120,
    num_buckets=2,
    num_sharded_batches=1,
    dim=768,
    heads=12,
    depth=6,
    num_grouped_query_heads=1,   # whether to split heads into groups to enable GQA
    dim_head=64,
    causal=False,
    pos_type='absolute',
    loss_on_gathered_seq=False,
):
    # hf trainer will ensure distributed env setup.
    world_size = get_world_size()
    ring_seq_size = ceil(seq_len / world_size) * num_sharded_batches
    bucket_size = ring_seq_size // num_buckets
    
    model = RingTransformerForExtractive(
        dim=dim,
        causal=causal, 
        depth=depth,
        heads=heads,
        num_grouped_query_heads=num_grouped_query_heads,
        dim_head=dim_head,
        ring_attn=True,
        striped_ring_attn=False, 
        ring_seq_size=ring_seq_size,
        bucket_size=bucket_size,
        pos_type=pos_type,
        loss_on_gathered_seq=loss_on_gathered_seq,
    )
    
    return model

@dataclass
class RingTransformerForExtractiveOutput(ModelOutput):
    loss: Optional[Tensor]
    # additional losses are for stats only; in trainer and future dist training, they need 
    probs: Optional[torch.Tensor] = None
    indices: Optional[torch.Tensor] = None
    unpack_log: Optional[Dict[str, float]] = None
    

class DummyLongformer(Module):
    """
    A dummy longformer model to meet the compatibility of `RingTransformerForExtractive` in extractive setting.
    """
    def __init__(self, dim=768):
        super().__init__()
        self.sep_embedding = nn.Parameter(
            torch.randn(dim)
        )
        
class MyRingAttention(RingAttention):
    def __init__(self, *args, **kwargs):
        dim = kwargs.get('dim', 768)
        self.dim = dim
        super().__init__(*args, **kwargs)
        norm_cls = nn.LayerNorm if self.norm_type == 'layer' else RMSNorm
        # replace RMSNorm with LayerNorm if possible
        for name, module in self.named_children():
            if isinstance(module, RMSNorm):
                del module
                setattr(self, name, norm_cls(self.dim))
    

class RingTransformerForExtractive(Module):
    """
    Modification to RingTransformer:
    1) no token embedding, input_ids will be asserted to None.
    2) logic adapted to take inputs_embeds directly (N, L, D input).
    3) absolute position embedding is added as an option.
    """
    @beartype
    def __init__(
        self,
        *,
        # num_tokens: int,
        dim: int,
        depth: int,
        causal: bool = False,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        num_grouped_query_heads: int = 1,   # grouped query attention - kv heads = (heads // num_grouped_query_heads)
        bucket_size: int = 512,
        ring_attn: bool = False,
        striped_ring_attn: bool = False,  # not enabled for encoder-only transformer
        ring_seq_size: int = 512,
        auto_shard_seq: Optional[bool] = None,
        max_lookback_seq_len: Optional[Union[Tuple[int, ...], int]] = None,   # None -> full look-back
        rotary_embed_theta: int = 10000,    # will need to be changed for the million token context
        ignore_index: int = -100,
        force_regular_attn: bool = False,
        use_cuda_kernel: Optional[bool] = None, 
        # 05/02: added for extractive setting upon RingTransformer
        pos_type: Optional[str] = 'rotary',   # choose from 'rotary' or 'absolute';
        seq_len: Optional[int] = 5120, 
        # we can afford a positional embedding for the extractive setting within 5k tokens
        loss_on_gathered_seq: bool = False,  # whether to calculate loss on gathered sequence
    ):
        super().__init__()
        
        # need a config to be compatible
        self.config = PretrainedConfig(hidden_size=dim, num_attention_heads=heads, num_hidden_layers=depth)
        # dummy longformer model for sep_embedding
        self.longformer = DummyLongformer(dim=dim)

        use_cuda_kernel = default(use_cuda_kernel, torch.cuda.is_available())
        self.use_cuda_kernel = use_cuda_kernel
        assert not (use_cuda_kernel and not torch.cuda.is_available())

        self.ring_attn = ring_attn
        self.striped_ring_attn = striped_ring_attn

        self.using_striped_ring_cuda = use_cuda_kernel and striped_ring_attn

        self.ring_seq_size = ring_seq_size
        self.bucket_size = bucket_size

        assert (not ring_attn) or divisible_by(ring_seq_size, bucket_size), f'ring seq size {ring_seq_size} is not divisible by bucket size {bucket_size}'

        self.auto_shard_seq = default(auto_shard_seq, ring_attn) # if ring attention is turned on, auto-shard across sequence dimension. this can also be turned off and done manually elsewhere in the data loading

        assert not (not self.ring_attn and self.auto_shard_seq)
        assert not (not self.ring_attn and self.striped_ring_attn)
        # no striped ring attention in our setting. 
        assert not (self.striped_ring_attn and not causal), 'striped ring attention only applies to autoregressive models'

        # self.token_emb = nn.Embedding(num_tokens, dim)
        
        self.pos_type = pos_type
        if self.pos_type == 'rotary':
            self.rotary_emb = RingRotaryEmbedding(
                dim = dim_head,
                ring = ring_attn,
                striped = striped_ring_attn,
                theta = rotary_embed_theta,
                buckets = ring_seq_size // bucket_size
            )
        elif self.pos_type == 'absolute':
            # need a position_ids to activate this
            self.pos_emb = nn.Embedding(seq_len + 2, dim)  # seq_len is only used here. 
        else: 
            raise ValueError(f"pos_type {self.pos_type} is not supported.")
        
        self.layers = ModuleList([])

        max_lookback_seq_len = cast_tuple(max_lookback_seq_len, depth)
        assert len(max_lookback_seq_len) == depth

        for layer_max_lookback_seq_len in max_lookback_seq_len:

            self.layers.append(ModuleList([
                RingAttention(
                    dim = dim,
                    causal = causal,
                    dim_head = dim_head,
                    heads = heads,
                    num_grouped_query_heads = num_grouped_query_heads,
                    bucket_size = bucket_size,
                    ring_attn = ring_attn,
                    ring_seq_size = ring_seq_size,
                    max_lookback_seq_len = layer_max_lookback_seq_len,
                    striped_ring_attn = striped_ring_attn,
                    force_regular_attn = force_regular_attn,
                    use_cuda_kernel = self.use_cuda_kernel,
                    auto_shard_seq = False,
                ),
                FeedForward(dim = dim, mult = ff_mult)  # 4 times the dimension, 768 -> 3072
            ]))
            
        # do not need `to_logits`. this is offloaded to `LongformerForExtrativeRetrieval`. 
        # need raw last-layer hidden states
        self.to_logits = RMSNorm(dim)
        # self.to_logits = nn.Sequential(
        #     RMSNorm(dim),
        #     nn.Linear(dim, num_tokens, bias = False)
        # )
        self.padding_idx = 1   # align with roberta. 0, 1 is reserved. position_ids starts from 2.
        
        # align with `LongformerForExtrativeRetrieval`
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(dim, 2)
        self.num_local_features = 50  
        self.recall_ks = [1, 5, 10]
        
        self.ignore_index = ignore_index
        self.loss_on_gathered_seq = loss_on_gathered_seq

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor inputs_embeds:

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
    
    @torch.inference_mode()
    def batch_predict(
        self, 
        logits, # [N, L, 2]; 2 for Inside/Outside
        num_local_features: int, 
        select_modes: List[str], 
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:  # return a list of ranked indices
        logits = torch.softmax(logits, dim=-1)
        in_logits = logits[..., 1]   # [N, L]
        N, L = in_logits.shape
        in_logits = in_logits.view(N, -1, num_local_features + 1)  # [N, num_imgs, 50]
        # strip SEP representations & query img
        in_logits = in_logits[:, 1:, :-1]  # [N, num_gallery, num_local_features]
        probs_dict = dict()
        indices_dict = dict()
        for mode in select_modes:
            # all probs are in [N, num_gallery]
            if mode == "max-prod":
                probs = in_logits.prod(dim=-1)   # this may suffers from precision issue.
            elif mode == "max-sum":
                probs = in_logits.sum(dim=-1)
            elif mode == "max-start":
                probs = in_logits[..., 0]
            elif mode == "max-end":
                probs = in_logits[..., -1]
            else:
                raise ValueError(f"Invalid mode {mode}")
            _, indices = torch.sort(probs, dim=-1, descending=True)
            probs_dict[mode] = probs
            indices_dict[mode] = indices
        
        return probs_dict, indices_dict
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        positive_mask: Optional[torch.Tensor] = None,  # [N, num_gallery_imgs]. very needed!
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # below for training-time evaluation
        is_eval: Optional[bool] = False,
    ):
        assert input_ids is None, "input_ids is not needed for Extractive setting."
        assert inputs_embeds is not None, "inputs_embeds is needed for Extractive setting."  # N, L, D
        
        # need to prepare labels in advance, 
        # as best practice of ringattention is `forward_chunk` to enforce CE loss on each device
        
        labels = None
        if positive_mask is not None and not is_eval:
            query_padded = torch.zeros(positive_mask.size(0), 1, dtype=torch.long, device=positive_mask.device)
            # query_padded = -100 * torch.ones(positive_mask.size(0), 1, dtype=torch.long, device=positive_mask.device)
            token_positive_mask = torch.cat([query_padded, positive_mask], dim=1)  # [N, (1 + num_gallery_imgs)]
            token_positive_mask = token_positive_mask[:, :, None].expand(-1, -1, self.num_local_features + 1).flatten(1)  # [N, (1 + num_gallery_imgs) * 50]
            labels = token_positive_mask
        
        if not self.loss_on_gathered_seq:
            loss, logits = self.forward_chunk(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                is_eval=is_eval
            )  # logits is already in N, L, 2
            
        else:  # do not provide labels. only use logits to do training here
            _, logits = self.forward_chunk(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=None,
                is_eval=is_eval
            )
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            loss = loss_fct(logits.reshape(-1, 2), labels.reshape(-1))
                        
        
        eval_modes = ["max-prod", "max-sum", "max-start", "max-end"]
        
        probs_dict, indices_dict = self.batch_predict(logits,
                                            num_local_features=self.num_local_features, 
                                            select_modes=eval_modes)  
        recalls = dict()
        for k in self.recall_ks:
            for mode in eval_modes:
                selected_k_indices = indices_dict[mode][:, :k]  # [N, k]
                selected_k_positive_mask = positive_mask.gather(1, selected_k_indices)  # [N, k]
                # have it still in torch Tensor so that we can reduce it.
                recalls[f"{mode}_recall_{k}"] = selected_k_positive_mask.any(dim=-1).mean(dtype=torch.float)
                
        if is_eval:
            recalls.update({
                "probs": probs_dict,
                "indices": indices_dict,
            })
        
        return RingTransformerForExtractiveOutput(
            loss=loss,
            unpack_log=recalls,   # in eval mode, use `unpack_log` to get in-batch results
        )
    
    # forward will call `forward_chunk` to enforce CE loss on each device
    # before gathering. This is in line of motivation of RingAttention.
    def forward_chunk(
        self, 
        inputs_embeds=None,    # (needed)  N, L, D
        input_ids=None, # (not needed, just for compability)
        attention_mask=None,  # (needed, 1 -> attended, 0 -> not attended)  N, L
        global_attention_mask=None,  # (not needed, as ring-attention is global by default)
        head_mask=None,   # (None by default)
        token_type_ids=None,   # (might be needed to inject token_type in future)
        position_ids=None,    # (needed if needed inject absolute position embeddings)
        output_attentions=None, # (not needed, just for compability)
        output_hidden_states=None,  # (not needed, just for compability)
        return_dict=None,  # (not needed, just for compability)
        # for ring attention_compability
        force_ring_reduce_off = False,
        ring_size = None, 
        # self,
        # x,
        # mask = None,
        labels = None,
        is_eval = False,
        # force_ring_reduce_off = False,
        # ring_size = None
    ):
        
        seq_len, device = inputs_embeds.shape[1], inputs_embeds.device
        
        attention_mask = default(attention_mask, torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.bool))
        attention_mask = attention_mask.bool()

        auto_shard_seq = not force_ring_reduce_off and self.auto_shard_seq and is_distributed()

        using_striped_ring_cuda = inputs_embeds.is_cuda and self.using_striped_ring_cuda
        striped_bucket_size = self.bucket_size if not using_striped_ring_cuda else self.ring_seq_size
        
        # take care of padding to divide sequence across the machines

        ring_size = default(ring_size, get_world_size())  # default to number of GPUs
        
        # absolute positions need to be inserted before auto_sharding
        if self.pos_type == 'absolute':
            position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
            pos_emb = self.pos_emb(position_ids)
            inputs_embeds = inputs_embeds + pos_emb

        if auto_shard_seq:  # used in extractive setting
            # first pad to right multiple
            
            inputs_embeds, attention_mask = maybe_pad_seq_and_mask(inputs_embeds, attention_mask, self.ring_seq_size)

            # labels

            if exists(labels):
                labels, label_mask = maybe_pad_seq_and_mask(labels, attention_mask, self.ring_seq_size)
                labels.masked_fill_(~label_mask, self.ignore_index)
            
            assert not self.striped_ring_attn, "striped_ring_attn is not supported in our setting."

            # if self.striped_ring_attn:
            #     x = rearrange(x, 'b (i j) -> b (j i)', i = striped_bucket_size)

            #     if exists(labels):
            #         labels = rearrange(labels, 'b (i j) -> b (j i)', i = striped_bucket_size)

            #     if exists(mask):
            #         mask = rearrange(mask, 'b (i j) -> b (j i)', i = striped_bucket_size)

            # gather across batch and divide across world

            (inputs_embeds, attention_mask), batch_sizes, num_sharded_batches = sharded_batch_to_sharded_seq(
                inputs_embeds, attention_mask, self.ring_seq_size
            )
            
            if exists(labels):
                (labels, _), *_ = sharded_batch_to_sharded_seq(labels, None, self.ring_seq_size)

            # calculate ring size from num sharded batches

            ring_size = get_world_size() // num_sharded_batches

        # rotary positions
        # taking into account ring and striping

        # rotary_emb = self.rotary_emb(x.shape[-1])
        rotary_emb = None
        if self.pos_type == 'rotary':
            rotary_emb = self.rotary_emb(inputs_embeds.shape[1])
            
            # usually there is a layernorm+dropout after this. It's safe to insert dropout here; 
            # but dropout in ringattention needs to be handled separately.

        # main transformer logic
        # no need for embedding layer; inputs_embeds is already in N, L, D shape
        # x = self.token_emb(x)
        
        # TODO: add grad_checkpointing here
        for attn, ff in self.layers:
            inputs_embeds = attn(
                inputs_embeds,
                mask = attention_mask,
                rotary_emb = rotary_emb,   # None -> no_pos applied for rotary
                force_ring_reduce_off = force_ring_reduce_off,
                ring_size = ring_size
            ) + inputs_embeds

            inputs_embeds = ff(inputs_embeds) + inputs_embeds

        logits = self.to_logits(inputs_embeds)  # a RMSNorm layer alone. (N, L, D)
        
        # above ended longformer inside logic. now apply token_classification logic
        
        logits = self.dropout(logits)
        logits = self.classifier(logits)
        
        loss = None
        
        if exists(labels):
            cls_logits = rearrange(logits, 'b n c -> b c n')
            loss = F.cross_entropy(
                cls_logits, labels, 
                ignore_index=self.ignore_index
            )
            # loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            # loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            
            # if return_loss:
            # logits = rearrange(logits, 'b n c -> b c n')

            # ce_loss = F.cross_entropy(
            #     logits,
            #     labels,
            #     ignore_index = self.ignore_index
            # )

            # return ce_loss
            
        # below we gather the sequence logits for in-training eval.
        # otherwise gather all sequence chunks for logits across machines and shard the batch dimension

        if not auto_shard_seq:
            return logits

        logits = sharded_seq_to_sharded_batch(logits, batch_sizes, num_sharded_batches)

        # if self.striped_ring_attn:
        #     logits = rearrange(logits, 'b (j i) d -> b (i j) d', i = striped_bucket_size)

        return loss, logits[:, :seq_len]  # N, L, 2
