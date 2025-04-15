# 11/03: refactor v2 version to only considers:
# 1. local feature re-ranking (global features)
# 2. image-wise positional encoding
# keep global-related params but prompt for warnings when provided

from transformers.models.longformer.modeling_longformer import (LongformerPreTrainedModel, 
                                                                LongformerModel, 
                                                                LongformerTokenClassifierOutput)
from transformers.utils import logging
from transformers import AutoConfig
import torch.nn as nn
import torch
from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass
from typing import List
from transformers.activations import get_activation


logger = logging.get_logger(__name__)


@dataclass
class LongformerExtractiveOutput(LongformerTokenClassifierOutput):
    # additional losses are for stats only; in trainer and future dist training, they need 
    probs: Optional[torch.Tensor] = None
    indices: Optional[torch.Tensor] = None
    unpack_log: Optional[Dict[str, float]] = None   


class ExtractiveLongformerForCache(nn.Module):
    def __init__(self, 
                 language_model: Union[str, AutoConfig] = None, 
                 local_dim: int = 1024, 
                 global_dim: int = 512, 
                 num_scales: int = 3,   # for scale encoder
                 query_global_attention: bool = True, 
                 pos_type: str = 'absolute', 
                 do_norm: bool = False,
                 num_layers_kept: Optional[int] = None, 
                 bottleneck_dim: Optional[int] = None,
                 use_pretrained_pos_only: bool = False,
                 num_features: Optional[int] = 49, 
                 force_linear: bool = False,
                 linear_activation: Optional[str] = None,   # select from ['relu', 'gelu', 'tanh']
                ) -> None:
        """
        language model: when provided a string, load from hf hub or local
        else init scratch model from hf config
        """
        super(ExtractiveLongformerForCache, self).__init__()
        
        self.query_global_attention = query_global_attention
        if query_global_attention:
            print("query_global_attention enabled for multi-span Longformer")
            
        self.pos_type = pos_type
        self.do_norm = do_norm

        if isinstance(language_model, str):
            self.language_model = LongformerForExtrativeRetrieval.from_pretrained(language_model, num_features=num_features)
            if use_pretrained_pos_only:    # call to init non-embedding parameters
                for name, param in self.language_model.longformer.named_modules():
                    if 'embeddings' not in name:
                        LongformerPreTrainedModel._init_weights(self.language_model, param)
                        print(f"Init {name} with random weights")
                        
        else:
            self.language_model = LongformerForExtrativeRetrieval(language_model, num_features=num_features)
        
        self.num_layers_kept = num_layers_kept
        if num_layers_kept is not None:
            self.language_model.longformer.encoder.layer = self.language_model.longformer.encoder.layer[:num_layers_kept]
            self.language_model.config.attention_window = self.language_model.config.attention_window[:num_layers_kept]
            self.language_model.config.num_hidden_layers = num_layers_kept
            print(f"Only keep {num_layers_kept} layers for Longformer!")
    
        self.padding_idx = self.language_model.longformer.config.pad_token_id
        
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.num_scales = num_scales
        self.scale_encoder = nn.Embedding(num_scales, self.language_model.config.hidden_size)
        
        self.local_to_lm = nn.Identity()
        self.global_to_lm = nn.Identity()
        # 25/02/05: add bottleneck layer to have fair comparison with AMES
        if bottleneck_dim is not None and linear_activation is not None:
            print("bottleneck_dim with activation in between for Longformer")
            self.local_to_lm = nn.Sequential(
                nn.Linear(local_dim, bottleneck_dim),
                get_activation(linear_activation), 
                nn.Linear(bottleneck_dim, self.language_model.config.hidden_size)
            )
            
            self.global_to_lm = nn.Sequential(
                nn.Linear(global_dim, bottleneck_dim),
                get_activation(linear_activation), 
                nn.Linear(bottleneck_dim, self.language_model.config.hidden_size)
            )
            nn.init.zeros_(self.local_to_lm[0].bias)
            nn.init.zeros_(self.global_to_lm[0].bias)
            nn.init.zeros_(self.local_to_lm[2].bias)
            nn.init.zeros_(self.global_to_lm[2].bias)
            print(f"Keep bottleneck layer for Longformer with dim {bottleneck_dim}")
            
        elif bottleneck_dim is not None:
            self.local_to_lm = nn.Sequential(
                nn.Linear(local_dim, bottleneck_dim),
                nn.Linear(bottleneck_dim, self.language_model.config.hidden_size)
            )
            self.global_to_lm = nn.Sequential(
                nn.Linear(global_dim, bottleneck_dim),
                nn.Linear(bottleneck_dim, self.language_model.config.hidden_size)
            )
            nn.init.zeros_(self.local_to_lm[0].bias)
            nn.init.zeros_(self.global_to_lm[0].bias)
            nn.init.zeros_(self.local_to_lm[1].bias)
            nn.init.zeros_(self.global_to_lm[1].bias)
            print(f"Keep bottleneck layer for Longformer with dim {bottleneck_dim}")
        
        elif force_linear:  # 25/02/12: force linear projection even if hidden_size == local_dim / global_dim
            print("Force linear projection for Longformer")
            self.local_to_lm = nn.Linear(local_dim, self.language_model.config.hidden_size)
            nn.init.zeros_(self.local_to_lm.bias)
            
            self.global_to_lm = nn.Linear(global_dim, self.language_model.config.hidden_size)
            nn.init.zeros_(self.global_to_lm.bias)
            
        else:
            if self.language_model.config.hidden_size != local_dim:
                self.local_to_lm = nn.Linear(local_dim, self.language_model.config.hidden_size)
                nn.init.zeros_(self.local_to_lm.bias)
            
            # still keep it for compatibility, only skip global_to_lm in forward func. 
            if self.language_model.config.hidden_size != global_dim:
                self.global_to_lm = nn.Linear(global_dim, self.language_model.config.hidden_size)
                nn.init.zeros_(self.global_to_lm.bias)
            
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # longformer supports gradient checkpointing
        print("Enable gradient checkpointing for Longformer")
        self.language_model.gradient_checkpointing_enable()

    def forward(self, 
                # query_global_features: torch.Tensor,
                query_local_features: torch.Tensor,
                query_local_scales: torch.Tensor,
                query_local_mask: torch.Tensor,
                # gallery_global_features: torch.Tensor,
                gallery_local_features: torch.Tensor,
                gallery_local_scales: torch.Tensor,
                gallery_local_mask: torch.Tensor,
                # fix on 11/03: global features become optional
                query_global_features: torch.Tensor = None,
                gallery_global_features: torch.Tensor = None,
                positive_mask: torch.LongTensor = None, 
                is_training: bool = True, 
                with_softmax: bool = True,
                ):
        """
        query_global_features torch.Size([4, 1, 512])
        query_local_features torch.Size([4, 49, 1024])
        query_local_scales torch.Size([4, 49])
        query_local_mask torch.Size([4, 49])
        gallery_global_features torch.Size([4, 100, 1, 512])
        gallery_local_features torch.Size([4, 100, 49, 1024])
        gallery_local_scales torch.Size([4, 100, 49])
        gallery_local_mask torch.Size([4, 100, 49])
        positive_mask torch.Size([4, 100])
        if global not provided, provide 49 local features; else 48 + 1 global
        then the final sequence would be (49+1) * 101 = 5050
        """
        bs, num_gallery_imgs, num_local_features, local_dim = gallery_local_features.size()
        hidden_size = self.language_model.config.hidden_size
        
        # make it compatible with label 2: toy indicator
        if positive_mask is not None:
            positive_mask = (positive_mask == 1) | (positive_mask == 2)
        
        # 04/10: [global][l0][l1]...[l49][SEP]|[global][l0][l1]...[l49][SEP]|...|[global][l0][l1]...[l49][SEP]
        
        # step0: be careful, local mask from data is 0 for valid tokens, 1 for padding
        query_local_mask = ~query_local_mask.bool()
        gallery_local_mask = ~gallery_local_mask.bool()
        
        # step1: project global & local features to language model hidden_size
        if query_global_features is not None:
            query_global_features = self.global_to_lm(query_global_features)  # [N, 1, hidden_size]
            gallery_global_features = self.global_to_lm(gallery_global_features)  # [N, num_gallery_imgs, 1, hidden_size]
        
        query_local_features = self.local_to_lm(query_local_features)  # [N, 49, hidden_size]
        gallery_local_features = self.local_to_lm(gallery_local_features)  # [N, num_gallery_imgs, 49, hidden_size]
        
        # step1.5: optional normalization
        if self.do_norm:
            if query_global_features is not None:
                query_global_features = torch.nn.functional.normalize(query_global_features, p=2, dim=-1)
                gallery_global_features = torch.nn.functional.normalize(gallery_global_features, p=2, dim=-1)
            
            query_local_features = torch.nn.functional.normalize(query_local_features, p=2, dim=-1)
            gallery_local_features = torch.nn.functional.normalize(gallery_local_features, p=2, dim=-1)
        
        # step2: encode scales to embeddings
        query_local_features = query_local_features + self.scale_encoder(query_local_scales)
        gallery_local_features = gallery_local_features + self.scale_encoder(gallery_local_scales)
        
        # step3: merge global & local features at corresponding locations
        query_local_features = torch.cat(
            [query_global_features, query_local_features], dim=1
        ).unsqueeze(1) if query_global_features is not None else query_local_features.unsqueeze(1)
        # [N, 1, 49, hidden_size], either 48 + 1 or 49
        
        query_local_mask = torch.cat(
            [torch.ones(bs, 1, dtype=torch.bool, device=query_local_mask.device), 
             query_local_mask,
             torch.ones(bs, 1, dtype=torch.bool, device=query_local_mask.device)], dim=1
        ).unsqueeze(1)  if query_global_features is not None else torch.cat(
            [query_local_mask, 
             torch.ones(bs, 1, dtype=torch.bool, device=query_local_mask.device)], dim=1
        ).unsqueeze(1)
        # [N, 1, 50]  for optinoal global & sep tokens
        
        if gallery_global_features is not None:
            gallery_local_features = torch.cat(
                [gallery_global_features, gallery_local_features], dim=2
            )  
        # [N, num_gallery_imgs, 49, hidden_size], either 48 + 1 or 49
        
        gallery_local_mask = torch.cat(
            [
                torch.ones(bs, num_gallery_imgs, 1, dtype=torch.bool, device=gallery_local_mask.device), 
                gallery_local_mask, 
                torch.ones(bs, num_gallery_imgs, 1, dtype=torch.bool, device=gallery_local_mask.device)], dim=2
        ) if gallery_global_features is not None else torch.cat(
            [gallery_local_mask, 
             torch.ones(bs, num_gallery_imgs, 1, dtype=torch.bool, device=gallery_local_mask.device)], dim=2
        )
        # [N, num_gallery_imgs, 50]  for optional global & sep tokens
        
        # step4: prepare sep tokens to interleave sep tokens
        sep_token_for_query = self.language_model.longformer.sep_embedding[None, None, None, :].expand(
           bs, 1, 1, hidden_size
        )  # [N, 1, 1, hidden_size]
        query_local_features = torch.cat([query_local_features, sep_token_for_query], dim=2)  # [N, 1, 51, hidden_size]
        
        # offset should always be 49, 48 + 1 or 49 + 0
        offset = num_local_features + (1 if gallery_global_features is not None else 0)   # all global & local features
        # +1 compensate for global feature token
        gallery_reshaped = gallery_local_features.reshape(bs * num_gallery_imgs, offset, hidden_size)
        sep_token_for_gallery = self.language_model.longformer.sep_embedding[None, None, None, :].expand(
            bs, num_gallery_imgs, 1, hidden_size
        )
        sep_token_reshaped = sep_token_for_gallery.reshape(-1, 1, hidden_size)
        gallery_local_features = torch.cat([gallery_reshaped, sep_token_reshaped], dim=1).reshape(
            bs, num_gallery_imgs, offset + 1, hidden_size
        )  # [N, num_gallery_imgs, 50, hidden_size]
        
        # step5: prepare input_embeds
        inputs_embeds = torch.cat([query_local_features, gallery_local_features], dim=1)
        inputs_embeds = inputs_embeds.reshape(
            bs, (1 + num_gallery_imgs) * (offset + 1), hidden_size
        )
        
        # step6: prepare attention_mask
        # print("query_local_mask shape: ", query_local_mask.size()) # ([8, 50])
        # print("gallery_local_mask shape: ", gallery_local_mask.size()) # ([8, 100, 50])
        attention_mask = torch.cat(
            [query_local_mask, gallery_local_mask], dim=1
        ).flatten(1)  # [N, 1 + num_gallery_imgs, 50] -> [N, (1 + num_gallery_imgs) * 50]
        
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[:, offset::offset + 1] = 1  # sep tokens are global attended
        
        if self.query_global_attention:
            global_attention_mask[:, :offset] = 1
        
        position_ids = None   # by default a full position ids
        
        if self.pos_type == 'no':
            position_ids = 2 * torch.ones(bs, (1 + num_gallery_imgs) * (offset + 1), dtype=torch.long, device=inputs_embeds.device)
            
        elif self.pos_type == 'segment':
            # query -> 0, gallery1 -> 1, gallery2 -> 2, ..., gallery100 -> 100
            position_ids = torch.arange(self.padding_idx + 1, 1 + num_gallery_imgs + self.padding_idx + 1, device=inputs_embeds.device).unsqueeze(0).expand(
                bs, -1
            ).repeat_interleave(offset + 1, dim=1)
            
        elif self.pos_type == 'absolute':   # regular None position_ids
            pass
        
        else:
            raise RuntimeError(f"Invalid pos_type {self.pos_type}")

        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            positive_mask=positive_mask,  # [N, num_gallery_imgs]
            is_eval=not is_training, 
            with_softmax=with_softmax,
            # num_features=offset, 
        )

        return outputs


class LongformerModelWithoutEmbedding(LongformerModel):
    def __init__(self, config, add_pooling_layer=True, keep_word_embeddings=False):
        super().__init__(config, add_pooling_layer)
        # before remove embedding layer, save some special token embeddings
        
        self.sep_embedding = nn.Parameter(
            self.get_input_embeddings().weight[self.config.sep_token_id].detach().clone()
        )
        self.pad_embedding = nn.Parameter(
            self.get_input_embeddings().weight[self.config.pad_token_id].detach().clone()
        )
        
        if not keep_word_embeddings:
            self.embeddings.word_embeddings = None
        # there is a LayerNorm hidden in `self.embedding`

    # override `_pad_to_window_size` to avoid pad_embedding_token invoked
    def _pad_to_window_size(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        pad_token_id: int,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        padding_len = (attention_window - seq_len % attention_window) % attention_window

        # this path should be recorded in the ONNX export, it is fine with padding_len == 0 as well
        if padding_len > 0:
            logger.info(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )
            if input_ids is not None:
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)
            if position_ids is not None:
                # pad with position_id = pad_token_id as in modeling_roberta.RobertaEmbeddings
                position_ids = nn.functional.pad(position_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                # input_ids_padding = inputs_embeds.new_full(
                #     (batch_size, padding_len),
                #     self.config.pad_token_id,
                #     dtype=torch.long,
                # )
                inputs_embeds_padding = self.pad_embedding[None, None, :].expand(batch_size, padding_len, self.config.hidden_size)
                # it's okay if we don't layernorm padding tokens, as they will be ignored by attention anyway
                # inputs_embeds_padding = self.embeddings(input_ids_padding)  # [bsz, padding_len, hidden_size]
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

            attention_mask = nn.functional.pad(
                attention_mask, (0, padding_len), value=0
            )  # no attention on the padding tokens
            token_type_ids = nn.functional.pad(token_type_ids, (0, padding_len), value=0)  # pad with token_type_id = 0

        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds
    
    
# multi-span variation here
class LongformerForExtrativeRetrieval(LongformerPreTrainedModel):
    def __init__(self, config, keep_word_embeddings=False, num_features=49):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.longformer = LongformerModelWithoutEmbedding(config, add_pooling_layer=False, 
                                                          keep_word_embeddings=keep_word_embeddings)
        # num_labels = 2 for start & end logits; however it only supports single extractive answer
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        self.num_local_features = num_features  # FIXME: 48 + 1
        # combining local & global, but exclude [SEP]: on 25/02/06 we use a custom number here
        
        self.post_init()
        
    @torch.inference_mode()
    def batch_predict(
        self, 
        logits, # [N, L, 2]; 2 for Inside/Outside
        num_local_features: int, 
        select_modes: List[str], 
        with_softmax: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:  # return a list of ranked indices
        
        if with_softmax:  # global-local ensemble handles this in eval loop
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
        positive_mask: Optional[torch.Tensor] = None,  # [N, num_gallery_imgs]
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_eval: Optional[bool] = False,
        with_softmax: Optional[bool] = True,  # only useful in batch_predict
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert global_attention_mask is not None, f"global_attention_mask is required for {self.__class__.__name__} model"
        assert input_ids is None, f"input_ids should be None, got {input_ids}"
        
        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        recalls = None

        if positive_mask is not None and not is_eval:
            loss_fct = nn.CrossEntropyLoss()
            # query_padded = torch.zeros(positive_mask.size(0), 1, dtype=torch.long, device=positive_mask.device)
            # update on 04/28: do not enforce loss on query tokens
            query_padded = -100 * torch.ones(positive_mask.size(0), 1, dtype=torch.long, device=positive_mask.device)
            token_positive_mask = torch.cat([query_padded, positive_mask], dim=1)  # [N, (1 + num_gallery_imgs)]
            token_positive_mask = token_positive_mask[:, :, None].expand(-1, -1, self.num_local_features + 1).flatten(1)  # [N, (1 + num_gallery_imgs) * 50]
            # +1 compensate for SEP
            loss = loss_fct(logits.view(-1, 2), token_positive_mask.view(-1))  # shuffling should be enabled
        
        eval_modes = ["max-prod", "max-sum", "max-start", "max-end"]
        if positive_mask is not None:  # in-train recall@k evaluation
            with torch.inference_mode():  # slow training by a bit...
                probs_dict, indices_dict = self.batch_predict(logits,
                                                    num_local_features=self.num_local_features, 
                                                    select_modes=eval_modes)  
                # indices: [N, num_gallery_imgs]; probs: [N, num_gallery_imgs]
                # with a positive mask, compute R@ks with 10 or less positive imgs.
                # num_positive_imgs = positive_mask.sum(dim=-1)  # [N, 1]
                recalls = dict()
                for k in [1, 5, 10]:
                    for mode in eval_modes:
                        selected_k_indices = indices_dict[mode][:, :k]  # [N, k]
                        selected_k_positive_mask = positive_mask.gather(1, selected_k_indices)  # [N, k]
                        # have it still in torch Tensor so that we can reduce it.
                        recalls[f"{mode}_recall_{k}"] = selected_k_positive_mask.any(dim=-1).mean(dtype=torch.float)
        else:  # in-eval, output raw scores for each gallery.
            probs_dict, indices_dict = self.batch_predict(logits,
                                                    num_local_features=self.num_local_features, # CAUTION: fixed param here for now.
                                                    select_modes=eval_modes,
                                                    with_softmax=with_softmax)
            # probs are in [N, num_gallery], indices_dict are in [N, num_gallery]
            return LongformerExtractiveOutput(
                loss=loss,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                global_attentions=outputs.global_attentions,
                unpack_log={
                    'probs': probs_dict,
                    'indices': indices_dict,
                },  
                # without shuffling, indices_dict can be mapped back to top-k ranks in main loop 
            )
            
        return LongformerExtractiveOutput(
            loss=loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
            unpack_log=recalls,   # in eval mode, use `unpack_log` to get in-batch results
        )