#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

@torch.no_grad()
def sinkhorn_normalized(K: torch.Tensor, n_iters: int=20, eps: float = 1e-12):
    # perform Sinkhorn-Knopp normalization
    for _ in range(n_iters):
        K /= (K.sum(dim=1, keepdim=True) + eps) # row
        K /= (K.sum(dim=0, keepdim=True) + eps) # column
    return K

def sinkhorn_onehot_lsd_shift(
        shift_logits: torch.Tensor, # [B, S, V]
        shift_labels: torch.Tensor, # [B, S], -100ignore
        tau_sd: float = 2.2,
        lam: float = 0.12,
        iters: int = 16,
        max_tokens: int | None = 768,
) -> torch.Tensor:
    # Regularization based on one-hot encoding batch Sinkhorn norm
    B, S, V = shift_logits.shape
    device = shift_logits.device

    valid = (shift_labels != -100)
    if not valid.any():
        return shift_logits.new_zeros(())
    
    # Probability distribution (float 32)
    s = torch.softmax(shift_logits.float() / tau_sd, dim=-1)[valid] # [N, V]
    y = shift_labels[valid].long() # [N]
    N = s.size(0)

    # Sampling boundary
    if max_tokens is not None and N > max_tokens:
        idx = torch.randperm(N, device=device)[:max_tokens]
        s = s.index_select(0, idx)
        y = y.index_select(0, idx)
        N = s.size(0)

    Pji = s.index_select(dim=1, index=y)
    D = 2.0 * (1.0 - Pji).transpose(0, 1)

    K = torch.exp(-D / lam).clamp_min(1e-12)
    P = sinkhorn_normalized(K, n_iters=iters)
    
    # loss
    lsd = (P * D).sum() / (N * N)
    return lsd.to(shift_logits.dtype)

class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        mode = None,
        h_block = None,
        w_block = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        prepared = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, mode, h_block, w_block)
        
        if prepared[0] is None and prepared[3] is None:
            dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return CausalLMOutputWithPast(loss=dummy_loss)

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = prepared

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits_sd = logits[..., :-1, :].contiguous()
            shift_labels_sd = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits_ce = shift_logits_sd.view(-1, self.config.vocab_size)
            shift_labels_ce = shift_labels_sd.view(-1)
            # Enable model/pipeline parallelism
            # Cross-entropy part
            shift_labels_ce = shift_labels_ce.to(shift_logits_ce.device)
            ce = loss_fct(shift_logits_ce, shift_labels_ce)

            # Sinkhorn Norm
            use_lsd = getattr(self.config, "use_sinkhorn_lsd", True)
            beta = getattr(self.config, "sinkhorn_beta", 0.6)
            tau_sd = getattr(self.config, "sinkhorn_tau_sd", 2.0)
            lam = getattr(self.config, "sinkhorn_lambda", 0.1)
            iters = getattr(self.config, "sinkhorn_iters", 20)
            max_tok = getattr(self.config, "sinkhorn_max_tokens", 512)
            alpha_ce = getattr(self.config, "sinkhorn_alpha_ce", 1.0)

            if use_lsd:
                lsd = sinkhorn_onehot_lsd_shift(
                    shift_logits=shift_logits_sd,
                    shift_labels=shift_labels_sd,
                    tau_sd=tau_sd,
                    lam=lam,
                    iters=iters,
                    max_tokens=max_tok,
                )
            else:
                lsd = ce.new_zeros(())

            loss = alpha_ce * ce + beta * lsd

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
