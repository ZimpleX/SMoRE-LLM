""" PyTorch LLaMA-MoE model."""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Literal, Dict

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaModel,
    LlamaPreTrainedModel,
)
from transformers.utils import ModelOutput, logging
from transformers.activations import ACT2FN

from .moe.configuration_llama_moe import LlamaMoEConfig
from .moe.moe_layers import LinearGLUMoELayer, MoEMlpOutput
from .model_utils.norm import WeightNorm
from .sft_constants import PRETRAINED_MODEL_ZOO
import re
from .model_utils import ParamStatsSMoRE
from transformers.cache_utils import Cache, DynamicCache

logger = logging.get_logger(__name__)


@dataclass
class MoEArchArguments:
    pretrained_ffn_connection: Literal["residual", "none"] = field(default="residual")
    regex_params2train: Dict[str, List[str]] = field(
        default_factory=lambda: {"moe": ["layers\.(\d+)\.mlp\.gate\.(.+)", "layers\.(\d+)\.mlp\.calculator\.(.+)"]}, 
        metadata={"description": "regex pattern for trainable params"}
    )
    base_model: Optional[str] = field(
        default=None, metadata={"description": "choose from sft_constants.py:PRETRAINED_MODEL_ZOO"}
    )
    residual_up_proj_dim: Optional[int] = field(
        default=None, metadata={"description": "e.g., for Llama2, residual_up_proj_dim=11008"}
    )

    def __post_init__(self):
        if self.base_model is not None:
            assert self.base_model in PRETRAINED_MODEL_ZOO, f"unknown {self.base_model=} for MoE arch"
            self.residual_up_proj_dim = PRETRAINED_MODEL_ZOO[self.base_model].up_proj_dim
        if self.pretrained_ffn_connection == "residual":
            assert self.residual_up_proj_dim is not None, "must specify up proj dim for residual PT FFN"


@dataclass
class BaseMoEModelOutputWithPast(ModelOutput):
    """
    Args:
        num_dropped_tokens: layer idx to the number of dropped tokens
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    balance_loss: Optional[float] = None
    num_dropped_tokens: Optional[Tuple[torch.Tensor]] = None
    gate_load: Optional[Tuple[list]] = None
    gate_importance: Optional[Tuple[list]] = None


@dataclass
class MoECausalLMOutputWithPast(CausalLMOutputWithPast):
    balance_loss: Optional[float] = None
    num_dropped_tokens: Optional[Tuple[int]] = None
    gate_load: Optional[Tuple[list[torch.Tensor]]] = None
    gate_importance: Optional[Tuple[list[torch.Tensor]]] = None


class LlamaMoEMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # NOTE this init may not be called if doing class casting in PEFT
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]
    
    def _decode_mlp_output(self, ret, balance_loss, num_dropped_tokens, gate_load, gate_importance):
        if isinstance(ret, torch.Tensor):
            x_ret = ret
        else:
            assert isinstance(ret, MoEMlpOutput)
            x_ret = ret.hidden_states
            if ret.balance_loss is not None:
                balance_loss += ret.balance_loss
            gate_load += ret.gate_load
            gate_importance += ret.gate_importance
            if ret.num_dropped_tokens < 0:
                num_dropped_tokens = ret.num_dropped_tokens
            elif num_dropped_tokens < 0:
                pass
            else:
                num_dropped_tokens += ret.num_dropped_tokens
        return {
            "hidden_states": x_ret, 
            "balance_loss": balance_loss, 
            "num_dropped_tokens": num_dropped_tokens,
            "gate_load": gate_load,
            "gate_importance": gate_importance,
        }

    def forward(self, x, attention_mask_flat=None):
        """
        self.up_proj: peft.tuners.hier_moe.layer.SMoREFFN
            or peft.tuners.hydralora.layer.HydraLoraLayer
        """
        ret_up = self.up_proj(x, attention_mask_flat=attention_mask_flat)
        ret_up_dec = self._decode_mlp_output(
            ret=ret_up, 
            balance_loss=0,
            num_dropped_tokens=0,
            gate_load=[],
            gate_importance=[],
        )
        ret_gate = self.gate_proj(x, attention_mask_flat=attention_mask_flat)
        ret_gate_dec = self._decode_mlp_output(
            ret=ret_gate, 
            balance_loss=ret_up_dec["balance_loss"],
            num_dropped_tokens=ret_up_dec["num_dropped_tokens"],
            gate_load=ret_up_dec["gate_load"],
            gate_importance=ret_up_dec["gate_importance"],
        )

        ret_down = self.down_proj(
            self.act_fn(ret_gate_dec["hidden_states"]) * ret_up_dec["hidden_states"],
            attention_mask_flat=attention_mask_flat
        )
        ret_down_dec = self._decode_mlp_output(
            ret=ret_down, 
            balance_loss=ret_gate_dec["balance_loss"],
            num_dropped_tokens=ret_gate_dec["num_dropped_tokens"],
            gate_load=ret_gate_dec["gate_load"],
            gate_importance=ret_gate_dec["gate_importance"],
        )
        return MoEMlpOutput(**ret_down_dec)


class LlamaMoEDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaMoEConfig, layer_index):
        super().__init__(config, layer_index)  # layer_index
        self.layer_index = layer_index

        gating_config = {
            # all gates
            "gate_type": config.gate_type,
            "gate_network": config.gate_network,
            "gate_use_softmax": config.gate_use_softmax,
            "gate_use_balance": config.gate_use_balance,
            "gate_balance_loss_weight": config.gate_balance_loss_weight,
            "gate_add_noise": config.gate_add_noise,
            # TopKBalancedNoisyGate
            "gate_noise_epsilon": config.gate_noise_epsilon,
        }
        calculator_config = {
            # all calculators
            "calculator_type": config.calculator_type,
            "multiply_gate_scores": config.multiply_gate_scores,
            "score_scale_factor": (
                config.score_scale_factor[layer_index]
                if isinstance(config.score_scale_factor, list)
                else config.score_scale_factor
            ),
            "add_weight_norm": config.add_weight_norm,
            # SwitchDropTokenCalculator
            "drop_tokens": config.drop_tokens,
            "dropped_padding": config.dropped_padding,
            "capacity_factor": config.capacity_factor,
        }
        self.moe_arch_config = MoEArchArguments(**config.moe_arch_config)
        self.mlp = LinearGLUMoELayer(
            input_size=self.hidden_size,
            hidden_size=config.intermediate_size,
            output_size=self.hidden_size,
            hidden_act=config.hidden_act,
            num_experts=config.num_experts,
            num_selects=config.num_selects,
            size_experts=(
                config.size_experts[layer_index]
                if config.size_experts is not None
                else None
            ),
            bias=False,
            moe_arch_config=self.moe_arch_config,
            layer_index=layer_index,
            **gating_config,
            **calculator_config,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
        attention_mask_flat=None,
    ) -> tuple:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        assert position_embeddings is not None, "Pls set position emb in LlamaModel"
        attention_ret = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        if len(attention_ret) == 2:
            hidden_states, self_attn_weights = attention_ret
            assert not use_cache, "don't know how to set present_key_value"
            present_key_value = None
        elif len(attention_ret) == 3:
            hidden_states, self_attn_weights, present_key_value = attention_ret
        else:
            raise Exception(f"self attention returned {len(attention_ret)} values. Don't know how to proc it")
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_outs: MoEMlpOutput = self.mlp(hidden_states, attention_mask_flat=attention_mask_flat)
        hidden_states = residual + mlp_outs.hidden_states

        outputs = (
            hidden_states,
            mlp_outs.balance_loss,
            mlp_outs.num_dropped_tokens,
            mlp_outs.gate_load,
            mlp_outs.gate_importance,
        )
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def reset_gate_network(self):
        self.mlp.reset_gate_network()

    def reset_experts(self):
        self.mlp.reset_experts()


class LlamaMoEPreTrainedModel(LlamaPreTrainedModel):
    config_class = LlamaMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True  # added
    _no_split_modules = ["LlamaMoEDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, WeightNorm):
            module.reset_parameters()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaMoEModel):
            module.gradient_checkpointing = value


class LlamaMoEModel(LlamaModel, LlamaMoEPreTrainedModel):
    def __init__(self, config: LlamaMoEConfig):
        super().__init__(config)
        self.moe_arch_config = MoEArchArguments(**config.moe_arch_config)
        self.layers = nn.ModuleList(
            [LlamaMoEDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.freeze_pretrained_weights()
        self.post_init()
    
    def freeze_pretrained_weights(self):
        pat2train = [i for sub in self.moe_arch_config.regex_params2train.values() for i in sub]
        for k, w in self.named_parameters():
            w.requires_grad = False
            for pat in pat2train:
                if bool(re.match(pat, k)):
                    w.requires_grad = True
                    break
        size2train = sum([v.numel() for k, v in self.named_parameters() if v.requires_grad])
        size2freeze = sum([v.numel() for k, v in self.named_parameters() if not v.requires_grad])
        logger.warning_once(f"{size2train/1e9:.2f}G params to train")
        logger.warning_once(f"{size2freeze/1e9:.2f}G params to freeze")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position: Optional[torch.LongTensor] = None,
        # **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # embed positions
        assert attention_mask is not None
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds
        balance_loss = 0.0

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        num_dropped_tokens = ()
        gate_load = ()
        gate_importance = ()
        if self.training:
            assert past_key_values is None, "NEED TO DOUBLE-CHECK PAST_KEY_VALUES & CACHE DURING TRAINING"
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:

                layer_outputs: tuple = torch.utils.checkpoint.checkpoint(
                    decoder_layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    attention_mask,
                )
            else:
                layer_outputs: tuple = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    attention_mask_flat=attention_mask,
                )

            hidden_states = layer_outputs[0]
            if layer_outputs[1] is not None:
                balance_loss += layer_outputs[1]

            if use_cache:
                next_decoder_cache = layer_outputs[6 if output_attentions else 5]
            if output_attentions:
                all_self_attns += (layer_outputs[5],)

            num_dropped_tokens += (layer_outputs[2],)
            gate_load += (layer_outputs[3],)
            gate_importance += (layer_outputs[4],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        if isinstance(balance_loss, float):
            assert balance_loss == 0.0
            balance_loss = None
        return BaseMoEModelOutputWithPast(
            last_hidden_state=hidden_states,
            balance_loss=balance_loss,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            num_dropped_tokens=num_dropped_tokens,
            gate_load=gate_load,
            gate_importance=gate_importance,
        )

    def update_config(self):
        self.config.vocab_size = self.config.vocab_size
        self.config.max_position_embeddings = self.config.max_position_embeddings
        # ↓↓↓↓↓↓↓↓↓↓↓↓ changed here ↓↓↓↓↓↓↓↓↓↓↓↓ #
        self.config.hidden_size = self.layers[0].mlp.input_size
        self.config.intermediate_size = self.layers[0].mlp.hidden_size
        self.config.num_hidden_layers = len(self.layers)
        self.config.num_attention_heads = self.layers[0].self_attn.num_heads
        self.config.hidden_act = self.layers[0].mlp.hidden_act
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ #
        self.config.initializer_range = self.config.initializer_range
        self.config.rms_norm_eps = self.config.rms_norm_eps
        self.config.pretraining_tp = self.config.pretraining_tp
        self.config.use_cache = self.config.use_cache
        self.config.rope_scaling = self.config.rope_scaling
        self.config._rope_scaling_validation()

        self.config.num_experts = self.layers[0].mlp.num_experts
        self.config.num_selects = self.layers[0].mlp.num_selects
        self.config.size_experts = [
            self.layers[i].mlp.calculator.experts.size_experts
            for i in range(self.config.num_hidden_layers)
        ]

        self.config.gate_type = vars(self.layers[0].mlp).get(
            "gate_type", "TopKBalancedNoisyGate"
        )
        self.config.gate_network = vars(self.layers[0].mlp.gate).get(
            "gate_network_type", "mlp"
        )
        self.config.gate_use_softmax = vars(self.layers[0].mlp.gate).get(
            "use_softmax", True
        )
        self.config.gate_use_balance = vars(self.layers[0].mlp.gate).get(
            "use_balance", True
        )
        self.config.gate_balance_loss_weight = vars(self.layers[0].mlp.gate).get(
            "balance_loss_weight", 1e-2
        )
        self.config.gate_add_noise = vars(self.layers[0].mlp.gate).get(
            "add_noise", True
        )
        self.config.gate_noise_epsilon = vars(self.layers[0].mlp.gate).get(
            "noise_epsilon", 1e-2
        )

        self.config.calculator_type = vars(self.layers[0].mlp).get(
            "calculator_type", "UniversalCalculator"
        )
        self.config.multiply_gate_scores = vars(self.layers[0].mlp.calculator).get(
            "multiply_gate_scores", True
        )
        self.config.score_scale_factor = [
            vars(self.layers[i].mlp.calculator).get("score_scale_factor", 1.0)
            for i in range(self.config.num_hidden_layers)
        ]
        self.config.drop_tokens = vars(self.layers[0].mlp.calculator).get(
            "drop_tokens", True
        )
        self.config.dropped_padding = vars(self.layers[0].mlp.calculator).get(
            "dropped_padding", "zero"
        )
        self.config.capacity_factor = vars(self.layers[0].mlp.calculator).get(
            "capacity_factor", 1.25
        )

    def set_moe_num_selects(self, num_selects):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_num_selects(num_selects)

    def set_moe_gate_use_softmax(self, use_softmax):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_gate_use_softmax(use_softmax)

    def set_moe_gate_use_balance(self, use_balance):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_gate_use_balance(use_balance)

    def set_moe_gate_balance_loss_weight(self, balance_loss_weight):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_gate_balance_loss_weight(balance_loss_weight)

    def set_moe_gate_add_noise(self, add_noise):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_gate_add_noise(add_noise)

    def set_moe_gate_noise_epsilon(self, noise_epsilon):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_gate_noise_epsilon(noise_epsilon)

    def set_moe_calculator_multiply_gate_scores(self, multiply_gate_scores):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_calculator_multiply_gate_scores(multiply_gate_scores)

    def set_moe_calculator_score_scale_factor(
        self, score_scale_factor, layer_index=None
    ):
        if layer_index is None:
            for idx, decoder_layer in enumerate(self.layers):
                decoder_layer.set_moe_calculator_score_scale_factor(score_scale_factor)
        else:
            self.layers[layer_index].set_moe_calculator_score_scale_factor(
                score_scale_factor
            )

    def set_moe_calculator_drop_tokens(self, drop_tokens):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_calculator_drop_tokens(drop_tokens)

    def set_moe_calculator_dropped_padding(self, dropped_padding):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_calculator_dropped_padding(dropped_padding)

    def set_moe_calculator_capacity_factor(self, capacity_factor):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_calculator_capacity_factor(capacity_factor)

    def reset_gate_network(self):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.reset_gate_network()

    def reset_experts(self):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.reset_experts()


class LlamaForCausalLM_(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.param_stats = ParamStatsSMoRE()


class LlamaMoEForCausalLM(LlamaForCausalLM, LlamaMoEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaMoEModel(config)
        self.param_stats = ParamStatsSMoRE()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position: Optional[torch.LongTensor] = None,
        **loss_kwargs,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseMoEModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        # hidden_states: [8, 53, 4096]
        hidden_states = outputs.last_hidden_state
        # self.lm_head has weight of shape [4096, 32000]
        # logits: [8, 53, 32000]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            if "num_items_in_batch" in loss_kwargs:
                loss = nn.functional.cross_entropy(
                    shift_logits, shift_labels, ignore_index=-100, reduction="sum"
                ) / loss_kwargs["num_items_in_batch"]
            else:
                loss = nn.functional.cross_entropy(
                    shift_logits, shift_labels, ignore_index=-100, reduction="mean"
                )
            if outputs.balance_loss is not None and outputs.balance_loss > 0:
                loss += outputs.balance_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return MoECausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            num_dropped_tokens=outputs.num_dropped_tokens,
            balance_loss=outputs.balance_loss,
            gate_load=outputs.gate_load,
            gate_importance=outputs.gate_importance,
        )

    def update_config(self):
        self.model.update_config()

    def set_moe_num_selects(self, num_selects):
        self.model.set_moe_num_selects(num_selects)

    def set_moe_gate_use_softmax(self, use_softmax):
        self.model.set_moe_gate_use_softmax(use_softmax)

    def set_moe_gate_use_balance(self, use_balance):
        self.model.set_moe_gate_use_balance(use_balance)

    def set_moe_gate_balance_loss_weight(self, balance_loss_weight):
        self.model.set_moe_gate_balance_loss_weight(balance_loss_weight)

    def set_moe_gate_add_noise(self, add_noise):
        self.model.set_moe_gate_add_noise(add_noise)

    def set_moe_gate_noise_epsilon(self, noise_epsilon):
        self.model.set_moe_gate_noise_epsilon(noise_epsilon)

    def set_moe_calculator_multiply_gate_scores(self, multiply_gate_scores):
        self.model.set_moe_calculator_multiply_gate_scores(multiply_gate_scores)

    def set_moe_calculator_score_scale_factor(
        self, score_scale_factor, layer_index=None
    ):
        self.model.set_moe_calculator_score_scale_factor(
            score_scale_factor, layer_index=layer_index
        )

    def set_moe_calculator_drop_tokens(self, drop_tokens):
        self.model.set_moe_calculator_drop_tokens(drop_tokens)

    def set_moe_calculator_dropped_padding(self, dropped_padding):
        self.model.set_moe_calculator_dropped_padding(dropped_padding)

    def set_moe_calculator_capacity_factor(self, capacity_factor):
        self.model.set_moe_calculator_capacity_factor(capacity_factor)

    def reset_gate_network(self):
        self.model.reset_gate_network()

    def reset_experts(self):
        self.model.reset_experts()
