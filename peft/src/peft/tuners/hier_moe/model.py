from __future__ import annotations

import math
import operator
import warnings
from contextlib import contextmanager
from dataclasses import asdict, replace
from enum import Enum
from functools import partial, reduce
from typing import Literal, Optional

import torch
from torch import nn
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaForCausalLM
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM

from .layer import SMoREFFN

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
    onload_layer,
    replicate_layers,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_peft_model_state_dict,
    get_quantization_config,
)
from peft.utils.merge_utils import dare_linear, dare_ties, magnitude_prune, task_arithmetic, ties
from peft.utils.other import get_pattern_key

# from .aqlm import dispatch_aqlm
# from .awq import dispatch_awq
# from .eetq import dispatch_eetq
# from .gptq import dispatch_gptq
# from .hqq import dispatch_hqq
# from .layer import Conv2d, LoraLayer, dispatch_default
# from .torchao import dispatch_torchao
# from .tp_layer import dispatch_megatron

from model_moe import (
    LlamaMoEForCausalLM, LlamaMoEModel, LlamaMoEDecoderLayer, LlamaMoEMLP,
    Gemma2MoEForCausalLM, Gemma2MoEModel, Gemma2MoEDecoderLayer, Gemma2MoEMLP,
)
import re


class HieRMoEModel(BaseTuner):
    """
    Wrapper outside of LlamaMoEForCausalLM
    """
    prefix: str = "moe_"

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
        # NOTE cast self.model to LlamaMoEForCausalLM; self.model.model to LlamaMoEModel
        if isinstance(self.model, LlamaForCausalLM):
            class_info = {
                "causal": LlamaMoEForCausalLM,
                "model": LlamaMoEModel,
                "layer": LlamaMoEDecoderLayer,
                "mlp": LlamaMoEMLP,
            }
        elif isinstance(self.model, Gemma2ForCausalLM):
            class_info = {
                "causal": Gemma2MoEForCausalLM,
                "model": Gemma2MoEModel,
                "layer": Gemma2MoEDecoderLayer,
                "mlp": Gemma2MoEMLP,
            }
        else:
            raise NotImplementedError

        self.model.__class__ = class_info["causal"]
        self.model.model.__class__ = class_info["model"]
        for decoder_layer in self.model.model.layers:
            decoder_layer.__class__ = class_info["layer"]
            moe_example_module = "moe_gate"
            if hasattr(decoder_layer.mlp, "base_layer"):
                pass
            elif (
                (
                    hasattr(decoder_layer.mlp, "gate_proj") 
                    and hasattr(decoder_layer.mlp.gate_proj, moe_example_module)
                ) or (
                    hasattr(decoder_layer.mlp, "up_proj") 
                    and hasattr(decoder_layer.mlp.up_proj, moe_example_module)
                ) or (
                    hasattr(decoder_layer.mlp, "down_proj") 
                    and hasattr(decoder_layer.mlp.down_proj, moe_example_module)
                )
            ):
                decoder_layer.mlp.__class__ = class_info["mlp"]
        
    @staticmethod
    def _check_target_module_exists(moe_config, key: str) -> bool:
        return check_target_module_exists(moe_config, key)
    
    @staticmethod
    def _create_new_module(moe_config, adapter_name, target, layer_index=-1, **kwargs):
        # dispatchers = []
        moe_arch = moe_config.moe_arch
        # NOTE currently just mimicing "dispatch_default"
        if moe_arch == "smore":
            kwargs = {
                "r": moe_config.expert_dims,
                "s": moe_config.num_experts,
                "f": moe_config.num_active,
                "d_g": moe_config.gate_dims,
                "lora_alpha": moe_config.lora_alpha,
                "dim_downproj_x": moe_config.dim_downproj_x,
                "gate_act_fn": moe_config.gate_act_fn,
                "expert_act_fn": moe_config.expert_act_fn,
                "init_method": moe_config.init_method,
                "gate_type": moe_config.gate_type,
                "gate_arch": moe_config.gate_arch,
                # "dim_hid_gate_mlp": moe_config
                "layer_forward_mode": moe_config.layer_forward_mode,
                "balance_loss_weight": moe_config.balance_loss_weight,
                #
                "is_shared_expert_pool": moe_config.is_shared_expert_pool,
                "activation_position": moe_config.activation_position,
            }
            if moe_config.gate_arch.startswith("mixlora"):
                smore_cls = MixLoRAFFN
            else:
                smore_cls = SMoREFFN
            new_module = smore_cls(target, adapter_name=adapter_name, **kwargs)
        else:
            raise NotImplementedError(f"not supporting '{moe_arch}' MoE arch")
        return new_module


    def _create_and_replace(
        self,
        moe_config, #PeftConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        """
        Called in BaseTuner: __init__() ==> inject_adapter()
        """
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        kwargs = {}
        layer_index = -1
        if isinstance(target, LlamaMLP):
            layer_index_search = re.search(r'\.layers\.(\d+)\.', current_key)
            if layer_index_search is not None:
                layer_index = int(layer_index_search.group(1))
        new_module = self._create_new_module(moe_config, adapter_name, target, layer_index=layer_index, **kwargs)
        assert adapter_name in self.active_adapters, "not sure how to handle 'adapter_name not in self.active_adapters'"
        self._replace_module(parent, target_name, new_module, target)
    
    def _replace_module(self, parent, child_name, new_module, child):
        """
        ref to implementation in lora/model.py:LoraModule
        """
        setattr(parent, child_name, new_module)
        meta = torch.device("meta")
        # dispatch to correct device
        processed_modules = []
        unprocessed_modules = []
        for name, module in new_module.named_modules():
            unprocessed_modules.append(name)
            if (self.prefix in name) or ("ranknum" in name):
                weight = next(child.parameters())
                if not any(p.device == meta for p in module.parameters()):
                    unprocessed_modules.pop()
                    processed_modules.append(name)
                    module.to(weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """
        the same as lora/model.py:LoraModel

        This is called in BaseTuner: __init__() ==> inject_adapter().
        Seems this function is not needed if we just have 1 adapter. 
        """
        self.active_adapter = adapter_name


    def _prepare_adapter_config(self, peft_config, model_config: dict): # -> PeftConfig
        return peft_config

    def disable_adapter_layers(self) -> None:
        pass

    def enable_adapter_layers(self) -> None:
        pass

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)