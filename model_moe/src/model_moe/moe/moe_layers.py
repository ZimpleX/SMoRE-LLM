import warnings
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from .moe_gates import (
    SMoREGate,
    SMoREDenseGate,
    SMoRESwitchGate,
)


@dataclass
class MoEMlpOutput(ModelOutput):
    hidden_states: Optional[torch.FloatTensor] = None
    balance_loss: Optional[torch.FloatTensor] = None
    num_dropped_tokens: Optional[int] = None
    gate_load: Optional[list] = None
    gate_importance: Optional[list] = None


@dataclass
class PretrainedMLPConfig:
    dim_io: Optional[int] = None
    dim_hidden: Optional[int] = None
    act_fn: Optional = None
    device: Optional[str] = None
    dtype: Optional[str] = None


class BaseMoELayer(nn.Module):
    def __init__(
        self, 
        pretrained_base_layer,
    ):
        super().__init__()
        self.adapter_name = "default"
        self.base_layer = pretrained_base_layer
        self.moe_gate = nn.ModuleDict({})
        self.moe_calculator = nn.ModuleDict({})
        
    def _create_gate(self, **kwargs):
        if "num_active" in kwargs:
            num_active = kwargs["num_active"]
        else:
            num_active = self.num_active[0]
        if "num_experts" in kwargs:
            num_experts = kwargs["num_experts"]
        else:
            num_experts = self.num_experts[0]
        self.moe_gate_type = kwargs.get("gate_type", "SMoREGate")
        self.moe_gate_arch = kwargs.get("gate_arch", "mlp")
        self.moe_parent_fanout = kwargs.get("parent_fanout", [])
        dim_in = kwargs.get("dim_in", self.in_features)

        if self.moe_gate_type == "SMoREGate":
            assert self.moe_gate_arch not in ["bottom_up"]
            _gate = SMoREGate(
                dim_in,
                kwargs["dim_hid"],
                num_experts,
                num_active,
                act_fn=kwargs.get("act_fn", "tanh"),
                use_softmax=kwargs.get("use_softmax", True),
                use_balance=kwargs.get("use_balance", True),
                balance_loss_weight=kwargs.get("balance_loss_weight", 1e-2),
                add_noise=kwargs.get("add_noise", True),
                noise_epsilon=kwargs.get("noise_epsilon", 1e-2),
            )
        elif self.moe_gate_type == "SMoREDenseGate":
            _gate = SMoREDenseGate(
                dim_in=dim_in,
                dim_hid=kwargs["dim_hid"],
                num_experts=num_experts,
                num_active=num_active,
                act_fn=kwargs.get("act_fn", "tanh"),
                gate_arch=self.moe_gate_arch,
                # 
            )
        elif self.moe_gate_type == "SMoRESwitchGate":
            _gate = SMoRESwitchGate(
                dim_in=dim_in,
                dim_hid=kwargs["dim_hid"],
                num_experts=num_experts,
                num_active=num_active,
                act_fn=kwargs.get("act_fn", "tanh"),
                gate_arch=self.moe_gate_arch,
                balance_loss_weight=kwargs.get("balance_loss_weight", 1e-2),
                #
                parent_fanout=self.moe_parent_fanout,
            )
        else:
            raise NotImplementedError
        return _gate

    def set_gate_use_softmax(self, use_softmax):
        if "use_softmax" not in vars(self.moe_gate):
            raise KeyError(f'{self.moe_gate_type} does not have a key named "use_softmax".')
        else:
            self.moe_gate.use_softmax = use_softmax

    def set_gate_use_balance(self, use_balance):
        if "use_balance" not in vars(self.moe_gate):
            raise KeyError(f'{self.moe_gate_type} does not have a key named "use_balance".')
        else:
            self.moe_gate.use_balance = use_balance

    def set_gate_balance_loss_weight(self, balance_loss_weight):
        if "balance_loss_weight" not in vars(self.moe_gate):
            raise KeyError(f'{self.moe_gate_type} does not have a key named "balance_loss_weight".')
        else:
            self.moe_gate.balance_loss_weight = balance_loss_weight

    def set_gate_add_noise(self, add_noise):
        if "add_noise" not in vars(self.moe_gate):
            raise KeyError(f'{self.moe_gate_type} does not have a key named "add_noise".')
        else:
            self.moe_gate.add_noise = add_noise

    def set_gate_noise_epsilon(self, noise_epsilon):
        if "noise_epsilon" not in vars(self.moe_gate):
            raise KeyError(f'{self.moe_gate_type} does not have a key named "noise_epsilon".')
        else:
            self.moe_gate.noise_epsilon = noise_epsilon

    def set_calculator_multiply_gate_scores(self, multiply_gate_scores):
        if "multiply_gate_scores" not in vars(self.moe_calculator):
            raise KeyError(f'{self.moe_gate_type} does not have a key named "multiply_gate_scores".')
        else:
            self.moe_calculator.multiply_gate_scores = multiply_gate_scores

    def set_calculator_score_scale_factor(self, score_scale_factor):
        if "score_scale_factor" not in vars(self.moe_calculator):
            raise KeyError(f'{self.moe_gate_type} does not have a key named "score_scale_factor".')
        else:
            self.moe_calculator.score_scale_factor = score_scale_factor

    def set_calculator_drop_tokens(self, drop_tokens):
        if "drop_tokens" not in vars(self.moe_calculator):
            raise KeyError(f'{self.moe_gate_type} does not have a key named "drop_tokens".')
        elif drop_tokens and self.moe_calculator.dropped_padding != "zero" and self.in_features != self.out_features:
            warnings.warn('Setting "drop_tokens=True" without zero dropped padding when "in_features != out_features" will cause error!')
        else:
            self.moe_calculator.drop_tokens = drop_tokens

    def set_calculator_dropped_padding(self, dropped_padding):
        if "dropped_padding" not in vars(self.moe_calculator):
            raise KeyError(f'{self.moe_gate_type} does not have a key named "dropped_padding".')
        elif dropped_padding not in self.moe_calculator.available_dropped_padding_choices:
            raise ValueError(f"'dropped_padding' type not available! (available choices: {self.moe_calculator.available_dropped_padding_choices})")
        elif self.moe_calculator.drop_tokens and dropped_padding != "zero" and self.in_features != self.out_features:
            warnings.warn(f'Setting "dropped_padding={dropped_padding}" with "drop_tokens=True" when "in_features != out_features" will cause error!')
        else:
            self.moe_calculator.dropped_padding = dropped_padding

    def set_calculator_capacity_factor(self, capacity_factor):
        if "capacity_factor" not in vars(self.moe_calculator):
            raise KeyError(f'{self.moe_gate_type} does not have a key named "capacity_factor".')
        else:
            self.moe_calculator.capacity_factor = capacity_factor

    def reset_gate_network(self):
        self.moe_gate.reset_gate_network()

    def reset_experts(self):
        self.moe_calculator.reset_experts()
    # fmt: on
