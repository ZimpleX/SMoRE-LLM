from __future__ import annotations

from dataclasses import dataclass, field
from peft.config import PeftConfig
from peft.utils import PeftType
from typing import List, Optional, Literal, Union


@dataclass
class HieRMoEConfig(PeftConfig):
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D "
                "(if the model is a PreTrainedModel, the output layer excluded)."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            ),
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from Lora."},
    )
    moe_arch: str = field(default="llama_moe")
    num_experts: List[int] = field(default=lambda: [8])
    num_active: List[int] = field(default=lambda: [2])
    expert_dims: List[int] = field(default=lambda: [512])
    # we mainly pass lora_alpha into the smore model. lora_scaling is for logging purposes
    lora_alpha: Optional[List[float]] = None
    lora_scaling: Optional[List[float]] = None
    gate_dims: List[int] = field(default=lambda: [32])
    gate_act_fn: str = field(default="relu")
    gate_type: str = field(default="noisy_topk")
    gate_arch: str = field(default="mlp")
    layer_forward_mode: str = field(default="subgraph")  # can be tree or subgraph. "tree" may NOT be numerically stable
    balance_loss_weight: float = field(default=1e-2)
    expert_act_fn: str = field(default="relu")
    init_method: str = field(default="b_zero")
    is_shared_expert_pool: bool = field(default=False)
    activation_position: str = field(default="all")
    dim_downproj_x: int = field(default=32)
    moe_dropout: float = field(default=0.0, metadata={"help": "moe dropout"})
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_moe_weights: (
        bool | Literal["gaussian", "eva", "olora", "pissa", "pissa_niter_[number of iters]", "corda", "loftq"]
    ) = field(
        default=True,
        metadata={
            "help": (
                "How to initialize the weights of the LoRA layers. Passing `'True'` (default) results in the default "
                "initialization from the reference implementation from Microsoft. Passing `'gaussian'` results "
                "in Gaussian initialization scaled by the LoRA rank for linear and layers. Setting the initialization "
                "to `'False'` leads to completely random initialization and *is discouraged.*"
                "Pass `'eva'` results in a data-driven initialization of Explained Variance Adaptation."
                "Passing `'olora'` results in OLoRA initialization."
                "Passing `'pissa'` results in PiSSA initialization."
                "Passing `'pissa_niter_[number of iters]'` initiates Fast-SVD-based PiSSA initialization, "
                "where [number of iters] indicates the number of subspace iterations to perform fsvd, and must be a nonnegative integer."
                "Passing `'corda'` results in CorDA initialization."
                "Pass `'loftq'` to use LoftQ initialization"
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index. "
            "This only works when target_modules is a list of str."
        },
    )
    bias: Literal["none", "all", "moe_only"] = field(
        default="none", metadata={"help": "Bias type for MoE. Can be 'none', 'all' or 'moe_only'"}
    )

    def __post_init__(self):
        # see llamafactory/model/adapter.py:_setup_moe_tuning
        super().__post_init__()
        self.peft_type = PeftType.HIER_MOE
        # align lora scaling configs
        # assert self.lora_alpha is None or self.lora_scaling is None
        assert len(self.expert_dims) == len(self.num_active) == len(self.num_experts)