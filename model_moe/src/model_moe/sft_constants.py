from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelMetaData:
    hf_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "huggingface path (from where we download checkpoint into local_path)"
        }
    )
    local_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "store pretrained chkpt here"
        }
    )
    local_moe_path_input_template: Optional[str] = field(
        default=None,
        metadata={
            "help": "store moe init checkpoint here"
        }
    )
    local_moe_path_output_template: Optional[str] = field(
        default=None,
        metadata={
            "help": "store moe trained checkpoint here"
        }
    )
    total_param: Optional[int] = field(
        default=None,
        metadata={
            "help": "may be used to validate param frozen setup"
        }
    )
    # arch spec
    up_proj_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "useful to instantiate weight tensor for residual FFN in MoE"
        }
    )



PRETRAINED_MODEL_ZOO = {
    "[peft] llama3_8b": ModelMetaData(
        hf_path="meta-llama/Meta-Llama-3-8B",
        local_path=None, 
        local_moe_path_input_template=None,
        local_moe_path_output_template="~/checkpoints/Llama-3-8B__{moe_arch}_{finetuning_type}_$[smore_arch]/sft/$[datasets]/$[commit]_$[ts]/",
        up_proj_dim=14336,
    ),
    "[peft] gemma2_9b": ModelMetaData(
        hf_path="google/gemma-2-9b",
        local_path=None,
        local_moe_path_input_template=None,
        local_moe_path_output_template="~/checkpoints/Gemma-2-9B__{moe_arch}_{finetuning_type}_$[smore_arch]/sft/$[datasets]/$[commit]_$[ts]/",
        up_proj_dim=14336,
    ),
    "[peft] llama3.2_1b": ModelMetaData(
        hf_path="meta-llama/Llama-3.2-1B",
        local_path=None, 
        local_moe_path_input_template=None,
        local_moe_path_output_template="~/checkpoints/Llama-3-2-1B__{moe_arch}_{finetuning_type}_$[smore_arch]/sft/$[datasets]/$[commit]_$[ts]/",
        up_proj_dim=8192,
    ),
}