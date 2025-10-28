# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, Literal, Optional, Union, List

import torch
from transformers.training_args import _convert_str_dict
from typing_extensions import Self
from transformers import Seq2SeqTrainingArguments
import re
import subprocess
from datetime import datetime
import pytz
from .data_args import DataArguments
from .finetuning_args import FinetuningArguments
from model_moe import PRETRAINED_MODEL_ZOO


@dataclass
class DummyArguments:
    """
    Just to allow user defined var in Yaml config, and avoid yaml parsing error
    """
    define: Optional[str] = field(
        default=None,
        metadata={"help": "when defining a var in yaml, the parser will load 'define' key. Useless"}
    )


@dataclass
class CustomizedSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    trainer_model_zoo_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "if not providing model_name_or_path, then retrieve path from "
                "llamafactory/hparams/sft_constants.py:PRETRAINED_MODEL_ZOO"
            ),
        }
    )
    _output_dir_format: Optional[Dict[str, str]] = field(
        default=None,
        metadata={
            "help": (
                "if provided, will use the kv pairs in _output_dir_format "
                "to generate the full output_dir"
            )
        }
    )
    data_template: Optional[str] = field(default=None, metadata={"help": "e.g., llama2_plain"})
    wandb_project: Optional[str] = field(default=None)
    wandb_job_type: Optional[str] = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        if self.trainer_model_zoo_name is not None:
            self.output_dir = PRETRAINED_MODEL_ZOO[self.trainer_model_zoo_name].local_moe_path_output_template
        vars = re.findall(r'\$\[(.*?)\]', self.output_dir)
        for var in vars:
            var_f = "$[" + var + "]"
            if var == "commit":
                try:
                    # Run the git command to get the latest commit hash
                    commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT)
                    # Decode the byte string and return it
                    val = commit_hash.decode('utf-8').strip()[:8]
                except subprocess.CalledProcessError as e:
                    print(f"Error while getting git commit hash: {e.output.decode('utf-8')}")
                    val = "xxx"
            elif var == "ts":
                val = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y-%m-%d_%H-%M-%S")
            else:
                assert var in ["datasets", "smore_arch"], f"unknown var {var} in output_dir spec"
                continue
            self.output_dir = self.output_dir.replace(var_f, val)
        if self._output_dir_format is not None:
            self.output_dir = self.output_dir.format(**self._output_dir_format)
    
    def amend_logs_by_data_args(self, data_args: DataArguments):
        vars = re.findall(r'\$\[(.*?)\]', self.output_dir)
        for var in vars:
            var_f = "$[" + var + "]"
            if var == "datasets":
                self.output_dir = self.output_dir.replace(var_f, "-".join(sorted(data_args.dataset)))
    
    def amend_logs_by_finetuning_args(self, finetuning_args: FinetuningArguments):
        vars = re.findall(r'\$\[(.*?)\]', self.output_dir)
        for var in vars:
            var_f = "$[" + var + "]"
            if var == "smore_arch":
                if finetuning_args.moe_arch != "smore":
                    self.output_dir = self.output_dir.replace(var_f, "")
                else:
                    smore_summary = "_".join(
                        [
                            "-".join(map(lambda x: f"{k}{x}", getattr(finetuning_args, v)))
                            for k, v in (
                                ("s", "moe_num_experts"),
                                ("f", "moe_num_active"),
                                ("g", "moe_expert_dims"),
                            )
                        ]
                    )
                    self.output_dir = self.output_dir.replace(var_f, smore_summary)
    
    # def add_wandb(self, wandb_project: str, wandb_run_name: str):
    #     # import os
    #     # os.environ["WANDB_PROJECT"] = wandb_project
    #     # os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    #     self.report_to = ["wandb"]
    #     self.run_name = wandb_run_name


@dataclass
class QuantizationArguments:
    r"""
    Arguments pertaining to the quantization method.
    """

    quantization_method: Literal["bitsandbytes", "hqq", "eetq"] = field(
        default="bitsandbytes",
        metadata={"help": "Quantization method to use for on-the-fly quantization."},
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model using on-the-fly quantization."},
    )
    quantization_type: Literal["fp4", "nf4"] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in bitsandbytes int4 training."},
    )
    double_quantization: bool = field(
        default=True,
        metadata={"help": "Whether or not to use double quantization in bitsandbytes int4 training."},
    )
    quantization_device_map: Optional[Literal["auto"]] = field(
        default=None,
        metadata={"help": "Device map used to infer the 4-bit quantized model, needs bitsandbytes>=0.43.0."},
    )


@dataclass
class ProcessorArguments:
    r"""
    Arguments pertaining to the image processor.
    """

    image_resolution: int = field(
        default=512 * 512,
        metadata={"help": "Keeps the number of pixels of image below this resolution."},
    )
    video_resolution: int = field(
        default=128 * 128,
        metadata={"help": "Keeps the number of pixels of video below this resolution."},
    )
    video_fps: float = field(
        default=2.0,
        metadata={"help": "The frames to sample per second for video inputs."},
    )
    video_maxlen: int = field(
        default=64,
        metadata={"help": "The maximum number of sampled frames for video inputs."},
    )


@dataclass
class ExportArguments:
    r"""
    Arguments pertaining to the model export.
    """

    export_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory to save the exported model."},
    )
    export_size: int = field(
        default=1,
        metadata={"help": "The file shard size (in GB) of the exported model."},
    )
    export_device: Literal["cpu", "auto"] = field(
        default="cpu",
        metadata={"help": "The device used in model export, use `auto` to accelerate exporting."},
    )
    export_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the exported model."},
    )
    export_quantization_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the dataset or dataset name to use in quantizing the exported model."},
    )
    export_quantization_nsamples: int = field(
        default=128,
        metadata={"help": "The number of samples used for quantization."},
    )
    export_quantization_maxlen: int = field(
        default=1024,
        metadata={"help": "The maximum length of the model inputs used for quantization."},
    )
    export_legacy_format: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the `.bin` files instead of `.safetensors`."},
    )
    export_hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the repository if push the model to the Hugging Face hub."},
    )


@dataclass
class VllmArguments:
    r"""
    Arguments pertaining to the vLLM worker.
    """

    vllm_maxlen: int = field(
        default=4096,
        metadata={"help": "Maximum sequence (prompt + response) length of the vLLM engine."},
    )
    vllm_gpu_util: float = field(
        default=0.9,
        metadata={"help": "The fraction of GPU memory in (0,1) to be used for the vLLM engine."},
    )
    vllm_enforce_eager: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable CUDA graph in the vLLM engine."},
    )
    vllm_max_lora_rank: int = field(
        default=32,
        metadata={"help": "Maximum rank of all LoRAs in the vLLM engine."},
    )
    vllm_config: Optional[Union[dict, str]] = field(
        default=None,
        metadata={"help": "Config to initialize the vllm engine. Please use JSON strings."},
    )


@dataclass
class ModelArguments(QuantizationArguments, ProcessorArguments, ExportArguments, VllmArguments):
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.
    """

    model_zoo_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "if not providing model_name_or_path, then retrieve path from "
                "llamafactory/hparams/sft_constants.py:PRETRAINED_MODEL_ZOO"
            ),
        }
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the model's tokenizer (if None, will load tokenizer from model_name_or_path)."
        },
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the adapter weight or identifier from huggingface.co/models. "
                "Use commas to separate multiple adapters."
            )
        },
    )
    adapter_folder: Optional[str] = field(
        default=None,
        metadata={"help": "The folder containing the adapter weights to load."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    resize_vocab: bool = field(
        default=False,
        metadata={"help": "Whether or not to resize the tokenizer vocab and the embedding layers."},
    )
    split_special_tokens: bool = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."},
    )
    new_special_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "Special tokens to be added into the tokenizer. Use commas to separate multiple tokens."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether or not to use memory-efficient model loading."},
    )
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={"help": "Which scaling strategy should be adopted for the RoPE embeddings."},
    )
    flash_attn: Literal["auto", "disabled", "sdpa", "fa2"] = field(
        default="auto",
        metadata={"help": "Enable FlashAttention for faster training and inference."},
    )
    shift_attn: bool = field(
        default=False,
        metadata={"help": "Enable shift short attention (S^2-Attn) proposed by LongLoRA."},
    )
    mixture_of_depths: Optional[Literal["convert", "load"]] = field(
        default=None,
        metadata={"help": "Convert the model to mixture-of-depths (MoD) or load the MoD model."},
    )
    use_unsloth: bool = field(
        default=False,
        metadata={"help": "Whether or not to use unsloth's optimization for the LoRA training."},
    )
    use_unsloth_gc: bool = field(
        default=False,
        metadata={"help": "Whether or not to use unsloth's gradient checkpointing."},
    )
    enable_liger_kernel: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable liger kernel for faster training."},
    )
    moe_aux_loss_coef: Optional[float] = field(
        default=None,
        metadata={"help": "Coefficient of the auxiliary router loss in mixture-of-experts model."},
    )
    moe_arch_config: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "specify hier residual MoE arch"}
    )
    disable_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable gradient checkpointing."},
    )
    upcast_layernorm: bool = field(
        default=False,
        metadata={"help": "Whether or not to upcast the layernorm weights in fp32."},
    )
    upcast_lmhead_output: bool = field(
        default=False,
        metadata={"help": "Whether or not to upcast the output of lm_head in fp32."},
    )
    train_from_scratch: bool = field(
        default=False,
        metadata={"help": "Whether or not to randomly initialize the model weights."},
    )
    infer_backend: Literal["huggingface", "vllm"] = field(
        default="huggingface",
        metadata={"help": "Backend engine used at inference."},
    )
    offload_folder: str = field(
        default="offload",
        metadata={"help": "Path to offload model weights."},
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether or not to use KV cache in generation."},
    )
    infer_dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(
        default="auto",
        metadata={"help": "Data type for model weights and activations at inference."},
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    ms_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with ModelScope Hub."},
    )
    om_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Modelers Hub."},
    )
    print_param_status: bool = field(
        default=False,
        metadata={"help": "For debugging purposes, print the status of the parameters in the model."},
    )
    compute_dtype: Optional[torch.dtype] = field(
        default=None,
        init=False,
        metadata={"help": "Torch data type for computing model outputs, derived from `fp/bf16`. Do not specify it."},
    )
    device_map: Optional[Union[str, Dict[str, Any]]] = field(
        default=None,
        init=False,
        metadata={"help": "Device map for model placement, derived from training stage. Do not specify it."},
    )
    model_max_length: Optional[int] = field(
        default=None,
        init=False,
        metadata={"help": "The maximum input length for model, derived from `cutoff_len`. Do not specify it."},
    )
    block_diag_attn: bool = field(
        default=False,
        init=False,
        metadata={"help": "Whether use block diag attention or not, derived from `neat_packing`. Do not specify it."},
    )
    _model_name_or_path_format: Optional[Dict[str, str]] = field(
        default=None,
        metadata={
            "help": "use this to format model_name_or_path to form the full path"
        },
    )
    _tokenizer_name_or_path_format: Optional[str] = field(
        default=None,
        metadata={
            "help": "use this to format tokenizer_name_or_path to form the full path"
        },
    )

    def __post_init__(self):
        if self.model_zoo_name is None:
            assert self.model_name_or_path is not None, "Please provide `model_name_or_path`."
            if self.tokenizer_name_or_path is None:
                self.tokenizer_name_or_path = self.model_name_or_path
        else:
            # if self.model_zoo_name provided, IGNORE user-provided model_name_or_path & tokenizer_name_or_path
            _pretrained_info = PRETRAINED_MODEL_ZOO[self.model_zoo_name]
            if _pretrained_info.local_path is None:
                self.model_name_or_path = _pretrained_info.hf_path
                self.tokenizer_name_or_path = _pretrained_info.hf_path
                if self._model_name_or_path_format is not None:
                    print("!!!!!! _model_name_or_path_format FROM YAML CONFIG NOT USED !!!!!")
            else:
                self.model_name_or_path = (
                    _pretrained_info.local_path
                    if self._model_name_or_path_format is None
                    else _pretrained_info.local_moe_path_input_template
                )
                self.tokenizer_name_or_path = (
                    _pretrained_info.local_path
                    if self._tokenizer_name_or_path_format is None
                    else _pretrained_info.local_moe_path_input_template
                )
        if self._tokenizer_name_or_path_format is not None:
            self.tokenizer_name_or_path = self.tokenizer_name_or_path.format(**self._tokenizer_name_or_path_format)
        if self._model_name_or_path_format is not None:
            self.model_name_or_path = self.model_name_or_path.format(**self._model_name_or_path_format)

        if self.split_special_tokens and self.use_fast_tokenizer:
            raise ValueError("`split_special_tokens` is only supported for slow tokenizers.")

        if self.adapter_name_or_path is not None:  # support merging multiple lora weights
            self.adapter_name_or_path = [path.strip() for path in self.adapter_name_or_path.split(",")]

        if self.new_special_tokens is not None:  # support multiple special tokens
            self.new_special_tokens = [token.strip() for token in self.new_special_tokens.split(",")]

        if self.export_quantization_bit is not None and self.export_quantization_dataset is None:
            raise ValueError("Quantization dataset is necessary for exporting.")

        if isinstance(self.vllm_config, str) and self.vllm_config.startswith("{"):
            self.vllm_config = _convert_str_dict(json.loads(self.vllm_config))

    @classmethod
    def copyfrom(cls, source: "Self", **kwargs) -> "Self":
        init_args, lazy_args = {}, {}
        for attr in fields(source):
            if attr.init:
                init_args[attr.name] = getattr(source, attr.name)
            else:
                lazy_args[attr.name] = getattr(source, attr.name)

        init_args.update(kwargs)
        result = cls(**init_args)
        for name, value in lazy_args.items():
            setattr(result, name, value)

        return result
