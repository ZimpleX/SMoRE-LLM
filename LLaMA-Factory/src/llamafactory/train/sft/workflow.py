# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

from typing import TYPE_CHECKING, List, Optional

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps, get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer
import wandb
from ..callbacks import ProgressCallback as ProgressCallbackCustom
from ..callbacks import WandbCallback as WandbCallbackCustom
from ..callbacks import LogCallback
from transformers import DefaultFlowCallback
from transformers import ProgressCallback as ProgressCallbackOrig
from transformers.integrations.integration_utils import WandbCallback as WandbCallbackOrig
from model_moe.model_utils import ParamStatsSMoRE


if TYPE_CHECKING:
    from transformers import TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments, CustomizedSeq2SeqTrainingArguments


logger = get_logger(__name__)

def decode_moe_name(
    model_args: "ModelArguments", 
    model, 
    cfg_prefix: Optional[str]="A"
):
    if model_args.moe_arch_config is None:
        moe_type = "NA"
        init = "NA"
        pt_arch = "NA"
        moe_scale = "NA"
    else:
        (
            moe_type,
            init, 
            pt_arch, 
            moe_scale
        ) = model_args._model_name_or_path_format["moe_type"].split("-")
    if (
        hasattr(model.model, "moe_arch_config") and 
        model.model.moe_arch_config is not None
    ):
        moe_arch_config = model.model.moe_arch_config
        trainable_param = list(moe_arch_config.regex_params2train.keys())
        if pt_arch == "res_mlp":
            assert moe_arch_config.pretrained_ffn_connection == "residual"
        else:
            raise NotImplementedError(
                f"pls specify how to check config consistency for {pt_arch}"
            )
    else:
        trainable_param = []
    ret = {
        "moe_type": moe_type,
        "init": init,
        "pt_arch": pt_arch,
        "moe_scale": moe_scale,
        "trainable_param": trainable_param,
    }
    if cfg_prefix is not None:
        ret = {f"{cfg_prefix}_{k}": v for k, v in ret.items()}
    return ret

def _log_model_stats(model, text_style="\033[93m"):
    num_trainable = sum(
        [v.numel() for k, v in model.named_parameters() if v.requires_grad]
    )
    num_frozen = sum(
        [v.numel() for k, v in model.named_parameters() if not v.requires_grad]
    )
    print(f"{text_style}Total trainable params={num_trainable/1e9:.3f}G\tTotal frozen params={num_frozen/1e9:.3f}G\033[0m")


def gen_wandb_run_name(hparam_exp_name: Optional[str]):
    pass


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "CustomizedSeq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
    hparam_exp_name: Optional[str]=None,
):
    training_args.report_to = ["wandb"]
    base_model = model_args.model_zoo_name
    if base_model.startswith("[peft] "):
        base_model = base_model[len("[peft] "):]
    training_args.wandb_project = f"{base_model} SMoRE [{' | '.join(sorted(data_args.dataset))}]"
    training_args.wandb_job_type = hparam_exp_name
    training_args.data_template = data_args.template
    tokenizer_module = load_tokenizer(model_args, finetuning_args.finetuning_type)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor
    else:
        raise NotImplementedError
    
    _log_model_stats(model)
    # handle wandb: https://docs.wandb.ai/guides/integrations/huggingface/
    arch_config = decode_moe_name(model_args, model)
    all_stats = []
    for mod in model.modules():
        if hasattr(mod, "param_stats") and isinstance(mod.param_stats, ParamStatsSMoRE):
            all_stats.append(mod.param_stats)
    if len(all_stats) > 0:
        trainable_params_from_stats = sum(s.trainable_param for s in all_stats)
        assert trainable_params_from_stats == sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.base_model.model.param_stats = ParamStatsSMoRE.agg_stats(*all_stats)
    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        # return_outputs=True,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )
    default_cb = None
    log_cb = None
    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, WandbCallbackOrig):
            cb.__class__ = WandbCallbackCustom
        elif isinstance(cb, DefaultFlowCallback):
            default_cb = cb
        elif isinstance(cb, LogCallback):
            log_cb = cb
    trainer.callback_handler.callbacks = [default_cb, log_cb] + [
        cb for cb in trainer.callback_handler.callbacks if cb not in (default_cb, log_cb)
    ]
    trainer.remove_callback(ProgressCallbackOrig)
    trainer.add_callback(ProgressCallbackCustom)
    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results)

    wandb.finish()

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
