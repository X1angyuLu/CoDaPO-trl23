# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from accelerate import logging
from datasets import load_dataset, load_from_disk

from trl import (
    DatasetMixtureConfig,
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_peft_config,
)
from trl.rewards import get_soft_overlong_punishment, think_format_reward
from transformers.trainer_utils import get_last_checkpoint
from utils.utils import make_conversation, convert_to_serializable
logger = logging.get_logger(__name__)

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


reward_funcs_registry = {
    "think_format_reward": think_format_reward,
    "get_soft_overlong_punishment": get_soft_overlong_punishment(max_completion_len=1280, soft_punish_cache=256),
}


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_model_name_or_path (`str`, *optional*):
            Reward model id of a pretrained model hosted inside a model repo on huggingface.co or local path to a
            directory containing model weights saved using [`~transformers.PreTrainedModel.save_pretrained`].
        reward_funcs (`list[str]`, *optional*):
            Reward functions to use. Supported values are:

                - `"think_format_reward"`
                - `"get_soft_overlong_punishment"` (used value are `max_completion_len=1280`, `soft_punish_cache=256`)
                - any dotted import path " (e.g., `'my_lib.rewards.custom_reward'`).
    """

    reward_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Reward model id of a pretrained model hosted inside a model repo on huggingface.co or "
            "local path to a directory containing model weights saved using `PreTrainedModel.save_pretrained`."
        },
    )
    reward_funcs: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "Reward functions to use. Supported values are: `think_format_reward`, "
            "`get_soft_overlong_punishment` (used value are `max_completion_len=1280`, `soft_punish_cache=256`), or "
            "any dotted import path (e.g., `'my_lib.rewards.custom_reward'`)."
        },
    )


def main(script_args, training_args, model_args, dataset_args):

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    os.makedirs(training_args.output_dir, exist_ok=True)
    all_args_path = os.path.join(training_args.output_dir, "all_args.json")
    all_args = {
        "model_args": vars(model_args),
        "script_args": vars(script_args),
        "training_args": vars(training_args),
    }
    all_args = convert_to_serializable(all_args)
    with open(all_args_path, "w") as f:
        json.dump(all_args, f, indent=4)
    logger.info(f"All parameters saved to {all_args_path}")


    # Get the reward models and functions
    reward_funcs = []
    if script_args.reward_model_name_or_path:
        reward_funcs.append(script_args.reward_model_name_or_path)

    if script_args.reward_funcs:
        for func_name in script_args.reward_funcs:
            if func_name in reward_funcs_registry:
                reward_funcs.append(reward_funcs_registry[func_name])
            elif "." in func_name:
                module_path, func_name = func_name.rsplit(".", 1)
                sys.path.insert(0, os.getcwd())
                module = importlib.import_module(module_path)
                reward_func = getattr(module, func_name)
                reward_funcs.append(reward_func)
            else:
                raise ValueError(
                    f"Could not load reward function '{func_name}'. Expected one of "
                    f"{list(reward_funcs_registry.keys())} or a valid import path."
                )

    # Load the dataset
    dataset = load_from_disk(script_args.dataset_name)
    dataset = dataset.map(make_conversation, fn_kwargs={"prompt_column": script_args.dataset_prompt_column, "system_prompt": training_args.system_prompt})
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    # if dataset_args.datasets and script_args.dataset_name:
    #     logger.warning(
    #         "Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used to load the "
    #         "dataset and `dataset_name` will be ignored."
    #     )
    # elif dataset_args.datasets and not script_args.dataset_name:
    #     dataset = get_dataset(dataset_args)
    # elif not dataset_args.datasets and script_args.dataset_name:
    #     dataset = load_dataset(
    #         script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
    #     )
    # else:
    #     raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    # Train the model
    logger.info("*** Train ***")
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    trainer.train(resume_from_checkpoint=checkpoint)

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    dataclass_types = (GRPOScriptArguments, GRPOConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("grpo", help="Run the GRPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args)