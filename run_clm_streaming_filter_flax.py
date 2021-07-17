#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
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
"""
Pre-training/Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

from ast import Str
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import json
import shutil
from flax import training
import numpy as np
import datasets
from datasets import load_dataset
from tqdm import tqdm

import jax
import jax.profiler
import jax.numpy as jnp
import optax
import transformers
import flax
from flax import jax_utils, traverse_util
from flax.jax_utils import unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
from flax.serialization import to_bytes, from_bytes
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    is_tensorboard_available,
)
from transformers.testing_utils import CaptureLogger

from importlib.util import find_spec
from utils import PrefetchDataloaderWithFilter, make_batch

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`."
        },
    )
    save_optimizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to store full train state including optimizer."},
    )
    repo_path_or_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the modelhub repo directory"},
    )
    repo_url: Optional[str] = field(
        default=None,
        metadata={"help": "URL of the modelhub repo"},
    )
    decay_steps: int = field(default=None, metadata={"help":"Number of steps from peak to final learning rate"})

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    data_dir: Optional[str] = field(default=None, metadata={"help": "Path to data directory."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    text_column_name: Optional[str] = field(
            default='text',
            metadata={"help": "Column containing main text data."},
        )
    shuffle_buffer_size: int = field(
        default=10000, metadata={"help": "The number of examples to pre-load for shuffling."}
    )
    num_train_steps: int = field(default=50000, metadata={"help": "The number of training steps."})
    num_eval_samples: int = field(default=50000, metadata={"help": "The number of samples to be used for evaluation"})
    prefetch_buffer: int = field(default=8, metadata={"help": "The number of batches to prefetch for loading"})

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))


def write_train_metric(summary_writer, train_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


def create_learning_rate_fn(
    num_train_steps: int, train_batch_size: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn
def gpt3_schedule(warmup_steps,
                  total_steps,
                  peak_lr,
                  end_lr):
    def sch(step):
        warmup_pct = jnp.clip(step, 0, warmup_steps) / warmup_steps
        anneal_pct = jnp.clip(step - warmup_steps, 0, total_steps) / total_steps

        return warmup_pct * peak_lr - (peak_lr - end_lr) * (1 - jnp.cos(jnp.pi * anneal_pct)) / 2

    return sch

# utils
def mb_item(x):
    return x.item() if hasattr(x, "item") else x


#checkpoint functions
def save_model_checkpoint(model, save_dir, state, with_opt:bool=True, push_to_hub:bool=False):
    """
    If `push_to_hub` is True, will save to `save_dir`. Otherwise will save to `save_dir/ckpt-{step}`.
    """
    state = jax_utils.unreplicate(state)
    logger.info(f"SAVING CHECKPOINT IN {save_dir}...")
    if not push_to_hub:
        save_dir = f"{save_dir}/ckpt-{mb_item(state.step)-1}"
    model.save_pretrained(
        save_dir,
        params=state.params,
        push_to_hub=push_to_hub,
        commit_message=f"Saving weights and logs at step {mb_item(state.step)-1}",
    )
    if with_opt:
        with open(os.path.join(save_dir, "opt_state.msgpack"), "wb") as f:
            f.write(to_bytes(state.opt_state))
        with open(os.path.join(save_dir, "training_state.json"), "w") as f:
            json.dump({"step": state.step.item()}, f)
    logger.info("checkpoint saved")

# this is added to make resuming from checkpoint to work with adafactor
# to be removed when issue is fixed
# notice that adafactor state is perturbed by fake_update
def _zeros_tree_like(inp_tree):
    return jax.tree_map(jnp.zeros_like, inp_tree)

def fake_update(state):
    fake_updates = _zeros_tree_like(state.params)
    _, new_inner_opt_state = state.tx.inner_opt.update(fake_updates, state.opt_state.inner_opt_state, state.params)
    opt_state = state.opt_state
    new_opt_state = optax.MultiStepsState(mini_step=opt_state.mini_step, 
                                        gradient_step=opt_state.gradient_step, 
                                        inner_opt_state=new_inner_opt_state,
                                        acc_grads=opt_state.acc_grads)
    return state.replace(opt_state=new_opt_state)

def reinstantiate_states(opt_state):
    new_state = []
    for state in opt_state:
        if isinstance(state, list):
            new_state.append(reinstantiate_states(state))
        else:
            cls = getattr(optax, type(state).__name__)
            new_state.append(cls(**{k:getattr(state, k) for k in state._fields}))
    return new_state

def restore_model_checkpoint(save_dir, state):
    logger.info(f"RESTORING CHECKPOINT FROM {save_dir}...")
    with open(os.path.join(save_dir, "flax_model.msgpack"), "rb") as f:
        params = from_bytes(state.params, f.read())

    with open(os.path.join(save_dir, "opt_state.msgpack"), "rb") as f:
        opt_state = from_bytes(state.opt_state, f.read())

    with open(os.path.join(save_dir, "training_state.json"), "r") as f:
        training_state = json.load(f)
    step = training_state["step"]

    logger.info("checkpoint restored")
    # reinstantiate inner opt state to avoid type conflict
    if hasattr(opt_state, "inner_opt_state"):
        print("restoring state ofmultisteps optimizer")
        inner_opt_state = reinstantiate_states(opt_state.inner_opt_state)
        ms_state_dict = {k:getattr(state.opt_state, k) for k in state.opt_state._fields}
        ms_state_dict["inner_opt_state"] = inner_opt_state
        opt_state = optax.MultiStepsState(**ms_state_dict)

    return state.replace(step=step, params=params, opt_state=opt_state)

def rotate_checkpoints(ckpt_dir:str, save_total_limit:int):
    "Removes older checkpoints so that `save_total_limit` checkpoints are kept"
    # TODO: what to remove is decided using step number only, we might want to improve that
    ckpts = [str(x) for x in Path(ckpt_dir).glob("ckpt-*")]
    # sort checkpoints by step
    ckpts_sorted = sorted(ckpts, key=lambda x: int(x.split('-')[-1]))
    ckpts_to_delete = ckpts_sorted[:-save_total_limit]
    for ckpt in ckpts_to_delete:
        logger.info(f"Deleting older checkpoint [{ckpt}] due to save_total_limit ({save_total_limit})")
        shutil.rmtree(ckpt)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    #  Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        train_dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            data_dir=data_args.data_dir,
            cache_dir=model_args.cache_dir, 
            streaming=True,
            split="train"
        )
        eval_dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            data_dir=data_args.data_dir,
            cache_dir=model_args.cache_dir, 
            streaming=True,
            split="validation"
        )
        
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = FlaxAutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, config=config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype)
        )
    else:
        model = FlaxAutoModelForCausalLM.from_config(
            config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype)
        )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # column_names = eval_dataset.column_names
    text_column_name = data_args.text_column_name # if data_args.text_column_name in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        return output

    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
    )
    tokenized_eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        # remove_columns=column_names,
        # num_proc=data_args.preprocessing_num_workers,
        # load_from_cache_file=not data_args.overwrite_cache,
    )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    
    train_loader = PrefetchDataloaderWithFilter(
        tokenized_dataset, 
        training_args.max_steps * training_args.gradient_accumulation_steps, 
        int(training_args.per_device_train_batch_size) * jax.device_count(),
        block_size,
        shuffle_buffer=10000,
        prefetch_buffer=data_args.prefetch_buffer,
        seed=training_args.seed
    )
    # evaluation data is not in streaming mode
    # if training_args.do_eval:
    #     eval_dataset = tokenized_eval_dataset.map(
    #         group_texts,
    #         batched=True,
    #         num_proc=data_args.preprocessing_num_workers,
    #         load_from_cache_file=not data_args.overwrite_cache,
    #     )
    #     if data_args.max_eval_samples is not None:
    #         eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
    
    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )
    
    # enable wandb tracking
    has_wandb = find_spec("wandb") is not None 
    if jax.process_index() == 0 and has_wandb and ("wandb" in training_args.report_to):
        try:
            import wandb
            wandb.init(
                name=training_args.run_name,
                entity="wandb", 
                project="hf-flax-gpt-neo-copilot",
                sync_tensorboard=True
            )
            wandb.config.update(training_args)
            wandb.config.update(model_args)
            wandb.config.update(data_args)
        except ImportError as e:
            print(e)
            has_wandb = False
    

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    # Store some constant
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count() * training_args.gradient_accumulation_steps
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()
    total_train_steps = training_args.max_steps * training_args.gradient_accumulation_steps

    # Create learning rate schedule
    gpt3_schedule_fn = gpt3_schedule(
        training_args.warmup_steps,
        model_args.decay_steps,
        training_args.learning_rate,
        training_args.learning_rate / 10.
    )

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    # Note that this mask is specifically adapted for FlaxGPT2.
    # For other models, one should correct the layer norm parameter naming
    # accordingly.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {
            path: (path[-1] != "bias" and path[-2:] not in [("ln_1", "scale"), ("ln_2", "scale"), ("ln_f", "scale")])
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)

    # create optimizer
    if training_args.adafactor:
        # We use the default parameters here to initialize adafactor,
        # For more details about the parameters please check https://github.com/deepmind/optax/blob/ed02befef9bf81cbbf236be3d2b0e032e9ed4a40/optax/_src/alias.py#L74
        optimizer = optax.adafactor(
            learning_rate=gpt3_schedule_fn,
        )
    else:
        optimizer = optax.adamw(
            learning_rate=gpt3_schedule_fn,
            b1=training_args.adam_beta1,
            b2=training_args.adam_beta2,
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
            mask=decay_mask_fn,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1),
            optimizer
        )
    if training_args.gradient_accumulation_steps > 1:
        optimizer = optax.MultiSteps(optimizer, training_args.gradient_accumulation_steps)
    grad_accum_steps = training_args.gradient_accumulation_steps

    # Setup train state
    state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=optimizer, dropout_rng=dropout_rng)
    
    if training_args.resume_from_checkpoint:
        state = restore_model_checkpoint(training_args.resume_from_checkpoint, state)
        resume_step = mb_item(state.step)
        if training_args.adafactor:
            state = fake_update(state)
    else:
        resume_step = 0

    def loss_fn(logits, labels):
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        loss = optax.softmax_cross_entropy(shift_logits, onehot(shift_labels, shift_logits.shape[-1]))
        return loss.mean()

    # Define gradient update step fn
    def train_step(state, batch):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def compute_loss(params):
            labels = batch.pop("labels")
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            loss = loss_fn(logits, labels)
            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)

        metrics = {"loss": loss, "learning_rate": gpt3_schedule_fn(state.step // grad_accum_steps)}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics

    # Define eval fn
    def eval_step(params, batch):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, train=False)[0]
        loss = loss_fn(logits, labels)

        # summarize metrics
        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        return metrics

    # Create parallel version of the train and eval step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step, "batch")

    # Replicate the train state on each device
    state = state.replicate()

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed and grad_accum) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {training_args.max_steps}")

    if not training_args.skip_memory_metrics:
        server = jax.profiler.start_server(9999)

    train_time = 0
    train_metrics = []
    # TODO: figure out training duration
    steps = tqdm(range(training_args.max_steps), position=0, initial=resume_step)
    for step in range(total_train_steps):
        # ======================== Training ================================
        train_start = time.time()
        rng, input_rng = jax.random.split(rng)
        
        cur_step = step
        # skip to the step from which we are resuming
        if cur_step < resume_step:
            continue
        
        # using advance_iter_and_group_samples seem to make training slower
        # samples = advance_iter_and_group_samples(iter(tokenized_dataset), int(training_args.per_device_train_batch_size) * jax.device_count(), block_size)
        # batch = shard(make_batch(samples))
        batch = shard(next(train_loader))
        # logger.info(f"{batch['input_ids'].shape}")
        state, train_metric = p_train_step(state, batch)
        train_metrics.append(train_metric)
        if step % grad_accum_steps == 0:
            steps.update(1)

        if cur_step % (training_args.logging_steps * grad_accum_steps)== 0 and cur_step > 0:
            # Save metrics
            train_metric = unreplicate(train_metric)
            train_time += time.time() - train_start
            if has_tensorboard and jax.process_index() == 0:
                write_train_metric(summary_writer, train_metrics, train_time, cur_step//grad_accum_steps)
            if has_wandb and jax.process_index() == 0 and ("wandb" in training_args.report_to):
                # TODO: add accumulation of metrics
                _metrics = {k if k=="learning_rate" else f"train_{k}":mb_item(v.mean()) for k, v in train_metric.items()}
                wandb.log({"training_step":cur_step//grad_accum_steps, **_metrics}, commit=True)

            steps.write(
                f"Step... ({cur_step // grad_accum_steps} | Loss: {train_metric['loss'].mean()}, Learning Rate: {train_metric['learning_rate'].mean()})"
            )

            train_metrics = []

        if cur_step % (training_args.eval_steps * grad_accum_steps) == 0 and cur_step > 0 and training_args.do_eval:
            # ======================== Evaluating ==============================
            eval_metrics = []
            eval_steps = data_args.max_eval_samples # len(eval_dataset) // eval_batch_size
            # eval_loader = data_loader(input_rng, eval_dataset, eval_batch_size)
            eval_loader = PrefetchDataloaderWithFilter(
                tokenized_eval_dataset, 
                eval_steps,
                eval_batch_size,
                block_size,
                prefetch_buffer=data_args.prefetch_buffer,
                shuffle=False,
            )
            eval_pbar = tqdm(range(eval_steps), desc="Evaluating...", position=2, leave=False)
            for _ in eval_pbar:
                # Model forward
                batch = shard(next(eval_loader))
                metrics = p_eval_step(state.params, batch)
                eval_metrics.append(metrics)

            # normalize eval metrics
            eval_metrics = get_metrics(eval_metrics)
            eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

            try:
                eval_metrics["perplexity"] = math.exp(eval_metrics["loss"])
            except OverflowError:
                eval_metrics["perplexity"] = float("inf")
            # TODO: this needs to be closed properly
            eval_loader.terminate()
            # Print metrics and update progress bar
            desc = f"Step... ({cur_step//grad_accum_steps} | Eval Loss: {eval_metrics['loss']} | Eval Perplexity: {eval_metrics['perplexity']})"
            eval_pbar.write(desc)
            eval_pbar.desc = desc

            # Save metrics
            if has_tensorboard and jax.process_index() == 0:
                # cur_step = epoch * (len(train_dataset) // train_batch_size)
                write_eval_metric(summary_writer, eval_metrics, cur_step//grad_accum_steps)
            if has_wandb and jax.process_index() == 0 and ("wandb" in training_args.report_to):
                _metrics = {f"eval_{k}":mb_item(v) for k, v in eval_metrics.items()}
                wandb.log({"eval_step":cur_step//grad_accum_steps, **_metrics})

        if cur_step % (training_args.save_steps * grad_accum_steps) == 0 and cur_step > 0:
            # save checkpoint after each epoch and push checkpoint to the hub
            if jax.process_index() == 0:
                save_model_checkpoint(model, training_args.output_dir, state, with_opt=model_args.save_optimizer,
                                      push_to_hub=training_args.push_to_hub)
                # if model_args.save_optimizer:
                    # this saves full state including optimizer
                    # save_checkpoint(training_args.output_dir, jax_utils.unreplicate(state), cur_step, keep=training_args.save_total_limit, overwrite=True)
                if training_args.save_total_limit is not None:
                    rotate_checkpoints(training_args.output_dir, training_args.save_total_limit)
    
    train_loader.terminate()
    # save model after training is over
    save_model_checkpoint(model, training_args.output_dir, state, with_opt=False,
                          push_to_hub=training_args.push_to_hub)

    logger.info("***Training comleted")


if __name__ == "__main__":
    main()

