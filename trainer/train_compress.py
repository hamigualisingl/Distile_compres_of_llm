import sys
import traceback
import torch
import os
import json
import logging
import time

from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Union

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import DataCollatorForSeq2Seq
from transformers import DataCollatorForLanguageModeling, Trainer
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import set_seed

from datasets import concatenate_datasets, interleave_datasets, load_dataset, load_from_disk

from models.qwen_modify_compress import Qwen2ForCausalLMModify

from arguments import ModelArguments, DataTrainingArguments
import random

logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

model_args, data_args, training_args = parser.parse_args_into_dataclasses()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if training_args.should_log:
    # The default of training_args.log_level is passive, so we set log level at info here to have that default.
    transformers.utils.logging.set_verbosity_info()
print("end set log")
log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.bf16}"
)
logger.info(f"Training/evaluation parameters {training_args}")

# Set seed before initializing model.
set_seed(training_args.seed)
print("start tokenizer")
tokenizer = AutoTokenizer.from_pretrained("", trust_remote_code=True)

logger.info("Start to load datasets")
# make dataset
data_path = data_args.data_path
print("start load data")
dataset = load_from_disk(data_path)
print("enb load data")
IGNORE_INDEX = -100


def print_supervised_dataset_example(example):
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print("labels:\n{}".format(tokenizer.decode([
        token_id if token_id != IGNORE_INDEX else tokenizer.pad_token_id for token_id in example["labels"]
    ], skip_special_tokens=False)))
    print("compress_ids:\n{}".format(example["compress_input_ids"]))
    print("compress:\n{}".format(tokenizer.decode(example["compress_input_ids"], skip_special_tokens=False)))


print_supervised_dataset_example(next(iter(dataset)))

# make data_collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=IGNORE_INDEX
)

logger.info("Start to load model")
try:
    time.sleep(random.randint(1, 10))
    model = Qwen2ForCausalLMModify.from_pretrained(model_args.model_name_or_path, trust_remote_code=True,
                                                   torch_dtype=torch.bfloat16).cuda()
except:
    try:
        time.sleep(random.randint(1, 10))
        model = Qwen2ForCausalLMModify.from_pretrained(model_args.model_name_or_path, trust_remote_code=True,
                                                       torch_dtype=torch.bfloat16).cuda()
    except:
        logger.error(traceback.format_exc())


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=None,
    train_dataset=dataset,
)

# Training
if training_args.do_train:
    logger.info("Start to train")
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model()
 
