import os
import sys
import logging
import time
import datetime
import json
import traceback
import copy
import transformers
from transformers import AutoTokenizer

from datasets import concatenate_datasets, interleave_datasets, load_dataset, load_from_disk
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Union

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
IGNORE_INDEX = -100
tokenizer = AutoTokenizer.from_pretrained("/mnt/data/user/lidehu/vae/qwen2_trainer-master/expand_0704", trust_remote_code=True)


IGNORE_INDEX = -100
eos_token_id = tokenizer.encode('[EOS]')[0]
pad_token_id = tokenizer.encode('[PAD]')[0]
max_target_length=2048

print(eos_token_id)
print(pad_token_id)

def replace_speaker(text):
    text = text.replace('<|im_start|>user', '[SPEAKER_1]')
    text = text.replace('<|im_start|>assistant', '[SPEAKER_0]')
    return text.replace("<|im_end|>", '')

def preprocess_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    # model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "compress_input_ids": []}
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "compress_input_ids": [], "compress_attention_mask": []}

    for caption, prompt in zip(examples['memory'], examples['text']):
    
        caption, prompt = examples['memory'], examples['text']
        ######### memory是要压缩的,这边的promt其实是正文
        caption, prompt = replace_speaker(caption), replace_speaker(prompt)
        ###########第一次前向准备工作
        caption_ids_one = tokenizer.encode(caption)[:149] + [eos_token_id]
        source_input_one = tokenizer.encode(prompt)[:max_target_length-150]#预留150个
        first_attention_mask=[1]*max_target_length
        fist_input=caption_ids_one+source_input_one+[tokenizer.pad_token_id]*(max_target_length-len(caption_ids_one)-len(source_input_one))
        fisrt_lable=[0]*len(caption_ids_one)+[1]*len(source_input_one)+[0]*(max_target_length-len(caption_ids_one)-len(source_input_one))
        #######只有b部分进行loss计算
        ###########第二次前向准备
        input_ids, labels, compress_input_ids = [], [], []
        caption_ids = tokenizer.encode(caption)[:149] + [eos_token_id]
        compress_input_ids = caption_ids + [tokenizer.pad_token_id]*(150-len(caption_ids))
        #压缩比只是一个范围,大概是在16附近,后面会修改成动态的qury,目前没这个精力了
        source_ids0 = tokenizer.encode(prompt)[:max_target_length-150]#注意,记忆部分(需要压缩到)是大于8的
         #           642                      如果不足max_target_length-8 就直接是原来长度
        source_ids = [tokenizer.pad_token_id]*8 + source_ids0
                      #这里面会替换成压缩后的promt
        
        input_ids += source_ids  + [tokenizer.pad_token_id]*(max_target_length-len(input_ids)-len(source_ids))#补到max_length
        labels =[0]*8+[1]*len(source_ids0)+[0]*(max_target_length-8-len(source_ids0)) 
        #2048
        if len(input_ids)==len(labels) and len(compress_input_ids)==150 and len(labels) == max_target_length:
            model_inputs["fist_input"].append(fist_input)
            model_inputs["first_attention_mask"].append(first_attention_mask)
            model_inputs["fisrt_lable"].append(fisrt_lable)
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["compress_input_ids"].append(compress_input_ids)
            model_inputs["compress_attention_mask"].append([False] * len(caption_ids) + [True] * (150-len(caption_ids)))
            model_inputs["labels"].append(labels)
        else:
            print('error', caption, len(input_ids), len(labels), len(compress_input_ids))
            raise IndexError
        
    return model_inputs

if __name__ == "__main__":
    # dataset = dataset.filter(lambda example: example["caption"] and example["prompt"])
    # with open('/mnt/data/user/lidehu/vae/qwen2_trainer-master/my_test.json', 'r') as file:
    #     data = json.load(file)
    
    # preprocess_supervised_dataset(data)
  
    
    dataset = load_dataset(
        "json",
        data_files='/mnt/data/user/lidehu/vae/qwen2_trainer-master/data_process/mt_0426_gbb_v2.json',
        split='train',
     
        # cache_dir='./tmp_hf_cache',
    )
    preprocess_function = preprocess_supervised_dataset
    column_names = dataset.column_names
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=8000,
        num_proc=32,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on train dataset",
    )

    dataset.save_to_disk('/mnt/data/user/lidehu/vae/qwen2_trainer-master/data_process/mem_align_0625.encode')

