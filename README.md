# Distile_compres_of_llm
<div align="center">
Author: lidehu 2201210265@stu.pku.edu.cn
    在soul app 实习期间的工作
</div>

## 动机与做法： 
- 数字分身大模型需求:模型对话前需要把用户的各种信息放在promt,promt长度过长,限制了实际使用.用小模型压缩promt,大模型接收压缩后的promt是一个可行的解决方案.
- 普通的压缩还原网络分为俩个阶段:压缩还原-训练小模型压缩promt,llm还原promt(冻结);对齐-根据压缩网络输出内容llm预测下文而非还原promt,小模型和llm均打开训练.(类似于pretrian和sft的关系)
- 压缩还原方案存在如下问题:对齐阶段需要高质量对齐数据,需要压缩部分和llm预测部分要有很强的联系,目前业务数据对齐很弱.二是还原了不必要的内容,造成浪费,比如:今天天气真的太好呀，哈哈哈.其中'的'和'哈哈',没有还原的必要
- 个人理解:和人类一样,llm在阅读了大量前置文段后,会高度浓缩的总结起来,后面遇到有联系的事物便会勾连起来-灵光乍现.理解难度小于生成,小模型压缩具有可行性.
- 思路:更换promt为压缩的promt,不改变后续预测行为,蒸馏大模型内在压缩机制.
- 做法:样本分为俩个部分:(a,b）,俩次前向:第一次llm(a,b)取出b部分输出的概率分布,第二次:llm( 小模型(a) ,b),取出b部分输出的概率分布,计算俩者的kl散度.
- 插播:有愿意一起写论文的朋友吗,共一或者你通讯(秋招太忙了,一个人干不来)

## 模型结构
- 小模型为千问0.5b+q_former.目前压缩比16,效果很好,遥远的记忆也可以cue到,应该还有很多压缩空间,不过目前够用了.
- 建议先训练q_former.

- ### Pretrained Model Weight
  本人自我驱动独立完成此份工作,从idea构建到实施,但训练模型使用了公司业务数据,因此不能开源权重.
- ### Training

    Start training by run
    ```
    ############注意.因为大家数据组织形式不同,所以这边只是大概对示例
    cd /qwen2_trainer;
    pip install transformers -U;
    deepspeed --hostfile /etc/mpi/hostfile  --master_port=$(shuf -n 1 -i 10000-65535) \  #注意,我是使用阿里云的dlc训练,里面会自动分配通讯端口,主节点ip地址,根据自身情况酌情更改
    trainer/train_compress.py \
    --deepspeed ./conf/deepspeed.json \
    --do_train \
    --model_name_or_path /mnt/data/ \
    --tokenizer_path /mnt/data/ \
    --data_path /mnt/data/ \
    --overwrite_cache \
    --preprocessing_num_workers 32 \
    --output_dir . \
    --max_source_length  \
    --max_target_length  \
    --lr_scheduler_type cosine \
    --num_train_epochs  \
    --warmup_steps 0 \
    --save_steps  \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --learning_rate 5e-5 \
    --bf16
    ```

## Acknowledgement

感谢soul app自然语言算法组.
