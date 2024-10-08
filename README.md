# Distile_compres_of_llm
<div align="center">
Author: lidehu 2201210265@stu.pku.edu.cn
    在soul app 实习期间的工作
</div>

## 动机与做法： 
- 数字分身大模型需求:模型对话前需要把用户的各种信息放在promt,promt长度过长(信息密度低,特别是历史聊天记录),限制了实际使用.用小模型压缩promt,大模型接收压缩后的promt是一个可行的解决方案.
- 前置条件:给出用户历史a,llm在后续回答中是可以cue到用户历史记忆的.比如在很长的一段历史里写到用户喜欢吃葡萄,那么llm在一些回答里面是知道用户喜欢吃葡萄,并给出一些贴合的回答.
- 思路:更换promt为压缩的promt,不改变后续预测行为,蒸馏大模型内在压缩机制.
- 训练一阶段:压缩还原任务,训练压缩网络.
- 做法一:压缩还原部分照旧,对齐阶段做法如下:样本分为俩个部分:(a,b）,俩次前向:第一次llm(a,b)取出b部分输出的概率分布,第二次:llm( 小模型(a) ,b),取出b部分输出的概率分布,计算俩者的kl散度.
- 做法二:离线操作,根据用户历史数据,当前聊天a,用数字分身基座模型输出回复b.得到数据(用户历史信息+a,b)，去预测b.

## 模型结构
- 小模型为千问0.5b+q_former.目前压缩比16,效果很好,遥远的记忆也可以cue到,应该还有很多压缩空间,不过目前够用了.
- 先训练压缩还原任务,然后再蒸馏.然后打开小模型和llm(llm找一个老师,拷贝自己一份)
  

- ### Pretrained Model Weight
  本人自我驱动独立完成此份工作,从idea构建到实施,但训练模型使用了公司业务数据,因此不能开源权重.
- ### Training

    Start training by run
    ```
    ############注意.代码仅供参考,毕竟大家数据差异挺大.
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
