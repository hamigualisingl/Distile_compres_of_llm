from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
from models.adapt import Adaptor
from transformers import AutoModelForCausalLM
import torch
import torch.nn.functional as F

import torch.nn as nn

class Qwen2ForCausalLMModify(Qwen2ForCausalLM):
    is_parallelizable = False
    def __init__(self, config):
        super().__init__(config)
        self.adaptor = Adaptor(
            num_queries=8,
            input_len=150
        ).to(torch.bfloat16)
        self.post_init()                            #千问0.5b
        self.encoder = AutoModelForCausalLM.from_pretrained('', torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.load_adaptor()

    def load_adaptor(self):
        state_dict = torch.load(" ")
        self.adaptor.load_state_dict(state_dict)

    def forward(
        self,
        fist_input=None,
        first_attention_mask=None,
        fisrt_lable=None,
        input_ids = None,
        attention_mask = None,
        compress_input_ids = None,
        compress_attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        """ """
      
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        ##############第一次前向,原promt
        with torch.no_grad():
            inputs_embeds_fist = self.model.embed_tokens(fist_input)
            outputs_fist = self.model(
            attention_mask=first_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds_fist,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
            hidden_states_fist = outputs_fist[0]
            logits_first = self.lm_head(hidden_states_fist)
            logits_first = logits_first.float()
            bs, seq_len, dim = logits_first.size()
            first_one_indices = (fisrt_lable == 1).float().argmax(dim=1)#位置正确取出正文预测开始的位置
            masked_logits = []
            for i in range(bs):
                idx = first_one_indices[i].item()
                masked_logits.append(logits_first[i, idx:])
            max_len = seq_len-8#
            padded_logits = []
            for logit in masked_logits:
                pad_size = max_len - logit.size(0)
                padded_logits.append(F.pad(logit, (0, 0, 0, pad_size), value=0.1))
            first_logits = torch.stack(padded_logits)#得到了最初的输出
         ##############第二次前向,更换prompt为压缩后的prompt
        compress_hidden_states = self.encoder.forward(compress_input_ids, output_hidden_states=True)
        ######## 压缩记忆,先由千问小模型提取特征,然后用q_former压缩
        inputs_embeds1 = self.adaptor.forward(x=compress_hidden_states.hidden_states[-1], key_padding_mask=compress_attention_mask)
        # 第二次前向,记忆替换为压缩后的记忆
        inputs_embeds0 = self.model.embed_tokens(input_ids)
        
        inputs_embeds = torch.concat([inputs_embeds1, inputs_embeds0[:,8:,:]], dim=-2)
        # bs,n,dim
        outputs = self.model(
            # input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()[:,8:,:]#前8个是压缩的token
        labels=labels[..., 8:]#后面有填充部分要消除
        ############################################蒸馏部分
        sum=labels.sum()
        student_probs = F.log_softmax(logits, dim=-1)  # 使用 log_softmax 以匹配 KLDivLoss 的输入要求
        teacher_probs = F.softmax(first_logits, dim=-1)
        soft_loss = nn.KLDivLoss(reduction='none') 
        mask = labels.unsqueeze(-1).expand_as(student_probs) 
        distillation_loss = soft_loss(student_probs, teacher_probs)
        distillation_loss = distillation_loss * mask
        distillation_loss_sum = distillation_loss.sum(dim=-1)

        loss = distillation_loss_sum.sum()/sum
        ############
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
