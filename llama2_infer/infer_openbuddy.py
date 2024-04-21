# coding=utf8
# @Time    : 2023/9/2 22:32
# @Author  : tk
# @FileName: tiger_infer
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser
from data_utils import config_args, NN_DataHelper, get_deepspeed_config
from deep_training.zoo.model_zoo.llm.llm_model import MyTransformer
from deep_training.zoo.utils.llm_generate import Generate
from deep_training.zoo.model_zoo.chatglm2.llm_model import RotaryNtkScaledArguments,RotaryLinearScaledArguments # aigc-zoo 0.1.20

deep_config = get_deepspeed_config()


sys_prompt = """You are a helpful, respectful and honest INTP-T AI Assistant named Buddy. You are talking to a human User.
Always answer as helpfully and logically as possible, while being safe. Your answers should not include any harmful, political, religious, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
You like to use emojis. You can speak fluently in many languages, for example: English, Chinese.
You cannot access the internet, but you have vast knowledge, cutoff: 2021-09.
You are trained by OpenBuddy team, (https://openbuddy.ai, https://github.com/OpenBuddy/OpenBuddy), you are based on LLaMA and Falcon transformers model, not related to GPT or OpenAI.

User: Hi.
Assistant: Hi, I'm Buddy, your AI assistant. How can I help you today?😊
"""




if __name__ == '__main__':


    parser = HfArgumentParser((ModelArguments,))
    (model_args,)  = parser.parse_dict(config_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args)
    tokenizer, config, _,_= dataHelper.load_tokenizer_and_config()

    enable_ntk = False
    rope_args = None
    if enable_ntk and config.model_type == 'llama':
        rope_args = RotaryNtkScaledArguments(name='rotary_emb',max_position_embeddings=2048, alpha=4)  # 扩展 8k
        # rope_args = RotaryLinearScaledArguments(name='rotary_emb',max_position_embeddings=2048, scale=4) # 扩展 8k


    pl_model = MyTransformer(config=config, model_args=model_args,
                             torch_dtype=torch.float16,
                             rope_args=rope_args,
                             device_map="auto",)
    model = pl_model.get_llm_model()
    model = model.eval()
    model.half()
    if hasattr(model,'quantize'):
        # 支持llama llama2量化
        if not model.quantized:
            # 按需修改，目前只支持 4/8 bit 量化 ， 可以保存量化模型
            model.half().quantize(4).cuda()
            # 保存量化权重
            # model.save_pretrained('llama2-7b-chat-int4',max_shard_size="2GB")
            # exit(0)
        else:
            # 已经量化
            model.half().cuda()
    else:
        model.half().cuda()

    generation_config = {
        # "do_sample": True,
        "max_new_tokens": 100,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        # "repetition_penalty": 1.1,
    }
    text_list = [
        "Write a poem about yourself.",
        "写一个诗歌，关于冬天",
        "晚上睡不着应该怎么办",
        "从南京到上海的路线",
    ]

    for input in text_list:
        input_text = sys_prompt + "\n\nUser: {}\nAssistant:".format(input)
        input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=512)
        output = model.generate(input_ids=input_ids.to('cuda'), **generation_config)
        answer = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)

        print('input', input)
        print('output', answer)

