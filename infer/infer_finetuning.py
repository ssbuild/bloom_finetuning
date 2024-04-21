# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser, AutoConfig
from data_utils import config_args, NN_DataHelper, get_deepspeed_config,build_template
from deep_training.zoo.model_zoo.llm.llm_model import MyTransformer
from deep_training.zoo.utils.llm_generate import Generate

deep_config = get_deepspeed_config()


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(config_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args)
    tokenizer, _, _,_= dataHelper.load_tokenizer_and_config()
    

    config = AutoConfig.from_pretrained('../scripts/best_ckpt', trust_remote_code=True,local_files_only=True,)
    pl_model = MyTransformer(config=config, model_args=model_args,
                             torch_dtype=config.torch_dtype,
                             trust_remote_code=True,
                             local_files_only=True,
                             )

    # deepspeed 权重使用转换脚本命令
    # 一般根据时间排序选最新的权重文件夹
    # cd best_ckpt/last
    # python zero_to_fp32.py . ../last.ckpt

    train_weight = '../scripts/best_ckpt/last.ckpt'
    pl_model.load_sft_weight(train_weight,strict=False)

    # 保存hf权重
    # config.save_pretrained('convert/')

    # 保存sft p-tuning-v2 权重
    #  pl_model.save_sft_weight('convert/pytorch_model_sft_ptv2.bin')

    # 保存sft权重
    # pl_model.save_sft_weight('convert/pytorch_model_sft.bin')

    model = pl_model.get_llm_model()

    model.eval().half().cuda()

    text_list = ["写一个诗歌，关于冬天",
                 "晚上睡不着应该怎么办",
                 "从南京到上海的路线",
                 ]
    for input in text_list:
        response = Generate.generate(model, query=build_template(input), tokenizer=tokenizer, max_length=512,
                                     eos_token_id=config.eos_token_id,
                                     do_sample=False, top_p=0.7, temperature=0.95, )
        print('input', input)
        print('output', response)