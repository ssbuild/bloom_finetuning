# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/16 16:03

from deep_training.utils.hf import register_transformer_model, register_transformer_config, \
    register_transformer_tokenizer
from transformers import AutoModelForCausalLM
from deep_training.nlp.models.rellama.modeling_llama import LlamaForCausalLM
from deep_training.zoo.model_zoo.bluelm.llm_model import MyBlueLMForCausalLM,BlueLMTokenizer,BlueLMConfig
from deep_training.zoo.model_zoo.xverse.llm_model import MyXverseForCausalLM,XverseConfig
from deep_training.zoo.model_zoo.internlm.llm_model import MyInternLMForCausalLM,InternLMTokenizer,InternLMConfig
from deep_training.zoo.model_zoo.skywork.llm_model import MySkyworkForCausalLM,SkyworkConfig,SkyworkTokenizer
# from deep_training.zoo.model_zoo.yi.llm_model import MyYiForCausalLM,YiConfig,YiTokenizer

__all__ = [
    "module_setup"
]

def module_setup():
    # 导入模型
    #register_transformer_config(XverseConfig)
    register_transformer_model(LlamaForCausalLM, AutoModelForCausalLM)


    register_transformer_config(BlueLMConfig)
    register_transformer_model(MyBlueLMForCausalLM, AutoModelForCausalLM)
    register_transformer_tokenizer(BlueLMConfig,BlueLMTokenizer,BlueLMTokenizer)

    register_transformer_config(XverseConfig)
    register_transformer_model(MyXverseForCausalLM, AutoModelForCausalLM)

    register_transformer_config(InternLMConfig)
    register_transformer_model(MyInternLMForCausalLM, AutoModelForCausalLM)
    register_transformer_tokenizer(InternLMConfig, InternLMTokenizer, InternLMTokenizer)

    register_transformer_config(SkyworkConfig)
    register_transformer_model(MySkyworkForCausalLM, AutoModelForCausalLM)
    register_transformer_tokenizer(SkyworkConfig,SkyworkTokenizer,SkyworkTokenizer)

    # register_transformer_config(YiConfig)
    # register_transformer_model(MyYiForCausalLM, AutoModelForCausalLM)
    # register_transformer_tokenizer(YiConfig, YiTokenizer, YiTokenizer)