# @Time    : 2023/1/22 16:22
# @Author  : tk
# @FileName: data_utils.py
import glob
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from functools import cache
import copy
import json
import random
import typing
import numpy as np
import torch
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, TrainingArgumentsHF, \
    TrainingArgumentsCL, DataArguments, TrainingArgumentsAC
from aigc_zoo.model_zoo.llm.llm_model import PetlArguments,LoraConfig,PromptArguments
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from transformers import PreTrainedTokenizer, HfArgumentParser, PretrainedConfig
from data_processer import DataStrategy, TokenIdsMaker, build_template, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, \
    DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN
from config import *
from module_setup import module_setup


module_setup()

data_conf = {
    'strategy': DataStrategy.tunction,  # 数据策略选项
    DataStrategy.tunction: {
        'sup': True, # 是否监督模式
    },

    DataStrategy.slidding: {
        'stride': int(train_info_args['max_seq_length'] / 3 * 2),
        'sup': True, # 是否监督模式
        "src_max_length": train_info_args['max_seq_length'] - 10,
        "dst_max_length": None,
    }

}



def preprocess(text):
  return text

def postprocess(text):
  return text


class NN_DataHelper(DataHelper):
    index = 1

    def __init__(self, *args, **kwargs):
        super(NN_DataHelper, self).__init__(*args, **kwargs)
        assert data_conf[DataStrategy.slidding]['stride'] > 0

    def load_tokenizer_and_config(self, *args, **kwargs):
        ret = super().load_tokenizer_and_config(*args, **kwargs)
        self._preprocess_tokenizer_config()
        return ret

    def _preprocess_tokenizer_config(self):
        model_args = self.model_args
        tokenizer = self.tokenizer
        config = self.config



        if "llama" in model_args.model_type.lower():
            special_tokens_dict = dict()
            if tokenizer.pad_token is None:
                special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
            if tokenizer.eos_token is None:
                special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
            if tokenizer.bos_token is None:
                special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
            if tokenizer.unk_token is None:
                special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

            _ = tokenizer.add_special_tokens(special_tokens_dict)

            # tokenizer.add_special_tokens({
            #     "eos_token": DEFAULT_EOS_TOKEN,
            #     "bos_token": DEFAULT_BOS_TOKEN,
            #     "unk_token": DEFAULT_UNK_TOKEN,
            # })
            # if tokenizer.pad_token_id is None or tokenizer.pad_token_id == -1:
            #     tokenizer.pad_token_id = tokenizer.eos_token_id

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({
                "pad_token": tokenizer.eos_token,
            })
        if config.pad_token_id is None or config.pad_token_id == -1:
            config.pad_token_id = tokenizer.eos_token_id



        if config.decoder_start_token_id is None:
            config.decoder_start_token_id = config.bos_token_id

        if config.decoder_start_token_id != tokenizer.bos_token_id:
            print('*' * 30, 'config.decoder_start_token_id != tokenizer.bos_token_id !!!')

        assert config.decoder_start_token_id == config.bos_token_id

    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        self.index += 1

        tokenizer: PreTrainedTokenizer
        config = self.config
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer

        examples = data

        strategy = data_conf['strategy']
        if strategy == DataStrategy.tunction:
            ds = TokenIdsMaker.tunction(tokenizer, config=config, max_seq_length=max_seq_length, examples=examples,
                                        **data_conf[strategy])
        elif strategy == DataStrategy.slidding:
            ds = TokenIdsMaker.slidding(tokenizer, config=config, max_seq_length=max_seq_length, examples=examples,
                                        **data_conf[strategy])

        else:
            raise ValueError('Invalid strategy', strategy)
        if not ds:
            return None

        if self.index < 3:
            print(ds[0])
        return ds

    def _get_paragraph(self,lines):
        D = []
        for line_id, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            paragraph = jd['paragraph']
            if line_id < 10:
                print(paragraph)

            prefix = jd.get('p', '')
            paragraph = [(preprocess(session['q']),
                          preprocess('\n'.join(session['a'])) if isinstance(session['a'], list) else preprocess(
                              session['a']))
                         for session in paragraph]
            sub = []
            # 自行做模板
            for (q,a) in paragraph:
                assert len(a), ValueError('answer cannot empty')
                sub.append((q, a))
            D.append((prefix, copy.deepcopy(sub)))
            sub.clear()
        return D

    def _get_messages(self,lines):
        D = []
        for line_id, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            conversations = jd['conversations']
            if line_id < 10:
                print(conversations)

            paragraph = []
            prefix = ''
            pair = [None,None]
            for m in conversations:
                if m["from"] == 'user':
                    pair[0] = preprocess(m["value"])
                elif m["from"] == 'assistant':
                    pair[1] = preprocess(m["value"])
                elif m["from"] == 'system':
                    prefix = preprocess(m["value"])
                if pair[0] is not None and pair[1] is not None:
                    paragraph.append(tuple(pair))
                    pair[0],pair[1] = None,None

            sub = []
            # 自行做模板
            for (q, a) in paragraph:
                assert len(a), ValueError('answer cannot empty')
                sub.append((q, a))
            D.append((prefix, copy.deepcopy(sub)))
            sub.clear()
        return D
    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        D = []
        files = sum([glob.glob(file) for file in files], [])
        for file in files:
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                lines = f.readlines()
            is_new = False
            if len(lines) > 0:
                is_new = 'conversations' in json.loads(lines[0])
            if is_new:
                D.extend(self._get_messages(lines))
            else:
                D.extend(self._get_paragraph(lines))
        return D

    def collate_fn(self, batch):
        o = {}
        for i, b in enumerate(batch):
            if i == 0:
                for k in b:
                    o[k] = [torch.tensor(b[k])]
            else:
                for k in b:
                    o[k].append(torch.tensor(b[k]))
        for k in o:
            o[k] = torch.stack(o[k])

        maxlen = torch.max(o.pop('seqlen'))
        o['input_ids'] = o['input_ids'][:, :maxlen]
        o['attention_mask'] = o['attention_mask'][:, :maxlen]
        o['labels'] = o['labels'][:, :maxlen].long()
        return o

    def make_dataset_all(self):
        data_args = self.data_args
        # schema for arrow parquet
        schema = {
            "input_ids": "int32_list",
            "attention_mask": "int32_list",
            "labels": "int32_list",
            "seqlen": "int32_list",
        }
        # 缓存数据集
        if data_args.do_train:
            self.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True, mode='train',
                                        schema=schema)
        if data_args.do_eval:
            self.make_dataset_with_args(data_args.eval_file, mode='eval', schema=schema)
        if data_args.do_test:
            self.make_dataset_with_args(data_args.test_file, mode='test', schema=schema)

        # 记录缓存文件
        with open(os.path.join(data_args.output_dir, 'intermediate_file_index.json'), mode='w',
                  encoding='utf-8') as f:
            f.write(json.dumps({
                "train_files": self.train_files,
                "eval_files": self.eval_files,
                "test_files": self.test_files,
            }, ensure_ascii=False))

    @cache
    def load_dataset_files(self):
        data_args = self.data_args
        filename = os.path.join(data_args.output_dir, 'intermediate_file_index.json')
        assert os.path.exists(filename), 'make you dataset firstly'
        with open(filename, mode='r', encoding='utf-8') as f:
            return json.loads(f.read())

if __name__ == '__main__':

    if global_args["trainer_backend"] == "hf":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsHF, DataArguments, PetlArguments, PromptArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args, prompt_args = parser.parse_dict(train_info_args,
                                                                                         allow_extra_keys=True, )
    elif global_args["trainer_backend"] == "pl":
        parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, PetlArguments, PromptArguments))
        model_args, training_args, data_args, _, _ = parser.parse_dict(train_info_args)
    elif global_args["trainer_backend"] == "cl":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsCL, DataArguments, PetlArguments, PromptArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args, prompt_args = parser.parse_dict(train_info_args,
                                                                                         allow_extra_keys=True, )
    else:
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsAC, DataArguments, PetlArguments, PromptArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args, prompt_args = parser.parse_dict(train_info_args,
                                                                                         allow_extra_keys=True, )


    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_kwargs={"torch_dtype": torch.float16})

    # 缓存数据集
    print(f'to make dataset is overwrite_cache {data_args.overwrite_cache}')
    dataHelper.make_dataset_all()

    print('make dataset complete!')
    print('check data !')
    dataset = dataHelper.load_sequential_sampler(dataHelper.load_dataset_files()["train_files"],
                                                 with_load_memory=data_args.data_backend == 'record',
                                                 batch_size=1,
                                                 collate_fn=dataHelper.collate_fn)

    print('total', len(dataset))
    for i, d in enumerate(dataset):
        print(d)
        if i > 3:
            break