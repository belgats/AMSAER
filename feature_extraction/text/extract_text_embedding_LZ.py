# *_*coding:utf-8 *_*
import os
import glob
import math
import pandas as pd
import numpy as np
import torch
import time
from tqdm import tqdm
import itertools
from transformers import AutoModel, BertTokenizer, AutoTokenizer, AutoModelForCausalLM # version: 4.5.1, pip install transformers
from transformers import GPT2Tokenizer, GPT2Model
import re
import argparse

from util import write_feature_to_csv, load_word2vec, load_glove, strip_accent
# import config
import sys
sys.path.append('../../')
import config

##################### English #####################
# word vectors
GLOVE = 'glove/glove.840B.300d.txt'
WORD2VEC = 'word2vec/GoogleNews-vectors-negative300.bin'

# transformers
BERT_BASE = 'bert-base-cased'
BERT_LARGE = 'bert-large-cased'
BERT_BASE_UNCASED = 'bert-base-uncased'
BERT_LARGE_UNCASED = 'bert-large-uncased'
ALBERT_BASE = 'albert-base-v2'
ALBERT_LARGE = 'albert-large-v2'
ALBERT_XXLARGE = 'albert-xxlarge-v2'
ROBERTA_BASE = 'roberta-base'
ROBERTA_LARGE = 'roberta-large'
ELECTRA_BASE = 'electra-base-discriminator'
ELECTRA_LARGE = 'electra-large-discriminator'
GPT = 'openai-gpt'
GPT2 = 'gpt2'
GPT2_MEDIUM = 'gpt2-medium'
GPT2_LARGE = 'gpt2-large'
GPT_NEO = 'gpt-neo-1.3B'
XLNET_BASE = 'xlnet-base-cased'
XLNET_LARGE = 'xlnet-large-cased'
T5_BASE = 't5-base'
T5_LARGE = 't5-large'
DEBERTA_BASE = 'deberta-base'
DEBERTA_LARGE = 'deberta-large'
DEBERTA_XLARGE = 'deberta-v2-xlarge'
DEBERTA_XXLARGE = 'deberta-v2-xxlarge'

##################### German #####################
BERT_GERMAN_CASED = 'bert-base-german-cased'
BERT_GERMAN_DBMDZ_CASED = 'bert-base-german-dbmdz-cased'
BERT_GERMAN_DBMDZ_UNCASED = 'bert-base-german-dbmdz-uncased'
##################### Arabic #####################
AraBERT_CASED = 'bert-base-german-cased'
MARBERTv2  = 'MARBERTv2'
JAIS_13B  = 'inception-mbzuai/jais-13b'

##################### Chinese #####################
BERT_BASE_CHINESE = 'bert-base-chinese' # https://huggingface.co/bert-base-chinese
ROBERTA_BASE_CHINESE = 'chinese-roberta-wwm-ext' # https://huggingface.co/hfl/chinese-roberta-wwm-ext
ROBERTA_LARGE_CHINESE = 'chinese-roberta-wwm-ext-large' # https://huggingface.co/hfl/chinese-roberta-wwm-ext-large
DEBERTA_LARGE_CHINESE = 'deberta-chinese-large' # https://huggingface.co/WENGSYX/Deberta-Chinese-Large
ELECTRA_SMALL_CHINESE = 'chinese-electra-180g-small' # https://huggingface.co/hfl/chinese-electra-180g-small-discriminator
ELECTRA_BASE_CHINESE  = 'chinese-electra-180g-base' # https://huggingface.co/hfl/chinese-electra-180g-base-discriminator
ELECTRA_LARGE_CHINESE = 'chinese-electra-180g-large' # https://huggingface.co/hfl/chinese-electra-180g-large-discriminator
XLNET_BASE_CHINESE = 'chinese-xlnet-base' # https://huggingface.co/hfl/chinese-xlnet-base
MACBERT_BASE_CHINESE = 'chinese-macbert-base' # https://huggingface.co/hfl/chinese-macbert-base
MACBERT_LARGE_CHINESE = 'chinese-macbert-large' # https://huggingface.co/hfl/chinese-macbert-large
PERT_BASE_CHINESE = 'chinese-pert-base' # https://huggingface.co/hfl/chinese-pert-base
PERT_LARGE_CHINESE = 'chinese-pert-large' # https://huggingface.co/hfl/chinese-pert-large
LERT_SMALL_CHINESE = 'chinese-lert-small' # https://huggingface.co/hfl/chinese-lert-small
LERT_BASE_CHINESE  = 'chinese-lert-base' # https://huggingface.co/hfl/chinese-lert-base
LERT_LARGE_CHINESE = 'chinese-lert-large' # https://huggingface.co/hfl/chinese-lert-large
GPT2_CHINESE = 'gpt2-chinese-cluecorpussmall' # https://huggingface.co/uer/gpt2-chinese-cluecorpussmall
CLIP_CHINESE = 'taiyi-clip-roberta-chinese' # https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese
WENZHONG_GPT2_CHINESE = 'wenzhong2-gpt2-chinese' # https://huggingface.co/IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese
ALBERT_TINY_CHINESE = 'albert_chinese_tiny' # https://huggingface.co/clue/albert_chinese_tiny
ALBERT_SMALL_CHINESE = 'albert_chinese_small' # https://huggingface.co/clue/albert_chinese_small
SIMBERT_BASE_CHINESE = 'simbert-base-chinese' # https://huggingface.co/WangZeJun/simbert-base-chinese

def extract_bert_embedding_chinese(model_name, trans_dir, save_dir, feature_level, layer_ids=None, gpu=6, overwrite=False):

    print('='*30 + f' Extracting "{model_name}" ' + '='*30)
    start_time = time.time()

    # layer ids
    if layer_ids is None:
        layer_ids = [-4, -3, -2, -1]
    else:
        assert isinstance(layer_ids, list)

    # save_dir
    dir_name = model_name if len(layer_ids) == 1 else f'{model_name}-{len(layer_ids)}'
    if feature_level == 'FRAME': save_dir = os.path.join(save_dir, dir_name+'-FRA')
    if feature_level == 'UTTERANCE': save_dir = os.path.join(save_dir, dir_name+'-UTT')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    elif overwrite or len(os.listdir(save_dir)) == 0:
        print(f'==> Warning: overwrite csv out dir "{dir_name}"!')
    else:
        raise Exception(f'==> Error: csv out dir "{dir_name}" already exists, set overwrite=TRUE if needed!')

    # load model and tokenizer: offline mode (load cached files)
    print('Loading pre-trained tokenizer and model...')
    model_dir = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'transformers/{model_name}')
    if model_name in [DEBERTA_LARGE_CHINESE, ALBERT_TINY_CHINESE, ALBERT_SMALL_CHINESE]:
        model = AutoModel.from_pretrained(model_dir)
        tokenizer = BertTokenizer.from_pretrained(model_dir, use_fast=False)
    elif model_name in [WENZHONG_GPT2_CHINESE]:
        model = GPT2Model.from_pretrained(model_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir, use_fast=False)
    else:
        model = AutoModel.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    device = torch.device(f'cuda:{gpu}')
    model.to(device)
    model.eval()

    # iterate videos
    feature_dim = -1
    df = pd.read_csv(trans_dir)
    for idx, row in df.iterrows():
        name = row['name']
        sentence = row['sentence']
        print(f'Processing {name} ({idx}/{len(df)})...')

        # extract embedding from sentences
        embeddings = []
        if pd.isna(sentence) == False and len(sentence) > 0:
            inputs = tokenizer(sentence, return_tensors='pt')
            # print (tokenizer.decode(inputs['input_ids'].cpu().numpy().tolist()[0])) # => 查看转换的token
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True).hidden_states # for new version 4.5.1
                outputs = torch.stack(outputs)[layer_ids].sum(dim=0) # sum => [batch, T, D=768]
                outputs = outputs.cpu().numpy() # (B, T, D)
                assert outputs.shape[0] == 1
                if model_name == XLNET_BASE_CHINESE:
                    embeddings = outputs[0, :-2]
                elif model_name == WENZHONG_GPT2_CHINESE:
                    embeddings = outputs[0, :]
                else:
                    embeddings = outputs[0, 1:-1] # (T, D)
                feature_dim = embeddings.shape[1]

        # align with label timestamp and write csv file
        print (f'feature dimension: {feature_dim}')
        csv_file = os.path.join(save_dir, f'{name}.npy')
        if feature_level == 'FRAME':
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((1, feature_dim))
            elif len(embeddings.shape) == 1:
                embeddings = embeddings[np.newaxis, :]
            np.save(csv_file, embeddings)
        else:
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((feature_dim, ))
            elif len(embeddings.shape) == 2:
                embeddings = np.mean(embeddings, axis=0)
            np.save(csv_file, embeddings)
    end_time = time.time()
    print(f'Total {len(df)} files done! Time used ({model_name}): {end_time - start_time:.1f}s.')

def extract_bert_embedding_arabic(model_name, trans_dir, save_dir, feature_level, layer_ids=None, gpu=-1, overwrite=False):

    print('='*30 + f' Extracting "{model_name}" ' + '='*30)
    start_time = time.time()

    # layer ids
    if layer_ids is None:
        layer_ids = [-4, -3, -2, -1]
    else:
        assert isinstance(layer_ids, list)

    # save_dir
    dir_name = model_name if len(layer_ids) == 1 else f'{model_name}-{len(layer_ids)}'
    if feature_level == 'FRAME': save_dir = os.path.join(save_dir, dir_name+'-FRA')
    if feature_level == 'UTTERANCE': save_dir = os.path.join(save_dir, dir_name+'-UTT')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    elif overwrite or len(os.listdir(save_dir)) == 0:
        print(f'==> Warning: overwrite csv out dir "{dir_name}"!')
    else:
        raise Exception(f'==> Error: csv out dir "{dir_name}" already exists, set overwrite=TRUE if needed!')

    # load model and tokenizer: offline mode (load cached files)
    print('Loading pre-trained tokenizer and model...')
    model_dir = os.path.join("/home/slasher/MER2023-Baseline/tools", f'transformers/{model_name}')
    #model = AutoModel.from_pretrained('/home/slasher/MER2023-Baseline/tools/transformers/MARBERTv2' )
    #tokenizer = BertTokenizer.from_pretrained('/home/slasher/MER2023-Baseline/tools/transformers/MARBERTv2/', use_fast=False)
    model =  AutoModelForCausalLM.from_pretrained('/home/slasher/MER2023-Baseline/tools/transformers/jais', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained('/home/slasher/MER2023-Baseline/tools/transformers/jais/', use_fast=False,  trust_remote_code=True)
    
    # 有 gpu 并且是 LLM，才会增加 half process
    if gpu != -1:
        model = model.half()

    # 有 gpu 才会放在cuda上
    if gpu != -1:
        torch.cuda.set_device(gpu)
        model.cuda()
    model.eval()
    
    #device = torch.device(f'cuda:{gpu}')
    #model.to(device)
    #model.eval()

    print('Calculate embeddings...')
    start, end = find_start_end_pos(tokenizer) # only preserve [start:end+1] tokens
    batch_pos, feature_dim = find_batchpos_embdim(tokenizer, model, gpu) # find batch pos

    # iterate videos
    feature_dim = -1
    df = pd.read_csv(trans_dir)
    for idx, row in df.iterrows():
        name = row['name']
        sentence = row['sentence']
        print(f'Processing {name} ({idx}/{len(df)})...')

        # extract embedding from sentences
        embeddings = []
        if pd.isna(sentence) == False and len(sentence) > 0:
            inputs = tokenizer(sentence, return_tensors='pt')
            # print (tokenizer.decode(inputs['input_ids'].cpu().numpy().tolist()[0])) # => 查看转换的token
            if gpu != -1: inputs = inputs.to('cuda')
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True).hidden_states # for new version 4.5.1
                outputs = torch.stack(outputs)[layer_ids].sum(dim=0) # sum => [batch, T, D=768]
                outputs = outputs.cpu().numpy() # (B, T, D)
                
                assert outputs.shape[0] == 1
                if batch_pos == 0:
                    embeddings = outputs[0, start:end]
                elif batch_pos == 1:
                    embeddings = outputs[start:end, 0]
                

        # align with label timestamp and write csv file
        print (f'feature dimension: {feature_dim}')
        csv_file = os.path.join(save_dir, f'{name}.npy')
        if feature_level == 'FRAME':
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((1, feature_dim))
            elif len(embeddings.shape) == 1:
                embeddings = embeddings[np.newaxis, :]
            np.save(csv_file, embeddings)
        else:
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((feature_dim, ))
            elif len(embeddings.shape) == 2:
                embeddings = np.mean(embeddings, axis=0)
            np.save(csv_file, embeddings)
    end_time = time.time()
    print(f'Total {len(df)} files done! Time used ({model_name}): {end_time - start_time:.1f}s.')




 
# 自动删除无意义token对应的特征
def find_start_end_pos(tokenizer):
    sentence = 'الطقس جميل جدا اليوم' # 句子中没有空格
    input_ids = tokenizer(sentence, return_tensors='pt')['input_ids'][0]
    start, end = None, None

    # find start, must in range [0, 1, 2]
    for start in range(0, 3, 1):
        # 因为decode有时会出现空格，因此我们显示的时候把这部分信息去掉看看
        outputs = tokenizer.decode(input_ids[start:]).replace(' ', '')
        if outputs == sentence:
            print (f'start: {start};  end: {end}')
            return start, None

        if outputs.startswith(sentence):
            break
   
    # find end, must in range [-1, -2]
    for end in range(-1, -3, -1):
        outputs = tokenizer.decode(input_ids[start:end]).replace(' ', '')
        if outputs == sentence:
            break
    
    assert tokenizer.decode(input_ids[start:end]).replace(' ', '') == sentence
    print (f'start: {start};  end: {end}')
    return start, end


# 找到 batch_pos and feature_dim
def find_batchpos_embdim(tokenizer, model, gpu):
    sentence = 'الطقس جميل جدا اليوم'
    inputs = tokenizer(sentence, return_tensors='pt')
    if gpu != -1: inputs = inputs.to('cuda')

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True).hidden_states # for new version 4.5.1
        outputs = torch.stack(outputs)[[-1]].sum(dim=0) # sum => [batch, T, D=768]
        outputs = outputs.cpu().numpy() # (B, T, D) or (T, B, D)
        batch_pos = None
        if outputs.shape[0] == 1:
            batch_pos = 0
        if outputs.shape[1] == 1:
            batch_pos = 1
        assert batch_pos in [0, 1]
        feature_dim = outputs.shape[2]
    print (f'batch_pos:{batch_pos}, feature_dim:{feature_dim}')
    return batch_pos, feature_dim


def main(model_name, feature_level, trans_dir, save_dir, layer_ids, overwrite, gpu, batch_size):
    
    if model_name.find('chinese') != -1:
        extract_bert_embedding_chinese(model_name=model_name, trans_dir=trans_dir, save_dir=save_dir,
                                       layer_ids=layer_ids, feature_level=feature_level, gpu=gpu, overwrite=overwrite)

    else: # for english model from transformer
        extract_bert_embedding_arabic(model_name=model_name, trans_dir=trans_dir, save_dir=save_dir,
                                       layer_ids=layer_ids,  feature_level=feature_level,
                                       gpu=gpu,  overwrite=overwrite) # batch_size=batch_size,


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run.')
    # choose one from supported models. Note: some transformers listed above are not available due to various errors!
    parser.add_argument('--model_name', type=str, help='name of pretrained model')
    parser.add_argument('--gpu', type=str, default='1', help='gpu id')
    parser.add_argument('--overwrite', action='store_true', default=True, help='whether overwrite existed feature folder.')
    parser.add_argument('--dataset', type=str, help='input dataset')
    parser.add_argument('--feature_level', type=str, default='UTTERANCE', choices=['UTTERANCE', 'FRAME'], help='output types')
    args = parser.parse_args()
    
    trans_dir = '/home/slasher/MER2023-Baseline/dataset-process/transcription.csv' #config.PATH_TO_TRANSCRIPTIONS[args.dataset] # directory of transcriptions
    save_dir ='/home/slasher/MER2023-Baseline/dataset-process/features' #config.PATH_TO_FEATURES[args.dataset] # directory used to store features
    layer_ids = [-4, -3, -2, -1] # hidden states of selected layers will be used to obtain the embedding, only for transformers

    main(args.model_name, args.feature_level, trans_dir, save_dir, layer_ids=layer_ids, overwrite=args.overwrite, gpu=args.gpu, batch_size=32)
