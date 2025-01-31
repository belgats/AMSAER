import re
import os
import sys
import time
import copy
import tqdm
import glob
import json
import math
import scipy
import shutil
import random
import pickle
import argparse
import numpy as np
import pandas as pd
import multiprocessing

import sklearn
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model.mult import MULT
from model.lmf import LMF
from model.tfn import TFN
from model.encoders import MLPEncoder,LSTMEncoder
import config
from toolkit.data import get_datasets

#emoss = ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise'] #  
 
sentss = ['neutral', 'positive', 'negative']  

emo2idx, idx2emo = {}, {}
sent2idx, idx2sent = {}, {}
for ii, emo in enumerate(emoss): emo2idx[emo] = ii
for ii, emo in enumerate(emoss): idx2emo[ii] = emo
for ii, sent in enumerate(sentss): sent2idx[sent] = ii
for ii, sent in enumerate(sentss): idx2sent[ii] = sent

 
tsks = ['easy', 'hard']
tsks2idx, idx2tsks = {}, {}
for ii, ts in enumerate(tsks): tsks2idx[ts] = ii
for ii, ts in enumerate(tsks): idx2tsks[ii] = ts

########################################################
############## multiprocess read features ##############
########################################################
def func_read_one(argv=None, feature_root=None, name=None):

    feature_root, name = argv
    feature_dir = glob.glob(os.path.join(feature_root, name+'.npy'))
    
    #print(name)
    if len(feature_dir) != 1:
       raise AssertionError(f'Expected {name} element in feature_dir, but not found!')

    feature_path = feature_dir[0]

    feature = []
    if feature_path.endswith('.npy'):
        single_feature = np.load(feature_path)
        single_feature = single_feature.squeeze()
        feature.append(single_feature)
    else:
        facenames = os.listdir(feature_path) 
        for facename in sorted(facenames):
            facefeat = np.load(os.path.join(feature_path, facename))
            feature.append(facefeat)

    single_feature = np.array(feature).squeeze()
    if len(single_feature) == 0:
        print ('feature has errors!!')
    elif len(single_feature.shape) == 2:
        single_feature =  np.mean(single_feature, axis=0)
    return single_feature
    
def read_data_multiprocess(label_path, feature_root, task='whole', data_type='train', debug=False):

    ## gain (names, labels)
    names, labels = [], []
    assert task in  ['emo', 'aro', 'val','sent','Task', 'whole']
    assert data_type in ['train', 'test1', 'test2', 'test3']
    if data_type == 'train': corpus = np.load(label_path, allow_pickle=True)['train_corpus'].tolist()
    if data_type == 'test1': corpus = np.load(label_path, allow_pickle=True)['test1_corpus'].tolist()
    if data_type == 'test2': corpus = np.load(label_path, allow_pickle=True)['test2_corpus'].tolist()
    if data_type == 'test3': corpus = np.load(label_path, allow_pickle=True)['test3_corpus'].tolist()
    for name in corpus:
        names.append(name)
        if task in ['aro', 'val']:
            labels.append(corpus[name][task])
        if task == 'emo':
            labels.append(emo2idx[corpus[name]['emo']])
        if task == 'sent':
            labels.append(sent2idx[corpus[name]['sent']])
        if task == 'Task':
            labels.append(tsks2idx[corpus[name]['Task']])
        if task == 'whole':
            corpus[name]['emo'] = emo2idx[corpus[name]['emo']]
            corpus[name]['val'] = corpus[name]['val']
            corpus[name]['sent'] = sent2idx[corpus[name]['sent']]
            corpus[name]['Task'] = tsks2idx[corpus[name]['Task']]
            labels.append(corpus[name])
    ## ============= for debug =============
    if debug: 
        names = names[:100]
        labels = labels[:100]
    ## =====================================

    ## names => features
    params = []
    for ii, name in tqdm.tqdm(enumerate(names)):
        params.append((feature_root, name))

    features = []
    #with multiprocessing.Pool(processes=1) as pool:
    #    features = list(tqdm.tqdm(pool.imap(func_read_one, params), total=len(params)))
    for param in tqdm.tqdm(params, total=len(params)):
        feature = func_read_one(param)
        features.append(feature)
    feature_dim = np.array(features).shape[1] 
    #feature_dim = np.array(features).shape[-1]

    ## save (names, features)
    print (f'Input feature {feature_root} ===> dim is {feature_dim}')
    assert len(names) == len(features), f'Error: len(names) != len(features)'
    name2feats, name2labels = {}, {}
    for ii in range(len(names)):
        name2feats[names[ii]]  = features[ii]
        name2labels[names[ii]] = labels[ii]
    return name2feats, name2labels, feature_dim

########################################################
##################### data loader ######################
########################################################
 
def read_names_labels( label_path, task='sent', data_type, debug=False):
        names, labels = [], []
        assert task in  ['emo', 'aro', 'val','sent','Task', 'whole']
        assert data_type in ['train', 'test1', 'test2', 'test3']
        if data_type == 'train': corpus = np.load(label_path, allow_pickle=True)['train_corpus'].tolist()
        if data_type == 'test1': corpus = np.load(label_path, allow_pickle=True)['test1_corpus'].tolist()
        if data_type == 'test2': corpus = np.load(label_path, allow_pickle=True)['test2_corpus'].tolist()
        if data_type == 'test3': corpus = np.load(label_path, allow_pickle=True)['test3_corpus'].tolist()
       
        for name in corpus:
            names.append(name)
            if task in ['aro', 'val']:
                labels.append(corpus[name][task])
            if task == 'emo':
                labels.append(emo2idx[corpus[name]['emo']])
            if task == 'sent':
                corpus[name]['visual'] = corpus[name]['visual']
                labels.append(sent2idx[corpus[name]['sent']])
            if task == 'Task':
                labels.append(tsks2idx[corpus[name]['Task']])
            if task == 'whole':
                corpus[name]['emo'] = emo2idx[corpus[name]['emo']]
                corpus[name]['visual'] = corpus[name]['visual']
                corpus[name]['val'] = corpus[name]['val']
                corpus[name]['sent'] = sent2idx[corpus[name]['sent']]
                corpus[name]['Task'] = tsks2idx[corpus[name]['Task']]
 
                labels.append(corpus[name])
        for ii, label in enumerate(labels):
            sent = label['sent']
            if task == 'whole':
               emo = label['emo']
               val = label['val']
               labels[ii] = {'emo': emo, 'val': val, 'sent': sent,'visual': label['visual']}
            else:
               labels[ii] = {'sent': sent }

            if args.visual == True:
               visual = label['visual']
               labels[ii] = {'emo': emo, 'val': val, 'sent': sent, 'visual': visual}
        # for debug
        if debug: 
            names = names[:100]
            labels = labels[:100]
        return names, labels
## for five-fold cross-validation on Train&Val
def get_loaders(args, config):
 
    data_type = 'train'
    names, labels = read_names_labels(label_path='/home/slasher/AMSAER-Baseline/dataset-process/label-AMSAER-visual.npz', data_type= 'train', debug=False)
    print (f'{data_type}: sample number {len(names)}')
    train_dataset = get_datasets(args, names, labels)
    # gain indices for cross-validation
    whole_folder = []
    whole_num = len(names)
    indices = np.arange(whole_num)
    random.shuffle(indices)

    # split indices into five-fold
    num_folder = args.num_folder
    each_folder_num = int(whole_num / num_folder)
    for ii in range(num_folder-1):
        each_folder = indices[each_folder_num*ii: each_folder_num*(ii+1)]
        whole_folder.append(each_folder)
    each_folder = indices[each_folder_num*(num_folder-1):]
    whole_folder.append(each_folder)
    assert len(whole_folder) == num_folder
    assert sum([len(each) for each in whole_folder if 1==1]) == whole_num

    ## split into train/eval
    train_eval_idxs = []
    for ii in range(num_folder):
        eval_idxs = whole_folder[ii]
        train_idxs = []
        for jj in range(num_folder):
            if jj != ii: train_idxs.extend(whole_folder[jj])
        train_eval_idxs.append([train_idxs, eval_idxs])

    ## gain train and eval loaders
    train_loaders = []
    eval_loaders = []
 
    for ii in range(len(train_eval_idxs)):# 
        train_idxs = train_eval_idxs[ii][0]
        eval_idxs  = train_eval_idxs[ii][1]
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=SubsetRandomSampler(train_idxs),
                                  num_workers=args.num_workers,
                                  collate_fn=train_dataset.collater_visual,
                                  pin_memory=False)
        eval_loader = DataLoader(train_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SubsetRandomSampler(eval_idxs),
                                 num_workers=args.num_workers,
                                 collate_fn=train_dataset.collater_visual,
                                 pin_memory=False)
        train_loaders.append(train_loader)
        eval_loaders.append(eval_loader)


    test_loaders = []
    for test_set in args.test_sets:
 
        names, labels = read_names_labels(label_path='/home/slasher/AMSAER-Baseline/dataset-process/label-AMSAER-visual.npz', data_type=test_set, debug=args.debug)
        print (f'{test_set}: sample number {len(names)}')
        test_dataset = get_datasets(args, names, labels)
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=test_dataset.collater_visual,
                                 shuffle=False,
                                 pin_memory=False)
        test_loaders.append(test_loader)

    ## return loaders
    adim, tdim, vdim = train_dataset.get_featdim()
    return train_loaders, eval_loaders, test_loaders, adim, tdim, vdim


########################################################
################ Define  your models  ##################
########################################################
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim1, output_dim2=1, output_dim3=3, layers='256,128', dropout=0.3):
        super(MLP, self).__init__()

        self.all_layers = []
        layers = list(map(lambda x: int(x), layers.split(',')))
        for i in range(0, len(layers)):
            self.all_layers.append(nn.Linear(input_dim, layers[i]))
            self.all_layers.append(nn.ReLU())
            self.all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        self.modul = nn.Sequential(*self.all_layers)
        self.fc_out_1 = nn.Linear(layers[-1], output_dim1)
        self.fc_out_2 = nn.Linear(layers[-1], output_dim2)
        self.fc_out_3 = nn.Linear(layers[-1], output_dim3)
        
    def forward(self, inputs):
        print(inputs.shape)
        features = self.modul(inputs)
        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        sents_out  = self.fc_out_3(features)
        return features, emos_out, vals_out, sents_out


class Attention(nn.Module):
    def __init__(self, audio_dim, text_dim, video_dim, output_dim1, output_dim2=1,output_dim3=3, hidden_dim= 128, layers='256,128', dropout=0.3, feat_type='utt'):
        super(Attention, self).__init__()

        if feat_type in ['utt']:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)
        elif feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = LSTMEncoder(video_dim, hidden_dim, dropout)

        self.attention_mlp = MLPEncoder(hidden_dim * 3, hidden_dim, dropout)

        self.audio_mlp = self.MLP(audio_dim, layers, dropout)
        self.text_mlp  = self.MLP(text_dim,  layers, dropout)
        self.video_mlp = self.MLP(video_dim, layers, dropout)

        self.attention_mlp = MLPEncoder(hidden_dim * 3, hidden_dim, dropout)

        layers_list = list(map(lambda x: int(x), layers.split(',')))
        #hiddendim = layers_list[-1] * 3
        #self.attention_mlp = self.MLP(hidden_dim * 3 , layers, dropout)

        self.fc_att   = nn.Linear(hidden_dim, 3)
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)
        self.fc_out_3 = nn.Linear(hidden_dim , output_dim3)
        #self.fc_out_3 = nn.Linear(layers_list[-1], output_dim3)
    
    def MLP(self, input_dim, layers, dropout):
        all_layers = []
        layers = list(map(lambda x: int(x), layers.split(',')))
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        module = nn.Sequential(*all_layers)
        return module

    def forward(self, batch):
        audio_hidden = self.audio_encoder(batch['audios']) # [32, 128]
        text_hidden  = self.text_encoder(batch['texts'])   # [32, 128]
        video_hidden = self.video_encoder(batch['videos']) # [32, 128]

        multi_hidden1 = torch.cat([audio_hidden, text_hidden, video_hidden], dim=1) # [32, 384]
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = torch.unsqueeze(attention, 2) # [32, 3, 1]

        multi_hidden2 = torch.stack([audio_hidden, text_hidden, video_hidden], dim=2) # [32, 128, 3]
        fused_feat = torch.matmul(multi_hidden2, attention)
        fused_feat = fused_feat.squeeze(axis=2) # [32, 128]
        emos_out  = self.fc_out_1(fused_feat)
        vals_out  = self.fc_out_2(fused_feat)
        sents_out  = self.fc_out_3(fused_feat)
        interloss = torch.tensor(0).cuda()
        return fused_feat, emos_out, vals_out, sents_out, interloss


class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum')

    def forward(self, pred, target):
        pred = F.log_softmax(pred, 1)
        target = target.squeeze().long()
        loss = self.loss(pred, target) / len(pred)
        return loss

class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target):
        pred = pred.view(-1,1)
        target = target.view(-1,1)
        loss = self.loss(pred, target) / len(pred)
        return loss


########################################################
########### main training/testing function #############
########################################################
def train_or_eval_model(args, model, reg_loss, cls_loss, dataloader, optimizer=None, train=False):
    
    vidnames = []
    val_preds, val_labels = [], []
    emo_probs, emo_labels = [], []
    sent_probs, sent_labels = [], []
    embeddings = []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        ## analyze dataloader

        batch, emos, vals, sents,  bnames = data
        vidnames += bnames
        multi_feat = torch.cat([batch['audios'], batch['texts'], batch['videos']], dim=2)
        for key in batch: batch[key] = batch[key].cuda()
        audio_feat, text_feat, visual_feat = batch['audios'], batch['texts'], batch['videos']
        #emos, vals = data[3], data[4].float()
        #sents, Task = data[5], data[6]
        #vidnames += data[-1]

        multi_feat = torch.cat([audio_feat, text_feat, visual_feat], dim=1)

        ## add cuda
        emos = emos.cuda()
        vals = vals.cuda()
        sents = sents.cuda()
        audio_feat  = audio_feat.cuda()
        text_feat   = text_feat.cuda()
        visual_feat = visual_feat.cuda()
        multi_feat  = multi_feat.cuda()

        ## feed-forward process
        if args.model_type == 'mlp':
            features, emos_out, vals_out, sents_out = model(multi_feat)
        elif args.model_type == 'attention':
            features, emos_out, vals_out, sents_out,interloss = model(batch)
        elif args.model_type == 'mult':
            features, emos_out, vals_out, sents_out,interloss  = model(batch)
        elif args.model_type == 'tfn':
            features, emos_out, vals_out, sents_out,interloss  = model(batch)
        elif args.model_type == 'lmf':
            features, emos_out, vals_out, sents_out,interloss  = model(batch)
        
        emo_probs.append(emos_out.data.cpu().numpy())
        sent_probs.append(sents_out.data.cpu().numpy())
        val_preds.append(vals_out.data.cpu().numpy())
        emo_labels.append(emos.data.cpu().numpy())
        sent_labels.append(sents.data.cpu().numpy())
        val_labels.append(vals.data.cpu().numpy())
        embeddings.append(features.data.cpu().numpy())

        ## optimize params
        if train:
            #loss1 = cls_loss(emos_out, emos)
            #loss2 = reg_loss(vals_out, vals)
            loss_sent = cls_loss(sents_out, sents)
            loss = loss_sent # Compute the loss of the model on the Sentiment Analysis  
            loss.backward()
            optimizer.step()

    ## evaluate on discrete 'Sentiment' labels
    sent_probs  = np.concatenate(sent_probs)
    sent_labels = np.concatenate(sent_labels)
    sent_preds = np.argmax(sent_probs, 1)
    sent_accuracy = accuracy_score(sent_labels, sent_preds)
    sent_fscore = f1_score(sent_labels, sent_preds, average='weighted')

    ## evaluate on discrete 'Emotion' labels 
    #emo_probs  = np.concatenate(emo_probs)
    #embeddings = np.concatenate(embeddings)
    #emo_labels = np.concatenate(emo_labels)
    #emo_preds = np.argmax(emo_probs, 1)
    #emo_accuracy = accuracy_score(emo_labels, emo_preds)
    #emo_fscore = f1_score(emo_labels, emo_preds, average='weighted')
    ## evaluate on dimensional labels
    #val_preds  = np.concatenate(val_preds)
    #val_labels = np.concatenate(val_labels)
    #val_mse = mean_squared_error(val_labels, val_preds)

    save_results = {}
    # item1: statistic results
    #save_results['val_mse'] = val_mse
    #save_results['emo_fscore'] = emo_fscore
    #save_results['emo_accuracy'] = emo_accuracy
    save_results['sent_fscore'] = sent_fscore
    save_results['sent_accuracy'] = sent_accuracy
    # item2: sample-level results
    save_results['sent_labels'] = sent_labels
    save_results['sent_probs'] = sent_probs
    #save_results['emo_probs'] = emo_probs
    #save_results['val_preds'] = val_preds
    #save_results['emo_labels'] = emo_labels
    #save_results['val_labels'] = val_labels
    save_results['names'] = vidnames
    # item3: latent embeddings
    if args.savewhole: save_results['embeddings'] = embeddings
    return save_results


########################################################
############# metric and save results ##################
########################################################
def overall_metric(emo_fscore, val_mse):
    final_score = val_mse # emo_fscore - val_mse  
    return final_score

def average_folder_results(folder_save, testname):
    name2preds = {}
    num_folder = len(folder_save)
    for ii in range(num_folder):
        names    = folder_save[ii][f'{testname}_names']
        #emoprobs = folder_save[ii][f'{testname}_emoprobs']
        sentprobs = folder_save[ii][f'{testname}_sentprobs']
        #valpreds = folder_save[ii][f'{testname}_valpreds']
        for jj in range(len(names)):
            name = names[jj]
            #emoprob = emoprobs[jj]
            sentprob = sentprobs[jj]
            #valpred = valpreds[jj]
            if name not in name2preds: name2preds[name] = []
            name2preds[name].append({'sent':sentprob}) #'emo': emoprob, 'val': valpred, })

    ## gain average results
    name2avgpreds = {}
    for name in name2preds:
        preds = np.array(name2preds[name])
        #emoprobs = [pred['emo'] for pred in preds if 1==1]
        sentprobs = [pred['sent'] for pred in preds if 1==1]
        #valpreds = [pred['val'] for pred in preds if 1==1]

        #avg_emoprob = np.mean(emoprobs, axis=0)
        #avg_emopred = np.argmax(avg_emoprob)
        avg_sentprob = np.mean(sentprobs, axis=0)
        avg_sentpred = np.argmax(avg_sentprob)
        #avg_valpred = np.mean(valpreds)
        name2avgpreds[name] = {'sent': avg_sentpred, 'sentprob': avg_sentpred} #'emo': avg_emopred, 'val': avg_valpred, 'emoprob': avg_emoprob, }
    return name2avgpreds

def gain_name2feat(folder_save, testname):
    name2feat = {}
    assert len(folder_save) >= 1
    names      = folder_save[0][f'{testname}_names']
    embeddings = folder_save[0][f'{testname}_embeddings']
    for jj in range(len(names)):
        name = names[jj]
        embedding = embeddings[jj]
        name2feat[name] = embedding
    return name2feat

def write_to_csv_pred(name2preds, save_path):
    names, emos, vals, sents = [], [], [], []
    for name in name2preds:
        names.append(name)
        #emos.append(idx2emo[name2preds[name]['emo']])
        sents.append(idx2sent[name2preds[name]['sent']])
        #vals.append(name2preds[name]['val'])

    columns = ['name', 'discrete', 'valence','sentiment']
    data = np.column_stack([names, emos, vals, sents])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(save_path, index=False)

def report_results_on_test1(test_label, test_pred):

    # read target file (few for test3)
    name2label = {}
    df_label = pd.read_csv(test_label)
    for _, row in df_label.iterrows():
        name = row['name']
        emo  = row['discrete']
        val  = row['valence']
        sen  = row['sentiment']
        name2label[name] = {'emo': emo2idx[emo], 'val': val, 'sent': sent2idx[sen]}
    print (f'labeled samples: {len(name2label)}')

    # read prediction file (more for test3)
    name2pred = {}
    df_label = pd.read_csv(test_pred)
    for _, row in df_label.iterrows():
        name = row['name']
        emo  = row['discrete']
        val  = row['valence']
        sen  = row['sentiment']
        name2pred[name] = {'emo': emo2idx[emo], 'val': val, 'sent': sent2idx[sen]}
    print (f'predict samples: {len(name2pred)}')
    assert len(name2pred) == len(name2label), f'make sure len(name2pred)=len(name2label)'

    emo_labels, emo_preds, val_labels, val_preds, sent_labels, sent_preds = [], [], [], [], [], []
    for name in name2label:
        emo_labels.append(name2label[name]['emo'])
        sent_labels.append(name2label[name]['sent'])
        val_labels.append(name2label[name]['val'])
        emo_preds.append(name2pred[name]['emo'])
        sent_preds.append(name2pred[name]['sent'])
        val_preds.append(name2pred[name]['val'])

    # analyze results
    #emo_fscore = f1_score(emo_labels, emo_preds, average='weighted')
    #print (f'emo results (weighted f1 score): {emo_fscore:.4f}')
    #emo_acc = accuracy_score(emo_labels, emo_preds)
    #print (f'emo results (weighted accuraccy): {emo_acc:.4f}')
    sent_fscore = f1_score(sent_labels, sent_preds, average='weighted')
    print (f'sentiment results (weighted f1 score): {sent_fscore:.4f}')
    sent_acc = accuracy_score(sent_labels, sent_preds)
    print (f'sentiment results (weighted accuraccy): {sent_acc:.4f}')
    #val_mse = mean_squared_error(val_labels, val_preds)
    #print (f'val results (mse): {val_mse:.4f}')
    #final_metric = overall_metric(emo_fscore, val_mse)
    #print (f'overall metric: {final_metric:.4f}')
    return sent_fscore #emo_fscore, val_mse, final_metric, sent_fscore

def calculate_results( emo_probs=[], emo_labels=[], val_preds=[], val_labels=[]):
        
        non_zeros = np.array([i for i, e in enumerate(val_labels) if e != 0]) # remove 0, and remove mask
        emo_accuracy = accuracy_score((val_labels[non_zeros] > 0), (val_preds[non_zeros] > 0))
        emo_fscore = f1_score((val_labels[non_zeros] > 0), (val_preds[non_zeros] > 0), average='weighted')

        results = { 
                    'valpreds':  val_preds,
                    'vallabels': val_labels,
                    'emoacc':    emo_accuracy,
                    'emofscore': emo_fscore
                    }
        outputs = f'f1:{emo_fscore:.4f}_acc:{emo_accuracy:.4f}'

        return results, outputs
## only fscore for test3
def report_results_on_test3(test_label, test_pred):

    # read target file (few for test3)
    name2label = {}
    df_label = pd.read_csv(test_label)
    for _, row in df_label.iterrows():
        name = row['name']
        emo  = row['discrete']
        name2label[name] = {'emo': emo2idx[emo]}
    print (f'labeled samples: {len(name2label)}')

    # read prediction file (more for test3)
    name2pred = {}
    df_label = pd.read_csv(test_pred)
    for _, row in df_label.iterrows():
        name = row['name']
        emo  = row['discrete']
        name2pred[name] = {'emo': emo2idx[emo]}
    print (f'predict samples: {len(name2pred)}')
    assert len(name2pred) >= len(name2label)

    emo_labels, emo_preds = [], []
    for name in name2label: # on few for test3
        emo_labels.append(name2label[name]['emo'])
        emo_preds.append(name2pred[name]['emo'])

    # analyze results
    emo_fscore = f1_score(emo_labels, emo_preds, average='weighted')
    print (f'emo results (weighted f1 score): {emo_fscore:.4f}')
    return emo_fscore, -100, -100


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Params for input
    parser.add_argument('--dataset', type=str, default=None, help='dataset')
    parser.add_argument('--feat_type', type=str, default=None, help='frm_align')
    parser.add_argument('--train_dataset', type=str, default=None, help='dataset') # for cross-dataset evaluation
    parser.add_argument('--test_dataset',  type=str, default=None, help='dataset') # for cross-dataset evaluation
    parser.add_argument('--audio_feature', type=str, default=None, help='audio feature name')
    parser.add_argument('--text_feature', type=str, default=None, help='text feature name')
    parser.add_argument('--video_feature', type=str, default=None, help='video feature name')
    parser.add_argument('--debug', action='store_true', default=False, help='whether use debug to limit samples')
    parser.add_argument('--test_sets', type=str, default='test1,test2', help='process on which test sets, [test1, test2, test3]')
    parser.add_argument('--task_set', type=str, default='emo', help='process on which task we train the model, [emo, sent, val]')
    parser.add_argument('--save_root', type=str, default='./saved', help='save prediction results and models')
    parser.add_argument('--savewhole', action='store_true', default=False, help='whether save latent embeddings')
    parser.add_argument('--feat_scale', type=int, default=None, help='pre-compress input from [seqlen, dim] -> [seqlen/scale, dim]')

    ## Params for model  
    parser.add_argument('--layers', type=str, default='256,128', help='hidden size in model training')
    parser.add_argument('--n_classes', type=int, default=-1, help='number of classes [defined by args.label_path]')
    parser.add_argument('--num_folder', type=int, default=-1, help='folders for cross-validation [defined by args.dataset]')
    parser.add_argument('--model_type', type=str, default='mlp', help='model type for training [mlp or attention]')
    parser.add_argument('--model', type=str, default='mlp', help='model type for training [mlp or attention]')

    ## Params for training
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=8, metavar='BS', help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, metavar='nw', help='number of workers')
    parser.add_argument('--epochs', type=int, default=30, metavar='E', help='number of epochs')
    parser.add_argument('--seed', type=int, default=100, help='make split manner is same with same seed')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
    args = parser.parse_args()
    
       
    args.n_classes = 6   #13  Emotion classes
    args.num_folder = 5
    args.debug = False
    args.test_sets = args.test_sets.split(',')
    args.task_set = args.task_set.split(',')
    args.visual =  True

    if args.feat_type == 'utt':
        args.feat_scale = 1
    elif args.feat_type == 'frm_align':
        args.feat_scale = 6
    elif args.feat_type == 'frm_unalign':
        args.feat_scale = 12

    if args.dataset is not None:
        args.train_dataset = args.dataset
        args.test_dataset  = args.dataset
    assert args.train_dataset is not None
    assert args.test_dataset  is not None

    whole_features = [args.audio_feature, args.text_feature, args.video_feature]
    if len(set(whole_features)) == 1:
        args.save_root = f'{args.save_root}-unimodal'
    elif len(set(whole_features)) == 2:
        args.save_root = f'{args.save_root}-bimodal'
    elif len(set(whole_features)) == 3:
        args.save_root = f'{args.save_root}-trimodal'

    torch.cuda.set_device(args.gpu)
    print(args)

    
    print (f'====== Reading Data =======')
    train_loaders, eval_loaders, test_loaders, adim, tdim, vdim = get_loaders(args, config)      
    #assert len(train_loaders) == args.num_folder, f'Error: folder number'
    #assert len(eval_loaders)   == args.num_folder, f'Error: folder number'
    
    
    print (f'====== Training and Evaluation =======')
    folder_save = []
    folder_evalres = []
    for ii in range(args.num_folder):
        print (f'>>>>> Cross-validation: training on the {ii+1} folder >>>>>')
        train_loader = train_loaders[ii]
        eval_loader  = eval_loaders[ii]
        start_time = time.time()
        name_time  = time.time()

        print (f'Step1: build model (each folder has its own model)')
        if args.model_type == 'mlp':
            
            model = MLP(input_dim=adim + tdim + vdim,
                        output_dim1=args.n_classes,
                        output_dim2=1,
                        output_dim3=3,
                        layers= args.layers)
        elif args.model_type == 'attention':
            model = Attention(audio_dim=adim+4,
                              text_dim=tdim+4,
                              video_dim=vdim+4,
                              output_dim1=args.n_classes,
                              output_dim2=1,
                              output_dim3=3,
                              hidden_dim=128,
                              layers=args.layers,
                              feat_type= args.feat_type)
        elif args.model_type == 'mult':
            model = MULT(audio_dim=adim,
                         text_dim=tdim,
                         video_dim=vdim,
                         output_dim1=args.n_classes,
                         output_dim2= 1,
                         output_dim3= 3,
                         layers= 4, #[2, 4, 6]
                         dropout= 0.5,# [0.0, 0.1, 0.2, 0.3]
                         num_heads= 8,
                         hidden_dim = 128,
                         attn_mask = True,
                         conv1d_kernel_size = 4,
                         grad_clip = 0.6 )
        elif args.model_type == 'lmf':
            model = LMF(audio_dim=adim,
                         text_dim=tdim,
                         video_dim=vdim,
                         output_dim1=args.n_classes,
                         output_dim2= 1,
                         output_dim3= 3,
                         dropout= 0.5,# [0.0, 0.1, 0.2, 0.3]
                         rank= 4,
                         hidden_dim = 128,
                         grad_clip = 0.6 ,
                         feat_type= args.feat_type)
        elif args.model_type == 'tfn':
            model = TFN(audio_dim=adim,
                         text_dim=tdim,
                         video_dim=vdim,
                         output_dim1=args.n_classes,
                         output_dim2= 1,
                         output_dim3= 3,
                         dropout= 0.5,# [0.0, 0.1, 0.2, 0.3]
                         hidden_dim = 128,
                         grad_clip = 0.6,
                         feat_type= args.feat_type )
        reg_loss = MSELoss()
        cls_loss = CELoss()
        model.cuda()
        reg_loss.cuda()
        cls_loss.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        print (f'Step2: training (multiple epoches)')
        eval_metrics = []
        eval_fscores = []
        eval_valmses = []
        test_save = []
        for epoch in range(args.epochs):

            store_values = {}

            ## training and validation
            train_results = train_or_eval_model(args, model, reg_loss, cls_loss, train_loader, optimizer=optimizer, train=True)
            eval_results  = train_or_eval_model(args, model, reg_loss, cls_loss, eval_loader,  optimizer=None,      train=False)
            eval_metric = overall_metric(eval_results['emo_fscore'], eval_results['val_mse']) # bigger -> better
            eval_metrics.append(eval_metric)
            eval_fscores.append(eval_results['emo_fscore'])
            eval_valmses.append(eval_results['val_mse'])
            store_values['eval_emoprobs'] = eval_results['emo_probs']
            store_values['eval_sentprobs'] = eval_results['sent_probs']
            store_values['eval_valpreds'] = eval_results['val_preds']
            store_values['eval_names']    = eval_results['names']
            print ('epoch:%d; train_fscore: %.4f; eval_metric: %.4f' %(epoch+1, train_results['emo_fscore'], eval_metric))

            ## testing and saving
            for jj, test_loader in enumerate(test_loaders):
                test_set = args.test_sets[jj]
                test_results = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, optimizer=None, train=False)
                store_values[f'{test_set}_emoprobs']   = test_results['emo_probs']
                store_values[f'{test_set}_sentprobs']   = test_results['sent_probs']
                store_values[f'{test_set}_valpreds']   = test_results['val_preds']
                store_values[f'{test_set}_names']      = test_results['names']
                if args.savewhole: store_values[f'{test_set}_embeddings'] = test_results['embeddings']
            test_save.append(store_values)
            
        print (f'Step3: saving and testing on the {ii+1} folder')
        best_index = np.argmax(np.array(eval_metrics))
        best_save  = test_save[best_index]
        best_evalfscore = eval_fscores[best_index]
        best_evalvalmse = eval_valmses[best_index]
        folder_save.append(best_save)
        folder_evalres.append([best_evalfscore, best_evalvalmse])
        end_time = time.time()
        print (f'>>>>> Finish: training on the {ii+1} folder, duration: {end_time - start_time} >>>>>')


    print (f'====== Gain predition on test data =======')
    assert len(folder_save)     == args.num_folder
    assert len(folder_evalres)  == args.num_folder
    save_modelroot = os.path.join(args.save_root, 'model')
    save_predroot  = os.path.join(args.save_root, 'prediction')
    if not os.path.exists(save_predroot): os.makedirs(save_predroot)
    if not os.path.exists(save_modelroot): os.makedirs(save_modelroot)
    feature_name = f'{args.audio_feature}+{args.text_feature}+{args.video_feature}'

    ## analyze cv results
    cv_fscore, cv_valmse = np.mean(np.array(folder_evalres), axis=0)
    cv_metric = overall_metric(cv_fscore, cv_valmse)
    res_name = f'f1:{cv_fscore:.4f}_val:{cv_valmse:.4f}_metric:{cv_metric:.4f}'
    save_path = f'{save_modelroot}/cv_features:{feature_name}_{res_name}_{name_time}.npz'
    print (f'save results in {save_path}')
    np.savez_compressed(save_path, args=np.array(args, dtype=object))

    for setname in args.test_sets:
        pred_path  = f'{save_predroot}/{setname}-pred-{name_time}.csv'
        label_path = f'./{setname}-label.csv'
        name2preds = average_folder_results(folder_save, setname)
        if args.savewhole: name2feats = gain_name2feat(folder_save, setname)
        write_to_csv_pred(name2preds, pred_path)
        
        res_name = 'nores'
        if os.path.exists(label_path):
            if setname in ['test1']: sent_fscore = report_results_on_test1(label_path, pred_path)
            #if setname in ['test3']:          emo_fscore, val_mse, final_metric = report_results_on_test3(label_path, pred_path)
            res_name = f'f1:{emo_fscore:.4f}_val:{val_mse:.4f}_metric:{final_metric:.4f}'

        save_path = f'{save_modelroot}/{setname}_features:{feature_name}_{res_name}_{name_time}.npz'
        print (f'save results in {save_path}')

        if args.savewhole:
            np.savez_compressed(save_path,
                            name2preds=name2preds,
                            name2feats=name2feats,
                            args=np.array(args, dtype=object))
        else:
            np.savez_compressed(save_path,
                                name2preds=name2preds,
                                args=np.array(args, dtype=object))
