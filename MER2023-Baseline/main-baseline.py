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

import cv2 # pip install opencv-python
import config


correspondence = {
    'neutral': ['interested'],
    'angry':   ['disgust'],
    'happy': ['loving', 'proud', 'glad'],
    'sad': ['sorry'],
    'worried': ['afraid', 'fear'],
    'surprise': ['excited']
}
# split audios from videos
def split_audio_from_video_16k(video_root, save_root):
    if not os.path.exists(save_root): os.makedirs(save_root)
    for video_path in tqdm.tqdm(glob.glob(video_root+'/*')):
        videoname = os.path.basename(video_path)[:-4]
        audio_path = os.path.join(save_root, videoname + '.wav')
        if os.path.exists(audio_path): continue
        cmd = "%s -loglevel quiet -y -i %s -ar 16000 -ac 1 %s" %(config.PATH_TO_FFMPEG, video_path, audio_path)
        os.system(cmd)

# preprocess dataset-release
def normalize_dataset_format(data_root, save_root):
    ## input path
    train_data  = os.path.join(data_root, 'train')
    train_label = os.path.join(data_root, 'train-label.csv')
    test1_data  = os.path.join(data_root, 'test1')
    label_path  = os.path.join(data_root, 'dataset-process/annotation/')
  
    ## output path 092363023  234151723
    save_video = os.path.join(save_root, 'video')
    if not os.path.exists(save_root): os.makedirs(save_root)
    if not os.path.exists(save_video): os.makedirs(save_video)

    for subfolder in os.listdir(train_data):
        subfolder_path = os.path.join(train_data, subfolder)
        if os.path.isdir(subfolder_path):  # Check if it's a directory
          video_paths = glob.glob(os.path.join(subfolder_path, '*'))
          for video_path in tqdm.tqdm(video_paths):
            video_name = os.path.basename(video_path)
            new_video_name = f"{subfolder}_{video_name}"
            new_path = os.path.join(save_video, new_video_name)
            shutil.copy(video_path, new_path)
 
def normalize_dataset_format_label(data_root, save_root):
    ## input path
    label_path = '/home/slasher/MER2023-Baseline/dataset-process/annotation/'
    label_test_path = '/home/slasher/MER2023-Baseline/dataset-process/test'
    save_label = os.path.join(save_root, 'label-6way.npz')
    ## generate label path
    train_corpus = {}
    test1_corpus = {}
    clips = {}
    clipstest = {}
    for temp_root in tqdm.tqdm(os.listdir(label_path)) :
        file_path = os.path.join(label_path, f'{temp_root}') # .json files
        video_name = os.path.basename(temp_root)[:-5]
        with open(file_path ) as f:
            data = f.read()
            line1, line2 = data.split('\n')[:2]          
            if line2:
                obj2 = json.loads(line2)
                for i,point_id in enumerate(obj2):
                     # Extract the numerical part from the segment ID using regular expressions
                    segment_id = obj2[point_id]['segment_id']
                    segment_number = int(re.search(r'\d+', segment_id).group())
                    name = f"{video_name}_v_{video_name}_{segment_number}"
                    
                    user = obj2[point_id]['user']
                    if name not in clips:
                      clips[name] = {}
                    # Initialize   
                    labelText = obj2[point_id]['labelText']
                    emotion = None
                    emotion_V6 = None
                    val = None
                    task = None
                    sent = None
        
                    if labelText in ['afraid', 'loving', 'happy', 'proud', 'excited', 'glad', 'interested', 'sad', 'angry', 'fear', 'disgust', 'sorry']:
                        emotion = labelText
                        for original_emotion, extended_emotions in correspondence.items():
                            if emotion == original_emotion or (isinstance(extended_emotions, list) and emotion in extended_emotions):
                               #print(emotion_1,'==', original_emotion) 
                               emotion_V6 = original_emotion
                               break

                    elif labelText in ['-3', '-2', '-1', '0', '1', '2', '3']:
                         val = labelText
                         if val in ['-3','-2','-1']:
                             sent = 'negative'
                         elif val in ['3','2','1']:
                             sent = 'positive'
                         else :
                             sent = 'neutral'
                    elif labelText in ['easy', 'hard']:
                         task = labelText
                    #vai_int = float(val)

                    if user not in clips[name]:
                       clips[name][user] = {'emotion': emotion,'emotionV6': emotion_V6, 'sentiment': sent, 'val': val, 'Task': task}
                    else:
                        if emotion:
                            # Check if clips[name][user]['emotion'] is None and initialize it as an empty string if needed
                            if clips[name][user]['emotion'] is None:
                                clips[name][user]['emotion'] =  emotion
                                clips[name][user]['emotionV6'] =  emotion_V6
                            else:
                               # Concatenate the strings
                               clips[name][user]['emotion'] += ',' + emotion                        
                               clips[name][user]['emotionV6'] += ',' + emotion_V6                       
                        if val:
                            clips[name][user]['val'] = val
                            clips[name][user]['sentiment'] = sent
                        if task:
                            clips[name][user]['Task'] = task

 
    # Specify the file path
    file_path = "complex_data_indexed.json"

    # Write the data to the JSON file
    with open(file_path, "w") as json_file:
      json.dump(clips, json_file, indent=4)  

    # Iterate through each segment
    for item, itemdata in clips.items():
        user1 =  itemdata['ikram'] 
        if "aida" not in itemdata:
            itemdata["aida"] = {
            "emotion": user1['emotion'],
            "emotionV6": user1['emotionV6'],
            "val": user1['val'],
            "sentiment": user1['sentiment'],
            "Task": user1['Task']
            }
        user2 = itemdata['aida']
        # Unify "emotion" by concatenating them
        emotion_1 = user1.get("emotion")
        if emotion_1 is None:
           emotion_1 = "neutral"

        for original_emotion, extended_emotions in correspondence.items():
          if emotion_1 == original_emotion or (isinstance(extended_emotions, list) and emotion_1 in extended_emotions):
            #print(emotion_1,'==', original_emotion)
            emotion_1 = original_emotion
            break
        emotion_2 = user2.get("emotion", "neutral")
        #unified_emotion = f"{emotion_1} {emotion_2}".strip()
         

        # Unify "val" by calculating the mean
        val_1 = float(user1.get("val", 0))
        val_2 = float(user2.get("val", 0))
        unified_val = (val_1 + val_2) / 2
        if unified_val >= 1:
            sent = 'positive'
        elif unified_val <= -1:
            sent = 'negative'
        else:
            sent = 'neutral'


        # Unify "Task" by concatenating them
        task_1 = user1.get("Task", "")
        task_2= user2.get("Task", "")
        #unified_task = f"{task_ikram} {task_aida}".strip()

        train_corpus[item] = {'emo': emotion_1, 'val': unified_val, 'sent': sent, 'Task': task_1}
    print(train_corpus) 

    print( 'Iterate through each segment') 

    for temp_root in tqdm.tqdm(os.listdir(label_test_path)) :
        file_path = os.path.join(label_test_path, f'{temp_root}') # .json files
        video_name = os.path.basename(temp_root)[:-5]
        with open(file_path ) as f:
            data = f.read()
            line1, line2 = data.split('\n')[:2]          
            if line2:
                obj2 = json.loads(line2)
                for i, point_id in enumerate(obj2):
                     # Extract the numerical part from the segment ID using regular expressions
                    segment_id = obj2[point_id]['segment_id']
                    segment_number = int(re.search(r'\d+', segment_id).group())
                    name = f"{video_name}_v_{video_name}_{segment_number}"
                    user = obj2[point_id]['user']
                    if name not in clipstest:
                      clipstest[name] = {}
                    # Initialize   
                    labelText = obj2[point_id]['labelText']
                    emotion = None
                    val = None
                    task = None
        
                    if labelText in ['afraid', 'loving', 'happy', 'proud', 'excited', 'glad', 'interested', 'sad', 'angry', 'fear', 'disgust', 'sorry']:
                         emotion = labelText
                    elif labelText in ['-3', '-2', '-1', '0', '1', '2', '3']:
                         val = labelText
                    elif labelText in ['easy', 'hard']:
                         task = labelText
                    #vai_int = float(val)

                    if user not in clipstest[name]:
                       clipstest[name][user] = {'emotion': emotion, 'val': val, 'Task': task}
                    else:
                        if emotion:
                            # Check if clips[name][user]['emotion'] is None and initialize it as an empty string if needed
                            if clipstest[name][user]['emotion'] is None:
                                clipstest[name][user]['emotion'] =  emotion
                            else:
                               # Concatenate the strings
                               clipstest[name][user]['emotion'] += ',' + emotion                        
                        if val:
                            clipstest[name][user]['val'] = val
                        if task:
                            clipstest[name][user]['Task'] = task
    print( 'Iterate through each segment') 
    # Iterate through each segment
    
    for item, itemdata in clipstest.items():
        user1 =  itemdata['ikram'] 
        if "aida" not in itemdata:
            itemdata["aida"] = {
            "emotion": user1['emotion'],
            "val": user1['val'],
            "Task": user1['Task']
            }
        user2 = itemdata['aida']
        # Unify "emotion" by concatenating them
        emotion_1 = user1.get("emotion")
        if emotion_1 is None:
           emotion_1 = "neutral"
        
        for original_emotion, extended_emotions in correspondence.items():
          if emotion_1 == original_emotion or (isinstance(extended_emotions, list) and emotion_1 in extended_emotions):
            #print(emotion_1,'==', original_emotion)
            emotion_1 = original_emotion
            break
        emotion_2 = user2.get("emotion", "neutral")
        #unified_emotion = f"{emotion_1} {emotion_2}".strip()
         

        # Unify "val" by calculating the mean
        val_1 = float(user1.get("val", 0))
        val_2 = float(user2.get("val", 0))
        unified_val = (val_1 + val_2) / 2
        if unified_val >= 1:
            sent = 'positive'
            unified_val =  round(unified_val) 
        elif unified_val <= -1:
            sent = 'negative'
            unified_val = round(unified_val)
        else:
            sent = 'neutral'
            unified_val = 0


        # Unify "Task" by concatenating them
        task_1 = user1.get("Task", "")
        task_2= user2.get("Task", "")
        #unified_task = f"{task_ikram} {task_aida}".strip()

        test1_corpus[item] = {'emo': emotion_1, 'val': unified_val, 'sent': sent, 'Task': task_1}
    print(test1_corpus) 
          
    write_to_csv(test1_corpus, './test1-label.csv')
    np.savez_compressed(save_label,
                        train_corpus=train_corpus,
                        test1_corpus=test1_corpus)

def write_to_csv(name2preds, save_path):
    names, emos, vals, sents = [], [], [], []
    for name in name2preds:
        names.append(name)
        emos.append(name2preds[name]['emo'])
        vals.append(name2preds[name]['val'])
        sents.append(name2preds[name]['sent'])

    columns = ['name', 'discrete', 'valence', 'sentiment']
    data = np.column_stack([names, emos, vals, sents])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(save_path, index=False)
         
 
def annotation(file_path,video_name):
        clips = {}
        with open(file_path ) as f:
            data = f.read()
            line1, line2 = data.split('\n')[:2]          
            if line2:
                obj2 = json.loads(line2)
                for point_id in obj2:
                     # Extract the numerical part from the segment ID using regular expressions
                    segment_id = obj2[point_id]['segment_id']
                    segment_number = int(re.search(r'\d+', segment_id).group())
                    name = f"{video_name}_v_{video_name}_{segment_number}"
                    user = obj2[point_id]['user']
                    if name not in clips:
                      clips[name] = {}
                    # Initialize   
                    labelText = obj2[point_id]['labelText']
                    emotion = None
                    val = None
                    task = None
        
                    if labelText in ['afraid', 'loving', 'happy', 'proud', 'excited', 'glad', 'interested', 'sad', 'angry', 'fear', 'disgust', 'sorry']:
                         emotion = labelText
                    elif labelText in ['-3', '-2', '-1', '0', '1', '2', '3']:
                         val = labelText
                    elif labelText in ['easy', 'hard']:
                         task = labelText
                    #vai_int = float(val)

                    if user not in clips[name]:
                       clips[name][user] = {'emotion': emotion, 'val': val, 'Task': task}
                    else:
                        if emotion:
                            # Check if clips[name][user]['emotion'] is None and initialize it as an empty string if needed
                            if clips[name][user]['emotion'] is None:
                                clips[name][user]['emotion'] =  emotion
                            else:
                               # Concatenate the strings
                               clips[name][user]['emotion'] += ',' + emotion                        
                        if val:
                            clips[name][user]['val'] = val
                        if task:
                            clips[name][user]['Task'] = task
                    #labels.setdefault(segment_id, []).append(label)
                    #for segment_id, label_values in labels.items():
                    # Calculate the mean of label values for each segment ID
                    #labels_mean = sum(label_values)/ len(label_values)
                    #labels_mean = max(-1, min(labels_mean, 1))
                    #if -1 < labels_mean < 1:
                    #    labels_mean = 0
            return clips
        
# generate transcription files using asr
def generate_transcription_files(text_root, save_path):
    import os
    #import wenetruntime as wenet
    #decoder = wenet.Decoder('./tools/wenet/wenetspeech_u2pp_conformer_libtorch', lang='chs')

    names = []
    sentences = []
    for text_path in tqdm.tqdm(glob.glob(text_root + '/*')):
        name = os.path.basename(text_path)[:-4]
        with open(text_path, "r") as file:
            # Read the text from the file
            sentence = file.read()
        names.append(name)
        sentences.append(sentence.strip())

    ## write to csv file
    columns = ['name', 'sentence']
    data = np.column_stack([names, sentences])
    df = pd.DataFrame(data=data, columns=columns)
    df[columns] = df[columns].astype(str)
    df.to_csv(save_path, index=False)

# add punctuation to transcripts
def refinement_transcription_files_asr(old_path, new_path):
    from paddlespeech.cli.text.infer import TextExecutor
    text_punc = TextExecutor()

    ## read 
    names, sentences = [], []
    df_label = pd.read_csv(old_path)
    for _, row in df_label.iterrows():
        names.append(row['name'])
        sentence = row['sentence']
        if pd.isna(sentence):
            sentences.append('')
        else:
            sentence = text_punc(text=sentence)
            sentences.append(sentence)
        print (sentences[-1])

    ## write
    columns = ['name', 'sentence']
    data = np.column_stack([names, sentences])
    df = pd.DataFrame(data=data, columns=columns)
    df[columns] = df[columns].astype(str)
    df.to_csv(new_path, index=False)

 
if __name__ == '__main__':
    import fire
    fire.Fire()