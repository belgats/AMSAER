import re
import os
 
import tqdm
 
import json
 
import numpy as np
import pandas as pd
 
import config


correspondence = {
    'neutral': ['interested'],
    'angry':   ['disgust'],
    'happy': ['loving', 'proud', 'glad'],
    'sad': ['sorry'],
    'worried': ['afraid', 'fear'],
    'surprise': ['excited']
}
 
 
def gen_rating_matrix():
    label_path = "annotation_large.json"
    # Assuming 'val' labels are integers [-3, -2, -1, 0, 1, 2, 3]

    num_classes = 7
    clips = {}
    # Matrix to store the counts 
    # Loop through each annotator (ikram, aida, hamid)
    # Write the data to the JSON file
    with open(label_path, "w") as json_file:
      json.dump(clips, json_file, indent=4)  
    number_of_items = len(clips)
    rating_matrix = np.zeros((num_classes, number_of_items), dtype=int)
    print(np.zeros((num_classes, number_of_items), dtype=int))
    # Iterate through each segment
    annotators = ['ikram', 'aida', 'hamid']
    for idx, (item, itemdata) in enumerate(clips.items()):
        user1_val = int(itemdata['ikram'].get("val", 0))
        user2_val = int(itemdata['aida'].get("val", 0))
        user3_val = int(itemdata['hamid'].get("val", 0))
        for val in itemdata:
           # Increment the count in the rating matrix
           rating_matrix[idx, val + 3] += 1      

    print("Rating Matrix:")
    print(rating_matrix)


 
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
        elif unified_val <= -1:
            sent = 'negative'
        else:
            sent = 'neutral'


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



# Print the results
print(f'Fleiss\' Kappa for Sentiment: ')
 