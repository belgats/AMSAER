import torch
import numpy as np
from torch.utils.data import Dataset
from toolkit.utils.read_data import *

class Data_Feat(Dataset):
    def __init__(self, args, names, labels):

        # analyze path
        self.names = names
        self.labels = labels
        #feat_root  = config.PATH_TO_FEATURES[args.dataset]
        audio_root = os.path.join('/home/slasher/MER2023-Baseline/dataset-process/features/', args.audio_feature)
        text_root  = os.path.join('/home/slasher/MER2023-Baseline/dataset-process/features/', args.text_feature )
        video_root = os.path.join('/home/slasher/MER2023-Baseline/dataset-process/features/', args.video_feature)
        print (f'audio feature root: {audio_root}')

        # --------------- temporal test ---------------
        # for name in names: assert os.path.exists(os.path.join(audio_root, name+'.npy'))

        # analyze params
        self.feat_type = args.feat_type
        self.feat_scale = args.feat_scale # 特征预压缩
        assert self.feat_scale >= 1
        assert self.feat_type in ['utt', 'frm_align', 'frm_unalign']

        # read datas (reduce __getitem__ durations)
        audios, self.adim = func_read_multiprocess(audio_root, self.names, read_type='feat')
        texts,  self.tdim = func_read_multiprocess(text_root,  self.names, read_type='feat')
        videos, self.vdim = func_read_multiprocess(video_root, self.names, read_type='feat')

        ## read batch (reduce collater durations)
        # step1: pre-compress features
        audios, texts, videos = feature_scale_compress(audios, texts, videos, self.feat_scale)
        # step2: align to batch
        if self.feat_type == 'utt': # -> 每个样本每个模态的特征压缩到句子级别
            audios, texts, videos = align_to_utt(audios, texts, videos)
        elif self.feat_type == 'frm_align':
            audios, texts, videos = align_to_text(audios, texts, videos) # 模态级别对齐
            audios, texts, videos = pad_to_maxlen_pre_modality(audios, texts, videos) # 样本级别对齐
        elif self.feat_type == 'frm_unalign':
            audios, texts, videos = pad_to_maxlen_pre_modality(audios, texts, videos) # 样本级别对齐
        self.audios, self.texts, self.videos = audios, texts, videos

 
    def __len__(self):
        return len(self.names)


    def __getitem__(self, index):
        instance = dict(
            audio = self.audios[index],
            text  = self.texts[index],
            video = self.videos[index],
            emo   = self.labels[index]['emo'],
            val   = self.labels[index]['val'],
            sent   = self.labels[index]['sent'],
            visual   = self.labels[index]['visual'],
            name  = self.names[index],
        )
        return instance
    

    def collater(self, instances):
        audios = [instance['audio'] for instance in instances]
        texts  = [instance['text']  for instance in instances]
        videos = [instance['video'] for instance in instances]
       
        print(audios,texts)
        batch = dict(
            audios = torch.FloatTensor(np.array(audios)),
            texts  = torch.FloatTensor(np.array(texts)),
            videos = torch.FloatTensor(np.array(videos)),
        )
        
        emos  = torch.LongTensor([instance['emo']  for instance in instances])
        sents  = torch.LongTensor([instance['sent']  for instance in instances])
        vals  = torch.FloatTensor([instance['val']  for instance in instances])
        names = [instance['name'] for instance in instances]
        
        return batch, emos, vals, sents, names
    
    def collater_visual(self, instances):
        audios = [instance['audio'] for instance in instances]
        texts  = [instance['text']  for instance in instances]
        videos = [instance['video'] for instance in instances]
        visuals = [instance['visual']  for instance in instances]
        batch_audios = np.array(audios, dtype=np.float32)
        batch_texts = np.array(texts, dtype=np.float32)
        batch_videos = np.array(videos, dtype=np.float32)
        # Convert visual strings to numerical arrays
        visuals_numeric = [np.array(list(map(int, instance['visual'])), dtype=np.float32) for instance in instances]
        # Check shapes and concatenate
 
        # Convert visuals to a numpy array
        batch_visuals = np.array(visuals_numeric)
        batch_visuals_expanded = np.expand_dims(batch_visuals, axis=1) 
        batch_visuals = np.repeat(batch_visuals_expanded, repeats=batch_videos.shape[1], axis=1) 
        combined_videos = np.concatenate((batch_videos, batch_visuals), axis=2)
        # Check shapes and concatenate
  
        batch_visuals = np.array(visuals_numeric)
        batch_visuals_expanded = np.expand_dims(batch_visuals, axis=1) 
        batch_visuals = np.repeat(batch_visuals_expanded, repeats=batch_texts.shape[1], axis=1) 
        combined_texts = np.concatenate((batch_texts, batch_visuals), axis=2)
        # Check shapes and concatenate
    
         
        batch_visuals = np.array(visuals_numeric)
        batch_visuals_expanded = np.expand_dims(batch_visuals, axis=1) 
        batch_visuals = np.repeat(batch_visuals_expanded, repeats=batch_audios.shape[1], axis=1)  
        combined_audios = np.concatenate((batch_audios, batch_visuals), axis=2)
    

        # Concatenate along the feature dimension (axis=1)
        combined_videos = np.concatenate((batch_videos, batch_visuals), axis=2)
       
        batch = dict(
            audios = torch.FloatTensor(combined_audios),
            texts  = torch.FloatTensor(combined_texts),
            videos = torch.FloatTensor(combined_videos),
        )
        
        emos  = torch.LongTensor([instance['emo']  for instance in instances])
        sents  = torch.LongTensor([instance['sent']  for instance in instances])
        vals  = torch.FloatTensor([instance['val']  for instance in instances])
        names = [instance['name'] for instance in instances]
        
        return batch, emos, vals, sents, names
    
    def get_featdim(self):
        print (f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim
    