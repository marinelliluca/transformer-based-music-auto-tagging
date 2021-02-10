# coding: utf-8
# modified from https://github.com/minzwon/self-attention-music-tagging
import os
import numpy as np
from torch.utils import data
import pickle as pkl
import librosa
import warnings
import math

class AudioFolder(data.Dataset):
    def __init__(self, 
                 input_length, # [s]
                 spec_path, 
                 audio_path, 
                 path_to_repo,
                 mode):

        self.spec_path = os.path.expanduser(spec_path)
        self.audio_path = os.path.expanduser(audio_path)
        self.path_to_repo = os.path.expanduser(path_to_repo)
        self.mode = mode
        self.fs = 16000
        self.window = 512
        self.hop = 256
        self.mel = 96
        self.input_length = math.floor(input_length*self.fs/self.hop)
        self.get_songlist()
        self.idmsd_to_id7d = pkl.load(open(os.path.join(self.path_to_repo,
                                                        "training/msd_metadata/MSD_id_to_7D_id.pkl"),'rb'))
        self.tags = pkl.load(open(os.path.join(self.path_to_repo,
                                               "training/msd_metadata/msd_id_to_tag_vector.cP"), 'rb'))
        
    def __getitem__(self, index):
        spec = None
        while spec is None:
            try:
                spec, tag_binary = self.get_spec(index)
            except: # audio not found or broken (very rare)
                index = np.random.randint(0,high=len(self.fl))
                spec = None
        return spec.astype('float32'), tag_binary.astype('float32')

    def get_songlist(self):
        train = pkl.load(open(os.path.join(self.path_to_repo,
                                           "training/msd_metadata/filtered_list_train.cP"), 'rb'))
        if self.mode == 'train':
            self.fl = train[:100] # train[:201680]
        elif self.mode == 'valid':
            self.fl = train[200:210] # train[201680:]
        elif self.mode == 'test':
            self.fl = pkl.load(open(os.path.join(self.path_to_repo,
                                                 "training/msd_metadata/filtered_list_train.cP"), 'rb'))
            
    def compute_melspectrogram(self, audio_fn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x, _ = librosa.core.load(audio_fn, sr=self.fs, res_type='kaiser_fast')
            spec = librosa.core.amplitude_to_db(librosa.feature.melspectrogram(x, 
                                                                               sr=self.fs, 
                                                                               n_fft=self.window, 
                                                                               hop_length=self.hop, 
                                                                               n_mels=self.mel))
        return spec

    def get_spec(self, index):
        
        fn=self.fl[index]
        id7d = self.idmsd_to_id7d[fn]
        audio_fn = os.path.join(self.audio_path,*id7d[:2],id7d+".clip.mp3")
        spec_fn = os.path.join(self.spec_path,*id7d[:2], id7d+'.npy') 
        
        if not os.path.exists(spec_fn):
            spec = self.compute_melspectrogram(audio_fn)
            spec_path = os.path.dirname(spec_fn)
            os.makedirs(spec_path)
            np.save(open(spec_fn, 'wb'), spec)
        else:
            spec = np.load(spec_fn, mmap_mode='r')
        
        upper_idx = math.floor(29*self.fs/self.hop)-self.input_length
        random_idx = np.random.randint(0, high = upper_idx)
        spec = spec[:, random_idx:random_idx+self.input_length][np.newaxis]

        tag_binary = self.tags[fn].astype(int).reshape(50)
        return spec, tag_binary

    def __len__(self):
        return len(self.fl)


def get_DataLoader(batch_size=32,
                   input_length=15, # [s]
                   spec_path ='/import/c4dm-datasets/rmri_self_att/msd',
                   audio_path='/import/c4dm-03/Databases/songs/',
                   path_to_repo='~/dl4am/',
                   mode='train', 
                   num_workers=20):
    
    data_loader = data.DataLoader(dataset=AudioFolder(input_length, 
                                                      spec_path, 
                                                      audio_path, 
                                                      path_to_repo, 
                                                      mode),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True, # for CUDA
                                  num_workers=num_workers)
    return data_loader
