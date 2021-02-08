# coding: utf-8
# adapted from https://github.com/minzwon/self-attention-music-tagging/blob/master/training/data_loader/msd_loader.py
import os
import numpy as np
from torch.utils import data
import pickle as pkl
import librosa
import warnings

class AudioFolder(data.Dataset):
    def __init__(self, input_length, path_to_repo, spec_path, audio_path, trval):
        self.path_to_repo = os.path.expanduser(path_to_repo)
        self.spec_path = os.path.expanduser(spec_path)
        self.audio_path = os.path.expanduser(audio_path)
        self.trval = trval
        self.input_length = input_length
        self.fs = 16000
        self.window = 512
        self.hop = 256
        self.mel = 96
        self.get_songlist()
        self.idmsd_to_id7d = pkl.load(open(os.path.join(self.path_to_repo, 'preprocessing/msd/MSD_id_to_7D_id.pkl'),'rb'))
        self.tags = pkl.load(open(os.path.join(self.path_to_repo, 'preprocessing/msd/msd_id_to_tag_vector.cP'), 'rb'))
        
    def __getitem__(self, index):
        spec = []
        while len(spec) == 0:
            try:
                spec, tag_binary = self.get_spec(index)
            except: # audio not found or broken
                index = np.random.randint(0,high=len(self.fl))
                spec = []
        return spec.astype('float32'), tag_binary.astype('float32')

    def get_songlist(self):
        train = pkl.load(open(os.path.join(self.path_to_repo, 'preprocessing/msd/filtered_list_train.cP'), 'rb'))
        if self.trval == 'TRAIN':
            self.fl = train[:201680]
        elif self.trval == 'VALID':
            self.fl = train[201680:]
        elif self.trval == 'TEST':
            self.fl = pkl.load(open(os.path.join(self.path_to_repo, 'preprocessing/msd/filtered_list_test.cP'), 'rb'))
            
    def compute_melspectrogram(self, audio_fn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x, _ = librosa.core.load(audio_fn, sr=self.fs, res_type='kaiser_fast')           
        return librosa.core.amplitude_to_db(librosa.feature.melspectrogram(x, sr=self.fs, 
                                                                           n_fft=self.window, 
                                                                           hop_length=self.hop, 
                                                                           n_mels=self.mel))

    def get_spec(self, index):
        fn=self.fl[index]
        id7d = self.idmsd_to_id7d[fn]
        audio_fn = os.path.join(self.audio_path,*id7d[:2],id7d+".clip.mp3")
        spec_fn = os.path.join(self.spec_path,*id7d[:2], id7d+'.npy') 
        
        if not os.path.exists(spec_fn):
            spec = self.compute_melspectrogram(audio_fn)
            spec_path = os.path.dirname(spec_fn)
            if not os.path.exists(spec_path):
                os.makedirs(spec_path)
            np.save(open(spec_fn, 'wb'), spec)
        else:
            spec = np.load(spec_fn, mmap_mode='r')
            
        random_idx = np.random.randint(0,high=int(29*self.fs/self.hop)-self.input_length)
        spec = np.array(spec[:, random_idx:random_idx+self.input_length])
            
        tag_binary = self.tags[fn].astype(int).reshape(50)
        return spec, tag_binary

    def __len__(self):
        return len(self.fl)


def get_audio_loader(batch_size,
                     input_length,
                     path_to_repo, 
                     spec_path,
                     audio_path,  
                     trval, 
                     num_workers):
    data_loader = data.DataLoader(dataset=AudioFolder(input_length, path_to_repo, spec_path, audio_path, trval),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader