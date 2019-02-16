import librosa
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from utils import STFT, datasets_importer

class CustomDatasetFromNpz(Dataset):
    def __init__(self, path, split = 75, samples = 5000, sample_rate = 44100, 
                 length = 255, hop_size = 1024, transforms = None, mode = "train", GPU_avail = True):
        """
        split: points to split the dataset. TODO: cross validation
        samples: how many random samples we want for one epoch
        lengh: how many frames in one sample
        """
        d = []
        for i in range(1, 5):
            d.append(np.sort(datasets_importer(path, "wave_track_" + str(i) + ".npz")))
        
        dd = [[], [], [], []]
        if(mode == "train"):
            for i in range(4):
                for ii in tqdm.tqdm(range(len(d[i][:split]))): 
                    dd[i].append(np.load(d[i][ii])['arr_0'])
        elif(mode == "validate"):
            for i in range(4):
                for ii in tqdm.tqdm(range(len(d[i][split:]))): 
                    dd[i].append(np.load(d[i][ii])['arr_0'])
       
        self.data = dd
        self.split = split
        self.samples = samples
        self.sample_rate = 44100
        self.length = length
        self.hop_size = hop_size
        self.mode = mode
        self.GPU_avail = GPU_avail
        
    def __getitem__(self, index):
        # Random generate
        # Target: voice and others
        # TODO: should we crop the number of frames to even number? 
        if(self.mode == 'train'):
            track_sample = np.random.randint(len(self.data[0]), size =  4)
            track_gain = 0.01* np.random.randint(70, 101, size = 4)
        
            num_samples = self.hop_size*self.length
            mix = np.zeros(num_samples)
            label = None
            for i in range(4):            
                start = np.random.randint(len(self.data[i][track_sample[0]]) - num_samples)
                part = self.data[i][track_sample[0]][start:start + num_samples]
                part = track_gain[i] * part
            
                if(i == 3):
                    label = torch.log1p(STFT(part, hop_length = self.hop_size, GPU_avail = self.GPU_avail))
                    label_others =  torch.log1p(STFT(mix, hop_length = self.hop_size, GPU_avail = self.GPU_avail))
                    label = torch.cat((label, label_others))
                mix += part
        
            mix_stft = torch.log1p(STFT(mix, hop_length = self.hop_size, GPU_avail = self.GPU_avail))
        
            return (mix_stft.float(), label.float())
        
        elif(self.mode == "validate"):
            part = self.data[0][index]
            mix = np.zeros(len(part))
            for i in range(4):            
                part = self.data[i][index]
                if(i == 3):
                    label = torch.log1p(STFT(part, hop_length = self.hop_size, GPU_avail = self.GPU_avail))
                    label_others =  torch.log1p(STFT(mix, hop_length = self.hop_size, GPU_avail = self.GPU_avail))
                    label = torch.cat((label, label_others))
                
                mix += part
                
            mix_stft = torch.log1p(STFT(mix, hop_length = self.hop_size, GPU_avail = self.GPU_avail))
            
            return (mix_stft.float(), label.float())
    
    def __len__(self):
        return self.samples