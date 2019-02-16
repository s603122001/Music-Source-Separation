from os import listdir
import torch
import librosa
import numpy as np

def datasets_importer(data_DIR, data_type):
    sets = []
    print("Load from %s" % data_DIR)
    for filename in os.listdir(data_DIR):
          if(data_type in filename):
            sets.append(data_DIR+filename)
    print("Load successfully!")
    return sets

def STFT(x, n_fft = 2048, hop_length = 1024, GPU_avail = "True", mode = "mag"):
    '''
    GPU_avail == True: use torch.stft. 
                   == False: use librosa.stft
    '''
    if(GPU_avail):
        x_c = torch.from_numpy(x)
        w = torch.hann_window(n_fft).double()
        out = torch.stft(x_c, n_fft = n_fft, hop_length=hop_length,  window = w)
        if(mode == "mag"):
            s = torch.pow((out[:, :, 0]**2 + out[:, :, 1]**2), 0.5)
        elif(mode == "stft"):
            s = (out[:, :, 0].numpy() + 1j*out[:, :, 1].numpy())
    else: 
        if(mode == "mag"):
            s = librosa.stft(x, n_fft = n_fft, hop_length = hop_length)
            s = np.abs(s)
        elif(mode == "stft"):
            s = librosa.stft(x, n_fft = n_fft, hop_length = hop_length)
        s = torch.from_numpy(s)
        
    return s

def save_model(model, model_name):
    torch.save(model.state_dict(), model_name)
    
def load_model(model, model_name):
    model.load_state_dict(torch.load(model_name))