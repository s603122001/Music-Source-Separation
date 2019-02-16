import torch
import librosa
from model import ARC, Enhancement

def SourceSeparation(song_path, sr, model_path = ["model/ARC_2", "model/Enhancement_voice_2", "model/Enhancement_others_2"]):
    arc = ARC()
    arc.cuda()
    arc.load_state_dict(torch.load(model_path[0]))
    arc.eval()

    en_voice = Enhancement()
    en_voice.cuda()
    en_voice.load_state_dict(torch.load(model_path[1]))
    en_voice.eval()

    en_others = Enhancement()
    en_others.cuda()
    en_others.load_state_dict(torch.load(model_path[2]))
    en_others.eval()
    
    x, sr = librosa.load(song_path, sr = sr)
    
    stft, y_voice, y_others = predict_song(x, arc, en_voice, en_others)
    
    return stft, y_voice, y_others