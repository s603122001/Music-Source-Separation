import librosa
import numpy as np
import torch
import tqdm

def predict_slice(x_part, model_arc, model_en0, model_en1, hop_length = 1024):
    stft = librosa.stft(x_part, hop_length = hop_length)
    stft_mag = np.log1p(np.abs(stft))
    phase = np.angle(stft)

    out = model_arc(torch.from_numpy(stft_mag[None,:,:]).cuda())
    out_voice = model_en0(out[:, :1025, :])
    out_others = model_en1(out[:, 1025:, :])

    # Mag spectrogram for voice
    out_voice = out_voice.cpu().detach().numpy()[0]
    out_voice  = np.exp(out_voice) - 1

    # Mag spectrogram for others
    out_others = out_others.cpu().detach().numpy()[0]
    out_others  = np.exp(out_others) - 1

    # Get phase from original mixture
    phase = np.angle(librosa.stft(x_part, hop_length = hop_length))

    # ISTFT for voice
    out_voice = out_voice * np.exp(1j*phase)
    y_voice = librosa.istft(out_voice , hop_length = hop_length)

    # ISTFT for others
    out_others = out_others * np.exp(1j*phase)
    y_others = librosa.istft(out_others, hop_length = hop_length)

    return out_voice, out_others, y_voice, y_others, stft

def predict_song(x, model_arc, model_en0, model_en1, hop_length = 1024):
    #TODO: overlap prediction
    GPU_avail = True
    win_len = 1024*255
    pad = 1024*111 # 112 + 32 + 112
    hop = 1024*31
    out_voice_total = np.zeros((1025, 1))
    out_others_total = np.zeros((1025, 1))
    stft_original = np.zeros((1025, 1))
    l = len(x)
    x_pad = np.pad(x, (0, win_len), mode = "constant")

    for i in tqdm.tqdm(range(0, l, win_len)):
        part = x_pad[i:i + win_len]
        o_v, o_o, y_v, y_o, stft_ori = predict_slice(part, model_arc, model_en0, model_en1)
    
        out_voice_total = np.concatenate((out_voice_total, o_v[:, :]), axis = 1)
        out_others_total = np.concatenate((out_others_total, o_o[:, :]), axis = 1)
        stft_original = np.concatenate((stft_original, stft_ori), axis = 1)
    
    out_voice_total = out_voice_total[:, 1:]
    out_others_total = out_others_total[:, 1:]
    stft_original = stft_original[:, 1:]
    
    est = MWF(out_voice_total, out_others_total, stft_original)
    y_voice = librosa.istft(est[0] , hop_length =  hop_length)
    y_others = librosa.istft(est[1] , hop_length =  hop_length)
                            
    return est, y_voice, y_others


def MWF(source0, source1, stft_original, iter_n = 3, M = 256):
    #  Multi-channel Wiener filtering
    
    # Initilization
    stack = np.stack((source0, source1))
    original_stack = np.stack((stft_original, stft_original))
    v = 0.5 * np.power(np.abs(stack), 2)

    est = np.array(stack)
    est_new = np.zeros(shape = est.shape, dtype = 'complex128')
    l = stft_original.shape[-1]

    # Estimate
    for i in range(iter_n):
        m = M
        for ii in range(0, l, m):
            if(ii + m  >= l):
                m = l - ii
            # Compute spatial covarience matrix
            R = np.sum(est[:, :, ii:ii + m]*est[:, :, ii:ii + m].conj(), axis = 2)
            
            # Second term to prevent dividing by zero
            v_sum = np.sum(v[:, :, ii:ii + m], axis = 2) + 0.0000001*np.ones(shape = (2, v.shape[1]))
            R /= v_sum
    
            # Estimate stft
            for iii in range(m):
                d = (np.sum(v[:, :, ii + iii]*R, axis = 0) + 0.0000001*np.ones(shape = (R.shape[-1])))
                est_new[:, :, ii + iii] = v[:, :, ii + iii]*R/d * original_stack[:, :, ii + iii]
     
        # Update
        est = np.array(est_new)
        v = 0.5 * np.power(np.abs(est), 2)
        
    return est