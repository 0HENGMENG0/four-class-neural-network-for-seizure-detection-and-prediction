import numpy as np
from scipy import signal
import math

def pre_stft(data_signal, sample_rate, win_len, overlap_len, cutoff_hz):
    data_num = data_signal.shape[0]
    electro_num = data_signal.shape[1]
    cutoff = math.ceil(cutoff_hz * win_len / sample_rate)
    # print("cutoff:", cutoff)

    electrode_features_i = []
    electrode_features_j = []
    for i in range(data_num):
        electrode_features_j.clear()
        for j in range(electro_num):
            # if i == data_num-1 and j == electro_num-1:
            #     print(data_signal[i][j].shape)
            f, t, Zxx = signal.stft(data_signal[i][j], sample_rate, nperseg=win_len, noverlap=overlap_len)
            # print("Zxx.shape", Zxx.shape)
            Zxx = np.abs(Zxx[0:cutoff])
            # print("Zxx.shape", Zxx.shape)
            # electrode_feature = np.transpose(Zxx)
            electrode_features_j.append(Zxx)
        electrode_features_j_np = np.array(electrode_features_j)
        # print("electrode_features_j_np.shape", electrode_features_j_np.shape)
        electrode_features_i.append(electrode_features_j_np)
    electrode_features_i_np = np.array(electrode_features_i)
    # print("electrode_features_i_np.shape", electrode_features_i_np.shape)

    # electrode_features = [electrode_features_i_np]
    # electrode_features_np = np.array(electrode_features)
    # print("electrode_features_np.shape", electrode_features_np.shape)

    return electrode_features_i_np
