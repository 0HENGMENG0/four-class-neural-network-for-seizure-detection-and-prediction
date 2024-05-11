import h5py
import numpy as np
import torch

from torch.utils.data import Dataset

import dsp

class MyDataset(Dataset):
    # 构造函数
    def __init__(self, x_tensor, y_tensor):
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor

    # 返回数据集大小
    def __len__(self):
        return self.x_tensor.size(0)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.x_tensor[index], self.y_tensor[index]


def normalize(x, m, s):
    return (x - m) / s


def normalize_to(train):
    m, s = train.mean(), train.std()
    return normalize(train, m, s)

def generate_data(index, type, win_len, overlap_len):
    file_path = '/home/test/zhangheng/data/data_artifact_4class/traintestvalidate/'+index+'_dataset.h5'
    # file_path = '/home/test/wdangan/data_win10_overlap8_cut_3class/4class/'+index+'_dataset.h5'
    # file_path = '/home/test/wdangan/data_win10_overlap8_cut_3class/4_class_noweicha/'+index+'_dataset.h5'
    with h5py.File(file_path, "r") as f:
        x_ = f[type+'_data']
        # x_test = []
        # x_test.append(x_)
        # x_test = np.array(x_test)
        # print("!", x_test.shape)
        # x_t = f[type+'_data']
        # print(x_t)
        # x_ = []
        # x_.append(x_t)
        y_t = f[type+'_label']
        y_ = []
        for i in y_t:
            if i == 3:
                y_.append(1)
            elif i == 2:
                y_.append(3)
            elif i == 1:
                y_.append(0)
            elif i == 4:
                y_.append(2)
            else:
                print("error")
        
        x_ = np.array(x_)
        print("x_.shape", x_.shape)
        x_ = dsp.pre_stft(x_, sample_rate=256, win_len=win_len, overlap_len=overlap_len, cutoff_hz=60)
        print("x_stft.shape", x_.shape)
        x_p1 = x_[:,:,0:40,:]
        x_p2 = x_[:,:,40:58,:]
        x_ = np.concatenate([x_p2,x_p1],axis=2)
        print("x_p1.shape", x_p1.shape)
        print("x_p2.shape", x_p2.shape)
        print("x_cat", x_.shape)
        y_ = np.array(y_)
        print("y_.shape", y_.shape)
        x_ = x_.astype(np.float32)
        y_ = y_.astype(np.int64)
        x_tensor = torch.tensor(x_)
        y_tensor = torch.tensor(y_)
        x_tensor = x_tensor.permute(0, 3, 1, 2)
        # x_tensor = x_tensor.permute(1, 4, 0, 2, 3)
        x_tensor1 = normalize_to(x_tensor)
        print("x_tensor1.shape:", x_tensor1.shape)
        my_dataset = MyDataset(x_tensor1, y_tensor)
        return my_dataset

def generate_data_k_fold_train_withfake(file_path_folder, index, k_list, win_len, overlap_len):
    file_path = file_path_folder + index + '_dataset.h5'
    # file_path = '/home/test/zhangheng/data/20240325_raw_4class/raw_400samples/' + index + '_dataset.h5'
    # file_path_f1 = '/home/test/zhangheng/data/fake/' + index + '_win_10/' + index + '_win_10_fake_data_ictal.h5'
    # file_path_f2 = '/home/test/zhangheng/data/fake/' + index + '_win_10/' + index + '_win_10_fake_label_ictal.h5'
    # file_path = '/home/test/wdangan/data_win10_overlap8_cut_3class/4class/'+index+'_dataset.h5'
    # file_path = '/home/test/wdangan/data_win10_overlap8_cut_3class/4_class_noweicha/'+index+'_dataset.h5'
    with h5py.File(file_path, "r") as f:
        #  h5py.File(file_path_f1,"r") as f1,\
        #  h5py.File(file_path_f2,"r") as f2:
        # print("f.keys():", f.keys())
        # print("f1.keys():", f1.keys())
        # print("f2.keys():", f2.keys())
        print("k_list:", k_list)
        k_0 = k_list[0]
        k_1 = k_list[1]
        k_2 = k_list[2]
        k_3 = k_list[3]

        x_0 = f['data_' + str(k_0)]
        x_1 = f['data_' + str(k_1)]
        x_2 = f['data_' + str(k_2)]
        x_3 = f['data_' + str(k_3)]
        x_ = np.concatenate((x_0,x_1,x_2,x_3),axis=0)
        print("x_concatenate.shape", x_.shape)

        y_0 = f['label_' + str(k_0)]
        y_1 = f['label_' + str(k_1)]
        y_2 = f['label_' + str(k_2)]
        y_3 = f['label_' + str(k_3)]
        y_ = []
        for i in y_0:
            if i == 3:
                y_.append(1)
            elif i == 2:
                y_.append(3)
            elif i == 1:
                y_.append(0)
            elif i == 4:
                y_.append(2)
            else:
                print("error")
        for i in y_1:
            if i == 3:
                y_.append(1)
            elif i == 2:
                y_.append(3)
            elif i == 1:
                y_.append(0)
            elif i == 4:
                y_.append(2)
            else:
                print("error")
        for i in y_2:
            if i == 3:
                y_.append(1)
            elif i == 2:
                y_.append(3)
            elif i == 1:
                y_.append(0)
            elif i == 4:
                y_.append(2)
            else:
                print("error")
        for i in y_3:
            if i == 3:
                y_.append(1)
            elif i == 2:
                y_.append(3)
            elif i == 1:
                y_.append(0)
            elif i == 4:
                y_.append(2)
            else:
                print("error")
        # y_fake = f2['fake_label']
        # for i in y_fake:
        #     if i == 2:
        #         y_.append(3)
        #     # if i == 3:
        #     #     y_.append(1)
        #     # elif i == 2:
        #     #     y_.append(3)
        #     # elif i == 1:
        #     #     y_.append(0)
        #     # elif i == 4:
        #     #     y_.append(2)
        #     else:
        #         print("error")
        
        # x_ = np.array(x_)
        # print("x_.shape", x_.shape)       
        x_ = dsp.pre_stft(x_, sample_rate=256, win_len=win_len, overlap_len=overlap_len, cutoff_hz=60)
        print("x_stft.shape", x_.shape)

        # x_fake = f1['fake_data']
        # print("x_fake.shape", x_fake.shape)
        # x_fake = x_fake[:,:,:,0:60]
        # x_fake = np.transpose(x_fake, (0,1,3,2))
        # print("x_fake.shape", x_fake.shape)

        # x_ = np.concatenate((x_, x_fake), axis=0)
        # print("x_.shape", x_.shape)

        x_p1 = x_[:,:,0:40,:]
        x_p2 = x_[:,:,40:58,:]
        print("x_p1.shape", x_p1.shape)
        print("x_p2.shape", x_p2.shape)
        x_ = np.concatenate([x_p2,x_p1], axis=2)
        print("x_cat", x_.shape)

        y_ = np.array(y_)
        print("y_.shape", y_.shape)
        x_ = x_.astype(np.float32)
        y_ = y_.astype(np.int64)
        x_tensor = torch.tensor(x_)
        y_tensor = torch.tensor(y_)
        x_tensor = x_tensor.permute(0, 3, 1, 2)
        # x_tensor = x_tensor.permute(1, 4, 0, 2, 3)
        x_tensor1 = normalize_to(x_tensor)
        print("x_tensor1.shape:", x_tensor1.shape)
        my_dataset = MyDataset(x_tensor1, y_tensor)
        return my_dataset

def generate_data_k_fold_test(file_path_folder, index, k_index, win_len, overlap_len):
    file_path = file_path_folder + index + '_dataset.h5'
    # file_path = '/home/test/zhangheng/data/20240325_raw_4class/raw_400samples/'+index+'_dataset.h5'
    # file_path = '/home/test/wdangan/data_win10_overlap8_cut_3class/4class/'+index+'_dataset.h5'
    # file_path = '/home/test/wdangan/data_win10_overlap8_cut_3class/4_class_noweicha/'+index+'_dataset.h5'
    with h5py.File(file_path, "r") as f:
        print("f.keys():", f.keys())
        print("k_index:", k_index)
        x_ = f['data_' + str(k_index)]
        print("x_.shape", x_.shape)
        y_t = f['label_' + str(k_index)]
        y_ = []
        for i in y_t:
            if i == 3:
                y_.append(1)
            elif i == 2:
                y_.append(3)
            elif i == 1:
                y_.append(0)
            elif i == 4:
                y_.append(2)
            else:
                print("error")
        
        x_ = np.array(x_)
        print("x_.shape", x_.shape)
        x_ = dsp.pre_stft(x_, sample_rate=256, win_len=win_len, overlap_len=overlap_len, cutoff_hz=60)
        print("x_stft.shape", x_.shape)
        x_p1 = x_[:,:,0:40,:]
        x_p2 = x_[:,:,40:58,:]
        x_ = np.concatenate([x_p2,x_p1],axis=2)
        print("x_p1.shape", x_p1.shape)
        print("x_p2.shape", x_p2.shape)
        print("x_cat", x_.shape)
        y_ = np.array(y_)
        print("y_.shape", y_.shape)
        x_ = x_.astype(np.float32)
        y_ = y_.astype(np.int64)
        x_tensor = torch.tensor(x_)
        y_tensor = torch.tensor(y_)
        x_tensor = x_tensor.permute(0, 3, 1, 2)
        # x_tensor = x_tensor.permute(1, 4, 0, 2, 3)
        x_tensor1 = normalize_to(x_tensor)
        print("x_tensor1.shape:", x_tensor1.shape)
        my_dataset = MyDataset(x_tensor1, y_tensor)
        return my_dataset

