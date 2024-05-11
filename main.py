# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import argparse

import h5py
import data
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import models, transforms, datasets
import numpy as np
from net import Net
#from spikingjelly.activation_based import functional, encoding

def infer_test(test_dataloader, patient_index, lstm_hidden, lstm_step, fc1_out):
    device = torch.device('cuda')
    model = Net(lstm_hidden, lstm_step, fc1_out).to(device)
    # model = torch.load('net_model_chb06.pth')
    # model.load_state_dict(torch.load('net.pth'))
    # model.load_state_dict(model_dict)
    # print(*list(model.children())[-3:-2])

    # print(model)
    model.load_state_dict(torch.load('net.pth'))
    model.eval()
    with torch.no_grad():
        # test
        total_correct = 0
        total_num = 0
        for idx, (x, label) in enumerate(test_dataloader, 1):
            # if idx <= 100:
            x, label = x.to(device), label.to(device)

            logits = model(x)
            pred = logits.argmax(dim=1)
            print("probability:",logits)

            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)
            acc = total_correct / total_num
            # print(idx, total_correct, total_num, 'test acc:', acc)
            # print(pred)
    # print(iter(cifar_test).__next__()[0].shape)
    print('test acc:', acc)

def train_test(epoch_arg, train_dataloader, validate_dataloader, patient_index, \
               out_channels1, out_channels2, out_channels3, out_channels4, lstm_hidden, lstm_step, fc1_out):
    device = torch.device('cpu')
    model = Net(out_channels1, out_channels2, out_channels3, out_channels4, lstm_hidden, lstm_step, fc1_out).to(device)
    print(model)
    # model = torch.load('net_model_chb06.pth')
    # model.load_state_dict(torch.load('net.pth'))
    # model.load_state_dict(model_dict)
    # print(*list(model.children())[-3:-2])

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr=1e-5)

    max_acc = 0.0
    count00 = 0
    count01 = 0
    count02 = 0
    count03 = 0
    count10 = 0
    count11 = 0
    count12 = 0
    count13 = 0
    count20 = 0
    count21 = 0
    count22 = 0
    count23 = 0
    count30 = 0
    count31 = 0
    count32 = 0
    count33 = 0

    jvbuzuiyou_count = 0
    acc_last = 0
    for epoch in range(epoch_arg):
        model.train()
        for batchidx, (x, label) in enumerate(train_dataloader, 1):
            # if batchidx <= 300:
                x, label = x.to(device), label.to(device)

                logits = model(x)
                # print(logits.shape)
                # print(label.shape)
                loss = criterion(logits, label)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #functional.reset_net(model)
                # print("epoch:", epoch, "index:", batchidx)
            # else:
            #     continue
        # print(epoch, 'loss:', loss.item())

        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            count_16 = np.zeros(16)
            for idx, (x, label) in enumerate(validate_dataloader, 1):
                # if idx <= 100:
                x, label = x.to(device), label.to(device)

                logits = model(x)
                # print(logits)
                pred = logits.argmax(dim=1)

                for i in range(pred.shape[0]):
                    count_16[0] += ((pred[i]==0) and (label[i]==0)).float().item()
                    count_16[1] += ((pred[i]==0) and (label[i]==1)).float().item()
                    count_16[2] += ((pred[i]==0) and (label[i]==2)).float().item()
                    count_16[3] += ((pred[i]==0) and (label[i]==3)).float().item()
                    count_16[4] += ((pred[i]==1) and (label[i]==0)).float().item()
                    count_16[5] += ((pred[i]==1) and (label[i]==1)).float().item()
                    count_16[6] += ((pred[i]==1) and (label[i]==2)).float().item()
                    count_16[7] += ((pred[i]==1) and (label[i]==3)).float().item()
                    count_16[8] += ((pred[i]==2) and (label[i]==0)).float().item()
                    count_16[9] += ((pred[i]==2) and (label[i]==1)).float().item()
                    count_16[10] += ((pred[i]==2) and (label[i]==2)).float().item()
                    count_16[11] += ((pred[i]==2) and (label[i]==3)).float().item()
                    count_16[12] += ((pred[i]==3) and (label[i]==0)).float().item()
                    count_16[13] += ((pred[i]==3) and (label[i]==1)).float().item()
                    count_16[14] += ((pred[i]==3) and (label[i]==2)).float().item()
                    count_16[15] += ((pred[i]==3) and (label[i]==3)).float().item()

                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                acc = total_correct / total_num
                # print(epoch, idx, 'test acc:', acc)
                # print(pred)
                #functional.reset_net(model)

            acc = total_correct / total_num
            print(epoch, 'validate acc:', acc)
            #     # # PATH="model/test.pth"
            if acc > max_acc:
                torch.save(model, "net_model.pth")
                torch.save(model.state_dict(), "net.pth")
                max_acc = acc
                count00 = count_16[0]
                count01 = count_16[1]
                count02 = count_16[2]
                count03 = count_16[3]
                count10 = count_16[4]
                count11 = count_16[5]
                count12 = count_16[6]
                count13 = count_16[7]
                count20 = count_16[8]
                count21 = count_16[9]
                count22 = count_16[10]
                count23 = count_16[11]
                count30 = count_16[12]
                count31 = count_16[13]
                count32 = count_16[14]
                count33 = count_16[15]
            
            # 提前叫停
            if max_acc == 1:
                break
            # 如果陷入局部最优，提前叫停
            if acc == acc_last:
                jvbuzuiyou_count = jvbuzuiyou_count + 1
            else:
                jvbuzuiyou_count = 0
            if jvbuzuiyou_count == 10:
                break
            acc_last = acc
    # print(logits)
    print("max_acc = ", max_acc)
    with open("acc_log.txt", 'a') as acc_file:
        acc_file.write(str(max_acc) + '\n')
        acc_file.write(str(count00) + '\t' + str(count01) + '\t' + str(count02) + '\t' + str(count03) + '\t' + \
                       str(count10) + '\t' + str(count11) + '\t' + str(count12) + '\t' + str(count13) + '\t' + \
                       str(count20) + '\t' + str(count21) + '\t' + str(count22) + '\t' + str(count23) + '\t' + \
                       str(count30) + '\t' + str(count31) + '\t' + str(count32) + '\t' + str(count33) + '\t' + \
                       '\n')
        acc_file.close()

def infer_train_test(train_dataloader, test_dataloader, validate_dataloader, patient_index, lstm_hidden, lstm_step, fc1_out):
    device = torch.device('cuda')
    model = Net(lstm_hidden, lstm_step, fc1_out).to(device)
    max_acc = 0.0
    # model = torch.load('net_model_chb06.pth')
    # model.load_state_dict(torch.load('net.pth'))
    # model.load_state_dict(model_dict)
    # print(*list(model.children())[-3:-2])

    # ------------------------------ infer ------------------------------
    # print(model)
    model.load_state_dict(torch.load('net.pth'))
    model.eval()
    with torch.no_grad():
        # test
        total_correct = 0
        total_correct_pre = 0
        total_correct_post = 0
        total_correct_inter = 0
        total_num = 0
        total_pre = 0
        total_post = 0
        total_inter = 0
        for idx, (x, label) in enumerate(test_dataloader, 1):
            # if idx <= 100:
            x, label = x.to(device), label.to(device)

            logits = model(x)
            pred = logits.argmax(dim=1)
            print(pred.shape)
            print(label.shape)

            correct = torch.eq(pred, label).float().sum().item()
            correct_pre = 0
            correct_post = 0
            correct_inter = 0
            for i in range(pred.shape[0]):
                correct_pre += ((pred[i]==0) and (label[i]==0)).float().item()
                correct_post += ((pred[i]==1) and (label[i]==1)).float().item()
                correct_inter += ((pred[i]==2) and (label[i]==2)).float().item()
            print("correct_pre:",correct_pre)
            total_correct += correct
            total_correct_pre += correct_pre
            total_correct_post += correct_post
            total_correct_inter += correct_inter
            mask0 = label==0
            mask1 = label==1
            mask2 = label==2
            total_pre += mask0.sum().item()
            total_post += mask1.sum().item()
            total_inter += mask2.sum().item()
            print("total_pre:",total_pre)
            total_num += x.size(0)
            acc = total_correct / total_num
            acc_pre = total_correct_pre / total_pre
            acc_post = total_correct_post / total_post
            acc_inter = total_correct_inter / total_inter
            # print(idx, total_correct, total_num, 'test acc:', acc)
            # print(pred)
    # print(iter(cifar_test).__next__()[0].shape)
    print('test acc:', acc)
    print('test acc_pre:', acc_pre)
    print('test acc_post:', acc_post)
    print('test acc_inter:', acc_inter)
    max_acc = acc

    # ------------------------------ train ------------------------------
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr=1e-5)

    print("max_acc = ", max_acc)
    jvbuzuiyou_count = 0
    acc_last = 0
    for epoch in range(200):
        model.train()
        for batchidx, (x, label) in enumerate(train_dataloader, 1):
            # if batchidx <= 300:
                x, label = x.to(device), label.to(device)

                logits = model(x)
                # print(logits.shape)
                # print(label.shape)
                loss = criterion(logits, label)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print("epoch:", epoch, "index:", batchidx)
            # else:
            #     continue
        # print(epoch, 'loss:', loss.item())

        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_correct_pre = 0
            total_correct_post = 0
            total_correct_inter = 0
            total_num = 0
            total_pre = 0
            total_post = 0
            total_inter = 0
            total_inter2pre = 0
            total_inter2post = 0
            for idx, (x, label) in enumerate(validate_dataloader, 1):
                # if idx <= 100:
                x, label = x.to(device), label.to(device)

                logits = model(x)
                # print(logits)
                pred = logits.argmax(dim=1)

                correct = torch.eq(pred, label).float().sum().item()
                correct_pre = 0
                correct_post = 0
                correct_inter = 0
                err_inter2pre = 0
                err_inter2post = 0
                for i in range(pred.shape[0]):
                    correct_pre += ((pred[i]==0) and (label[i]==0)).float().sum().item()
                    correct_post += ((pred[i]==1) and (label[i]==1)).float().sum().item()
                    correct_inter += ((pred[i]==2) and (label[i]==2)).float().sum().item()
                    err_inter2pre += ((pred[i]==0) and (label[i]==2)).float().sum().item()
                    err_inter2post += ((pred[i]==1) and (label[i]==2)).float().sum().item()
                total_correct += correct
                total_correct_pre += correct_pre
                total_correct_post += correct_post
                total_correct_inter += correct_inter
                total_inter2pre += err_inter2pre
                total_inter2post += err_inter2post
                mask0 = label==0
                mask1 = label==1
                mask2 = label==2
                total_pre += mask0.sum().item()
                total_post += mask1.sum().item()
                total_inter += mask2.sum().item()
                total_num += x.size(0)
                acc = total_correct / total_num
                acc_pre = total_correct_pre / total_pre
                acc_post = total_correct_post / total_post
                acc_inter = total_correct_inter / total_inter
                err_inter2pre = total_inter2pre / total_inter
                err_inter2post = total_inter2post / total_inter
                # print(epoch, idx, 'test acc:', acc)
                # print(pred)

            acc = total_correct / total_num
            acc_pre = total_correct_pre / total_pre
            acc_post = total_correct_post / total_post
            acc_inter = total_correct_inter / total_inter
            err_inter2pre = total_inter2pre / total_inter
            err_inter2post = total_inter2post / total_inter
            print(epoch, 'validate acc:', acc,acc_pre,acc_post,acc_inter)
            print(epoch, 'validate err:', err_inter2pre, err_inter2post)
            #     # # PATH="model/test.pth"
            if acc > max_acc:
                torch.save(model, "net_model.pth")
                torch.save(model.state_dict(), "net.pth")
                max_acc = acc
            # 如果陷入局部最优，提前叫停
            if acc == acc_last:
                jvbuzuiyou_count = jvbuzuiyou_count + 1
            else:
                jvbuzuiyou_count = 0
            if jvbuzuiyou_count == 10:
                break
            acc_last = acc

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Epilepsy_prediction_4class_v2')
    parser.add_argument('-path', default='/home/test/zhangheng/data/0%_artifact/', type=str, help='file_path of folder')
    parser.add_argument('-chb', default='chb01', type=str, help='choose which chb')
    parser.add_argument('-k', default=0, type=int, help='k_fold')
    parser.add_argument('-epochs', default=120, type=int, help='number of total epochs to run')
    parser.add_argument('-out_channels1', default=64, type=int, help='out_channels1')
    parser.add_argument('-out_channels2', default=128, type=int, help='out_channels2')
    parser.add_argument('-out_channels3', default=256, type=int, help='out_channels3')
    parser.add_argument('-out_channels4', default=512, type=int, help='out_channels4')
    parser.add_argument('-lstm_hidden', default=256, type=int, help='lstm_hidden')
    parser.add_argument('-fc1_out', default=128, type=int, help='fc1_out')

    # parser.add_argument('-data_path', default='', type=str, help='input data path')
    # parser.add_argument('-device', default='cuda:0', help='device')
    # parser.add_argument('-b', default=128, type=int, help='batch size')
    # parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    # parser.add_argument('-amp', action='store_false', help='automatic mixed precision training')
    # parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam')
    # parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    # parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    # parser.add_argument('-tr_cyc', default=1, type=int, help='train cycles')
    args = parser.parse_args()
    print(args)

    file_path_folder = args.path

    patient_index = args.chb
    # patient_index = 'chb10'
    print("patient_index:", patient_index)

    k_index = args.k
    # k_index = 2
    if(k_index == 0):
        train_k_list = [1, 2, 3, 4]
    elif(k_index == 1):
        train_k_list = [0, 2, 3, 4]
    elif(k_index == 2):
        train_k_list = [0, 1, 3, 4]
    elif(k_index == 3):
        train_k_list = [0, 1, 2, 4]
    elif(k_index == 4):
        train_k_list = [0, 1, 2, 3]
    else:
        print("error")
    # print(k_index)
    # print(train_k_list)
    # for i in train_k_list:
    #     print(i)

    epoch_arg=args.epochs

    batch_size = 64
    win_len = 256
    overlap_len = 128

    out_channels1=args.out_channels1
    out_channels2=args.out_channels2
    out_channels3=args.out_channels3
    out_channels4=args.out_channels4

    lstm_hidden = args.lstm_hidden
    lstm_step = int(2560 / (win_len - overlap_len)) + 1
    fc1_out = args.fc1_out

    train_dataset = data.generate_data_k_fold_train_withfake(file_path_folder=file_path_folder, index=patient_index, k_list=train_k_list, win_len=win_len, overlap_len=overlap_len)
    train_dataloader = DataLoader(dataset=train_dataset,  # 传入的数据集, 必须参数
                                  batch_size=batch_size,  # 输出的batch大小
                                  shuffle=True,  # 数据是否打乱
                                  num_workers=0)  # 进程数, 0表示只有主进程

    test_dataset = data.generate_data_k_fold_test(file_path_folder=file_path_folder, index=patient_index, k_index=k_index, win_len=win_len, overlap_len=overlap_len)
    test_dataloader = DataLoader(dataset=test_dataset,  # 传入的数据集, 必须参数
                                  batch_size=batch_size,  # 输出的batch大小
                                  shuffle=True,  # 数据是否打乱
                                  num_workers=0)  # 进程数, 0表示只有主进程

    # train_dataset = data.generate_data(patient_index, 'train', win_len=win_len, overlap_len=overlap_len)
    # train_dataloader = DataLoader(dataset=train_dataset,  # 传入的数据集, 必须参数
    #                               batch_size=batch_size,  # 输出的batch大小
    #                               shuffle=True,  # 数据是否打乱
    #                               num_workers=0)  # 进程数, 0表示只有主进程
    # test_dataset = data.generate_data(patient_index, 'test', win_len=win_len, overlap_len=overlap_len)
    # test_dataloader = DataLoader(dataset=test_dataset,  # 传入的数据集, 必须参数
    #                               batch_size=batch_size,  # 输出的batch大小
    #                               shuffle=True,  # 数据是否打乱
    #                               num_workers=0)  # 进程数, 0表示只有主进程
    # validate_dataset = data.generate_data(patient_index, 'validate', win_len=win_len, overlap_len=overlap_len)#
    # validate_dataloader = DataLoader(dataset=validate_dataset,  # 传入的数据集, 必须参数
    #                              batch_size=batch_size,  # 输出的batch大小
    #                              shuffle=True,  # 数据是否打乱
    #                              num_workers=0)  # 进程数, 0表示只有主进程

    # infer_test(test_dataloader=test_dataloader, patient_index=patient_index, lstm_hidden=lstm_hidden, lstm_step=lstm_step, fc1_out=fc1_out)
    train_test(epoch_arg=epoch_arg, train_dataloader=train_dataloader, validate_dataloader=test_dataloader, patient_index=patient_index, \
               out_channels1=out_channels1, out_channels2=out_channels2, out_channels3=out_channels3, out_channels4=out_channels4, \
                lstm_hidden=lstm_hidden, lstm_step=lstm_step, fc1_out=fc1_out)
    # infer_train_test(train_dataloader=train_dataloader, test_dataloader=test_dataloader, validate_dataloader=test_dataloader, patient_index=patient_index,\
    #              lstm_hidden=lstm_hidden, lstm_step=lstm_step, fc1_out=fc1_out)

