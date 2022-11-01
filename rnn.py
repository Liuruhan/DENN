import argparse
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error

#batch_size = 5
#batch_time = 10
#train_p = 0.8

parser = argparse.ArgumentParser('Rnn demo')
parser.add_argument('--method', type=str, choices=['lstm', 'rnn', 'gru'], default='gru')
parser.add_argument('--train_prob', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--input_size', type=int, default=3)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--test_freq', type=int, default=5)
parser.add_argument('--viz', action='store_true', default=True)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

def data_handle(data_path, args):
    read = np.loadtxt(data_path, delimiter=',', skiprows=1)[:, 1:(1+args.input_size)]

    data = np.zeros((1, read.shape[0]-1, args.input_size))
    all_data = np.zeros((1, int(read.shape[0]/1)-1, args.input_size))
    all_target = np.zeros((1, int(read.shape[0]/1)-1, args.input_size))
    for i in range(all_data.shape[1]):
        each_batch_time = read[i*1:(i+1)*1, :]
        all_data[:, i, :] = read[i*1:(i+1)*1, :]
        all_target[:, i, :] = read[(i + 1) * 1:(i + 2) * 1, :]
    all_data = torch.from_numpy(all_data)
    all_target = torch.from_numpy(all_target)
    for i in range(data.shape[1]):
        data[:, i, :] = read[i:i+1, :]

    Data = data[:, 0:data.shape[1]-1, :]
    Target = data[:, 1:data.shape[1], :]
    #all_data = Data[:,0:int((data.shape[1]-1)/args.batch_size)*args.batch_size, :]
    #all_target = Target[:,0:int((data.shape[1]-1)/args.batch_size)*args.batch_size, :]
    #all_data = np.reshape(all_data, (all_data.shape[0], int(all_data.shape[1]/args.batch_size), args.batch_size, all_data.shape[2]))
    #all_target = np.reshape(all_target,
    #    (all_target.shape[0], int(all_target.shape[1] / args.batch_size), args.batch_size, all_target.shape[2]))
    #all_data = all_data.permute(1, 0, 2, 3)
    #all_target = all_target.permute(1, 0, 2, 3)

    train_len = Data.shape[1]-args.train_prob
    Train_data = Data[:, 0:train_len, :]
    Train_target = Target[:, 0:train_len, :]
    Test_data = Data[:, train_len:Data.shape[1], :]
    Test_target = Target[:, train_len:Data.shape[1], :]
    print('Test:', Test_data.shape, Test_target.shape, 'Train:', Train_data.shape, Train_target.shape)

    train_data = torch.from_numpy(Train_data)
    train_target = torch.from_numpy(Train_target)
    test_data = torch.from_numpy(Test_data)
    test_target = torch.from_numpy(Test_target)

    train_data = train_data.permute(1, 0, 2)
    train_target = train_target.permute(1, 0, 2)
    test_data = test_data.permute(1, 0, 2)
    test_target = test_target.permute(1, 0, 2)

    return train_data, train_target, test_data, test_target, all_data, all_target

class RNN(nn.Module):
    def __init__(self, input_size, method='lstm'):
        super(RNN, self).__init__()
        self.method = method
        if method == 'rnn':
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=16,
                num_layers=1,
                batch_first=True
            )
        elif method == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=16,
                num_layers=1,
                batch_first=True
            )
        else:
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=16,
                num_layers=1,
                batch_first=True
            )
        self.out = nn.Sequential(
            nn.Linear(16, 3)
        )

    def forward(self, x):
        if self.method == 'lstm':
            r_out, (h_n, h_c) = self.rnn(x)
            out = self.out(r_out)
        elif self.method == 'gru' or self.method == 'rnn':
            r_out, h_n = self.rnn(x)
            out = self.out(r_out)
        return out

def vis(output, target, itr, data_file, method_name):
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_I = fig.add_subplot(131, frameon=False)
    ax_R = fig.add_subplot(132, frameon=False)
    ax_D = fig.add_subplot(133, frameon=False)
    plt.show(block=False)

    ax_I.cla()
    ax_I.set_title('I')
    ax_I.set_xlabel('t')
    ax_I.set_ylabel('I(t)')
    ax_I.plot(range(output.shape[0]), output[:, 0], 'b--')
    ax_I.plot(range(target.shape[0]), target[:, 0], 'g-')
    # ax_phase.set_xlim(-2, 2)
    # ax_phase.set_ylim(-2, 2)

    ax_R.cla()
    ax_R.set_title('R')
    ax_R.set_xlabel('t')
    ax_R.set_ylabel('R(t)')
    ax_R.plot(range(output.shape[0]), output[:, 1], 'b--')
    ax_R.plot(range(target.shape[0]), target[:, 1], 'g-')

    ax_D.cla()
    ax_D.set_title('D')
    ax_D.set_xlabel('t')
    ax_D.set_ylabel('D(t)')
    ax_D.plot(range(output.shape[0]), output[:, 2], 'b--')
    ax_D.plot(range(target.shape[0]), target[:, 2], 'g-')
    fig.tight_layout()
    plt.savefig('rnn/'+data_file+'_'+method_name+'_'+str(itr)+'.png')
    np.savetxt('rnn/'+data_file+'_'+method_name+'_'+str(itr)+'.csv', output, delimiter=',')
    print('AveR2_I:', r2_score((target[:, 0]), (output[:, 0])))
    print('AveR2_R:', r2_score(target[:, 1], output[:, 1]))
    print('AveR2_D:', r2_score(target[:, 2], output[:, 2]))
    print('AveMSE_I:',
          mean_squared_error((target[:, 0]), (output[:, 0])))
    print('AveMSE_R:', mean_squared_error(target[:, 1], output[:, 1]))
    print('AveMSE_D:', mean_squared_error(target[:, 2], output[:, 2]))

    return
#def fetch_batch():
def train(data_file, method_name):
    Train_data, Train_target, Test_data, Test_target, All_data, All_target = data_handle(data_file, args)
    print('Test:', Test_data.size(), Test_target.size(), 'Train:', Train_data.size(), Train_target.size())
    trainDateset = TensorDataset(Train_data, Train_target)
    testDateset = TensorDataset(Test_data, Test_target)
    trainLoader = DataLoader(trainDateset, batch_size=args.batch_size, shuffle=True)
    testLoader = DataLoader(testDateset, batch_size=args.batch_size, shuffle=True)

    rnn = RNN(args.input_size, method_name)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=args.lr)  # optimize all cnn parameters
    loss_func = nn.MSELoss()

    for step in range(args.niters):
        time = 0
        loss_value = 0
        for tx, ty in trainLoader:

            input_x = tx.permute(1, 0, 2).float()
            output_y = ty.permute(1, 0, 2).float()
            output = rnn(input_x)
            loss = loss_func(output, output_y)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # back propagation, compute gradients
            optimizer.step()
            time += 1
            loss_value += loss.item()
            #print('time:', time, 'loss:', loss.item())
        print('data_file:', data_file, 'method_name:', method_name, 'Iteration:', step, 'AveLoss:', loss_value/time)
        if step % args.test_freq:
            loss_value = 0
            for tx, ty in testLoader:
                input_x = tx.permute(1, 0, 2).float()
                output_y = ty.permute(1, 0, 2).float()
                output = rnn(input_x)
                loss = loss_func(output, output_y)
                loss_value += loss.item()
                #print('val_time:', time, 'val_loss:', loss.item())
            print('Iteration:', step, 'AveValLoss:', loss_value / time)
            all_output = rnn(All_data.float())
            val_output = rnn(Test_data.float())
            all_loss = loss_func(all_output, All_target.float())
            all_out_np = all_output.detach().numpy()
            all_tar_np = All_target.float().numpy()
            val_out_np = val_output.detach().numpy()
            val_tar_np = Test_target.float().numpy()
            out_np = np.zeros((all_out_np.shape[0]*all_out_np.shape[1], all_out_np.shape[2]))
            tar_np = np.zeros((all_tar_np.shape[0]*all_tar_np.shape[1], all_tar_np.shape[2]))
            val_out = np.zeros((val_out_np.shape[0]*val_out_np.shape[1], val_out_np.shape[2]))
            val_tar = np.zeros((val_tar_np.shape[0]*val_tar_np.shape[1], val_tar_np.shape[2]))
            for i in range(all_out_np.shape[1]):
                out_np[i*1:(i+1)*1,:] = all_out_np[:, i, :]
            for i in range(all_tar_np.shape[1]):
                tar_np[i*1:(i+1)*1,:] = all_tar_np[:, i, :]

            for i in range(val_out_np.shape[1]):
                val_out[i*1:(i+1)*1,:] = val_out_np[i, :, :]
            for i in range(val_tar_np.shape[1]):
                val_tar[i*1:(i+1)*1,:] = val_tar_np[i, :, :]
            vis(out_np, tar_np, step, data_file, method_name)
            #print(val_tar[:, 1].shape, val_out[:, 1].shape)
            print('ValR2_I:', r2_score((val_tar[:, 0]), (val_out[:, 0])))
            print('ValR2_R:', r2_score(val_tar[:, 1], val_out[:, 1]))
            print('ValR2_D:', r2_score(val_tar[:, 2], val_out[:, 2]))
            print('ValMSE_I:',
          mean_squared_error((val_tar[:, 0]), (val_out[:, 0])))
            print('ValMSE_R:', mean_squared_error(val_tar[:, 1], val_out[:, 1]))
            print('ValMSE_D:', mean_squared_error(val_tar[:, 2], val_out[:, 2]))

            torch.save(rnn, 'rnn.pkl')
    return

if __name__ == '__main__':
    data_file = ['Italy_IRD.csv', 'USA_IRD.csv', 'Columbia_IRD.csv', 'South_africa_IRD.csv', 'wuhan_IRD.csv', 'Piedmont_IRD.csv', 'AU_NSW_IRD.csv', 'AU_VIC_IRD.csv']
    method_name = ['lstm', 'rnn', 'gru']
    for i in range(len(data_file)):
        for j in range(len(method_name)):
            train(data_file[i], method_name[j])






