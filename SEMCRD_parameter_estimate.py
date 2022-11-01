import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--areas', type=str, default='wuhan')
parser.add_argument('--prob', type=float, default=20)
parser.add_argument('--data_size', type=int, default=68)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=5000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true', default=True)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true', default=False)
args = parser.parse_args()

print("Method:", args.method)
print("Probability:", args.prob)
print("Areas:", args.areas)
print("Iteration:", args.niters)
print("LearningRate:", args.lr)
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
print("GPU:", device)
# WUHAN: true_y0 = torch.tensor([[0.0502, 0.0032, 0.0038]]) #I,R,D
# ENGLAND: true_y0 = torch.tensor([[0.002, 0., 0.]]) # I, R, D

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

#cities & areas
if args.areas == 'wuhan':
    true_y0 = torch.tensor([[1.004, 0.04518, 0.00502, 0.0032, 0.0038]]) # E, M, C, R, D
    N = 1101.0612
    data = np.loadtxt('wuhan_IRD.csv', delimiter=',', skiprows=1)
elif args.areas == 'Piedmont':
    true_y0 = torch.tensor([[0.006, 0.00027, 0.00003, 0.00, 0.00]]) # E, M, C, R, D
    N = 430.0009
    data = np.loadtxt('Piedmont_IRD.csv', delimiter=',', skiprows=1)
elif args.areas == 'AU_NSW':
    true_y0 = torch.tensor([[0.532, 0.02394, 0.00266, 0.00, 0.00]]) # E, M, C, R, D
    N = 749.5586
    data = np.loadtxt('AU_NSW_IRD.csv', delimiter=',', skiprows=1)
elif args.areas == 'AU_VIC':
    true_y0 = torch.tensor([[0.242, 0.01089, 0.00121, 0.00, 0.00]]) # E, M, C, R, D
    N = 593.2541
    data = np.loadtxt('AU_VIC_IRD.csv', delimiter=',', skiprows=1)
#countries
elif args.areas == 'USA':
    true_y0 = torch.tensor([[0.002, 0.00009, 0.00001, 0.00, 0.00]]) # E, M, C, R, D
    N = 32700.0021
    data = np.loadtxt('USA_IRD.csv', delimiter=',', skiprows=1)
elif args.areas == 'Italy':
    true_y0 = torch.tensor([[0.444, 0.01998, 0.00222, 0.0002, 0.0006]]) # E, I, R, D
    N = 6000.467
    data = np.loadtxt('Italy_IRD.csv', delimiter=',', skiprows=1)
elif args.areas == 'Columbia':
    true_y0 = torch.tensor([[0.002, 0.00009, 0.00001, 0.00, 0.00]]) # M, C, R, D
    N = 4965.0021
    data = np.loadtxt('Columbia_IRD.csv', delimiter=',', skiprows=1)
elif args.areas == 'South_Africa':
    true_y0 = torch.tensor([[0.004, 0.00018, 0.00002, 0.00, 0.00]]) # I, R, D
    N = 5652.0042
    data = np.loadtxt('South_africa_IRD.csv', delimiter=',', skiprows=1)
args.data_size = data.shape[0] - args.prob
tp = data[0:args.data_size, 0] # t
true_yp = data[0:args.data_size, 1:4] # I, R, D
t = np.linspace(0, len(tp)-1, args.data_size)
true_y = np.full((args.data_size, true_yp.shape[1]), 1.0)

true_y[:, 0] = np.interp(t, tp, true_yp[:, 0])
true_y[:, 1] = np.interp(t, tp, true_yp[:, 1])
true_y[:, 2] = np.interp(t, tp, true_yp[:, 2])
fig = plt.figure(figsize=(6, 4))
plt.plot(range(true_y.shape[0]), true_y[:, 0], color='b')
plt.plot(range(true_y.shape[0]), true_y[:, 1], color='g')
plt.plot(range(true_y.shape[0]), true_y[:, 2], color='r')
plt.show()
t = torch.Tensor(t)
true_y = torch.Tensor(np.expand_dims(true_y, axis=1))
print("Train:", t.size(), true_y.size())

all_t = torch.Tensor(data[:, 0])
all_truey = torch.Tensor(np.expand_dims(data[:, 1:4], axis=1))
print("All:", all_t.size(), all_truey.size())

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 8), facecolor='white')
    ax_S = fig.add_subplot(241, frameon=False)
    ax_I = fig.add_subplot(242, frameon=False)
    ax_R = fig.add_subplot(243, frameon=False)
    ax_D = fig.add_subplot(244, frameon=False)
    ax_M = fig.add_subplot(245, frameon=False)
    ax_C = fig.add_subplot(246, frameon=False)
    ax_ep = fig.add_subplot(247, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, param, odefunc, itr, t):
    scale = 10000
    if args.viz:

        ax_S.cla()
        ax_S.set_title('S')
        ax_S.set_xlabel('t')
        ax_S.set_ylabel('S(t)')
        t_N = torch.full(true_y[:, 0, 0].size(), N)
        true_S = t_N - true_y[:, 0, 0] - true_y[:, 0, 1] - true_y[:, 0, 2]
        t_N = torch.full(pred_y[:, 0, 0, 0].size(), N)
        pred_S = t_N - pred_y[:, 0, 0, 0] - pred_y[:, 0, 0, 1] - pred_y[:, 0, 0, 2]
        ax_S.plot(t.numpy(), true_S, 'g-')
        ax_S.plot(t.numpy(), pred_S, 'b--')


        ax_I.cla()
        ax_I.set_title('I')
        ax_I.set_xlabel('t')
        ax_I.set_ylabel('I(t)')
        pred_I = pred_y[:, 0, 0, 1] + pred_y[:, 0, 0, 2]
        ax_I.plot(t.numpy(), true_y.numpy()[:, 0, 0]*scale, 'g-')
        ax_I.plot(t.numpy(), pred_I.numpy()*scale, 'b--')
        #ax_phase.set_xlim(-2, 2)
        #ax_phase.set_ylim(-2, 2)

        ax_R.cla()
        ax_R.set_title('R')
        ax_R.set_xlabel('t')
        ax_R.set_ylabel('R(t)')
        ax_R.plot(t.numpy(), true_y.numpy()[:, 0, 1]*scale, 'g-')
        ax_R.plot(t.numpy(), pred_y.numpy()[:, 0, 0, 3]*scale, 'b--')

        ax_D.cla()
        ax_D.set_title('D')
        ax_D.set_xlabel('t')
        ax_D.set_ylabel('D(t)')
        ax_D.plot(t.numpy(), true_y.numpy()[:, 0, 2]*scale, 'g-')
        ax_D.plot(t.numpy(), pred_y.numpy()[:, 0, 0, 4]*scale, 'b--')

        ax_M.cla()
        ax_M.set_title('M')
        ax_M.set_xlabel('t')
        ax_M.set_ylabel('M(t)')
        ax_M.plot(t.numpy(), pred_y.numpy()[:, 0, 0, 1], 'b--')

        ax_C.cla()
        ax_C.set_title('C')
        ax_C.set_xlabel('t')
        ax_C.set_ylabel('C(t)')
        ax_C.plot(t.numpy(), pred_y.numpy()[:, 0, 0, 2], 'b--')

        ax_ep.cla()
        ax_ep.set_title('Ep')
        ax_ep.set_xlabel('t')
        ax_ep.set_ylabel('Epp(t)')
        ax_ep.plot(t.numpy(), torch.full([len(t.numpy())], param[2]).numpy(), 'r--')

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
      
        np.savetxt('txt/SEMCRD_pmter' + args.areas + str(itr) + '.csv', pred_y.numpy()[:, 0, 0, :], delimiter=',')
        
        print('I-R2:', r2_score(true_y.numpy()[:, 0, 0]*scale, pred_I.numpy()*scale))
        print('R-R2:', r2_score(true_y.numpy()[:, 0, 1]*scale, pred_y.numpy()[:, 0, 0, 3]*scale))
        print('D-R2:', r2_score(true_y.numpy()[:, 0, 2]*scale, pred_y.numpy()[:, 0, 0, 4]*scale))
        print('I-MSE:', mean_squared_error(true_y.numpy()[:, 0, 0]*scale, pred_I.numpy()*scale))
        print('R-MSE:', mean_squared_error(true_y.numpy()[:, 0, 1]*scale, pred_y.numpy()[:, 0, 0, 3]*scale))
        print('D-MSE:', mean_squared_error(true_y.numpy()[:, 0, 2]*scale, pred_y.numpy()[:, 0, 0, 4]*scale))


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        # parameters
        self.beta = Variable(torch.abs(torch.Tensor([0.5])), requires_grad=True)
        self.gamma = Variable(torch.abs(torch.Tensor([0.15])), requires_grad=True)
        self.alpha = Variable(torch.abs(torch.Tensor([0.15])), requires_grad=True)
        self.cita1 = Variable(torch.abs(torch.Tensor([0.07])), requires_grad=True)
        self.cita2 = Variable(torch.abs(torch.Tensor([0.03])), requires_grad=True)
        self.ep = Variable(torch.abs(torch.Tensor([0.03])), requires_grad=True)

    def parameter_get(self, y):
        return self.net(y)

    def forward(self, t, y):

        Y = torch.zeros([1,1,5])
        Y[:,:,0] = torch.abs(self.beta)*(N - y[:,:,0] - y[:,:,1] - y[:,:,2]-y[:,:,3]-y[:,:,4]) * (y[:, :, 1]+y[:,:,2])/N - torch.abs(self.gamma)*y[:,:,0]
        Y[:,:,1] = torch.abs(self.gamma)*y[:,:,0] - torch.abs(self.alpha)*y[:,:,1] - torch.abs(self.cita1)*y[:,:,1]
        Y[:,:,2] = torch.abs(self.alpha)*y[:,:,1] - torch.abs(self.cita2)*y[:,:,2] - torch.abs(self.ep)*y[:,:,2]
        Y[:,:,3] = torch.abs(self.cita1)*y[:,:,1] + torch.abs(self.cita2)*y[:,:,2]
        Y[:,:,4] = torch.abs(self.ep)*y[:,:,2]
        return Y


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0

    func = ODEFunc()
    optimizer = optim.RMSprop([func.beta, func.gamma, func.alpha, func.cita1, func.cita2, func.ep], lr=args.lr)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        #print(itr)
        optimizer.zero_grad()
        pred_y = odeint(func, torch.unsqueeze(true_y0, dim=0), t, method=args.method)
        loss_I = torch.mean(torch.abs((pred_y[:, :, :, 1]+ pred_y[:,:,:,2]) - torch.unsqueeze(true_y[:, :, 0], dim=1)))
        loss_R = torch.mean(torch.abs(pred_y[:, :, :, 3] - torch.unsqueeze(true_y[:, :, 1], dim=1)))
        loss_D = torch.mean(torch.abs(pred_y[:, :, :, 4] - torch.unsqueeze(true_y[:, :, 2], dim=1)))
        loss = loss_I + loss_R + loss_D
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                print('Iter {:04d} | Loss {:.6f}'.format(itr, loss.item()))
                pred_y = odeint(func, torch.unsqueeze(true_y0, dim=0), all_t, method=args.method)
                loss_I = torch.mean(torch.abs((pred_y[:, :, :, 1]+ pred_y[:,:,:,2]) - torch.unsqueeze(all_truey[:, :, 0], dim=1)))
                loss_R = torch.mean(torch.abs(pred_y[:, :, :, 3] - torch.unsqueeze(all_truey[:, :, 1], dim=1)))
                loss_D = torch.mean(torch.abs(pred_y[:, :, :, 4] - torch.unsqueeze(all_truey[:, :, 2], dim=1)))
                loss = loss_I + loss_R +loss_D
                param = [func.beta.item(), func.gamma.item(), func.alpha.item(), func.cita1.item(), func.cita2.item(), func.ep.item()]
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                print(param)
                visualize(all_truey, pred_y, param, func, ii, all_t)
                ii += 1

        end = time.time()
