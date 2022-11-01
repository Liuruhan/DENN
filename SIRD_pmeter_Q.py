import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=66)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=3000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true', default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true', default=False)
args = parser.parse_args()


# WUHAN: true_y0 = torch.tensor([[0.0502, 0.0032, 0.0038]]) #I,R,D
# ENGLAND: true_y0 = torch.tensor([[0.002, 0., 0.]]) # I, R, D

model='Italy'

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

if model=='wuhan':
    true_y0 = torch.tensor([[0.04518, 0.00502, 0.0032, 0.0038]]) # M, C,  R, D
    N = 1100.0572
    data = np.loadtxt('wuhan_IRD.csv', delimiter=',', skiprows=1)
elif model=='USA':
    true_y0 = torch.tensor([[0.00009, 0.00001, 0.00, 0.00] ]) # M, C,  R, D
    N = 32700.0001
    data = np.loadtxt('USA_IRD.csv', delimiter=',', skiprows=1)
elif model=='Italy':
    true_y0 = torch.tensor([[0.01998, 0.00222, 0.0002, 0.0006]]) # M, C,  R, D
    N = 6000.0230
    data = np.loadtxt('Italy_IRD.csv', delimiter=',', skiprows=1)
elif model=='UK':
    true_y0 = torch.tensor([[0.00018, 0.00002, 0.000, 0.000]])  # M, C,  R, D
    N = 6600.0002
    data = np.loadtxt('UK_IRD.csv', delimiter=',', skiprows=1)

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
print(t.size(), true_y.size())

all_t = torch.Tensor(data[:, 0])
all_truey = torch.Tensor(np.expand_dims(data[:, 1:4], axis=1))
#t = torch.linspace(0., 0.25, args.data_size)

class Lambda(nn.Module):

    def forward(self, t, y):
        #print(t, y)
        #Y = torch.zeros(y.size())
        Y = torch.zeros([1, 2])
        Y[:, 0] = (N-y[:, 0]-y[:,1]-y[0:,2]) * y[:, 0]
        Y[:, 1] = y[:, 0]
        Yp = torch.mm(Y, true_A)
        return Yp

#with torch.no_grad():
    #true_y = odeint(Lambda(), true_y0, t, method='adams')
    #print(true_y)
    #y_np = true_y.numpy()
    #fig = plt.figure(figsize=(6, 4))
    #plt.plot(range(y_np.shape[0]), y_np[:, 0, 0], color='b')
    #plt.plot(range(y_np.shape[0]), y_np[:, 0, 1], color='g')
    #plt.plot(range(y_np.shape[0]), y_np[:, 0, 2], color='r')
    #plt.plot(range(y_np.shape[0]), y_np[:, 0, 3], color='black')
    #plt.show()
    #np.savetxt('true_y.csv', true_y[:, 0, :], delimiter=',')
    #print(true_y)

def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)

    return batch_y0, batch_t, batch_y


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
    ax_beta = fig.add_subplot(245, frameon=False)
    ax_cita = fig.add_subplot(246, frameon=False)
    ax_ep = fig.add_subplot(247, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, param, q, odefunc, itr, t):

    if args.viz:

        ax_S.cla()
        ax_S.set_title('S')
        ax_S.set_xlabel('t')
        ax_S.set_ylabel('S(t)')
        t_N = torch.full(true_y[:, 0, 0].size(), N)
        true_S = t_N - true_y[:, 0, 0] - true_y[:, 0, 1] - true_y[:, 0, 2]
        t_N = torch.full(pred_y[:, 0, 0, 0].size(), N)
        pred_S = t_N - pred_y[:, 0, 0, 0] - pred_y[:, 0, 0, 1] - pred_y[:, 0, 0, 2] - pred_y[:, 0, 0, 3]
        ax_S.plot(t.numpy(), true_S, 'g-')
        ax_S.plot(t.numpy(), pred_S, 'r--')


        ax_I.cla()
        ax_I.set_title('I')
        ax_I.set_xlabel('t')
        ax_I.set_ylabel('I(t)')
        pred_I = pred_y[:, 0, 0, 0] + pred_y[:, 0, 0, 1]
        ax_I.plot(t.numpy(), true_y.numpy()[:, 0, 0], 'g-')
        ax_I.plot(t.numpy(), pred_I.numpy(), 'r--')
        #ax_phase.set_xlim(-2, 2)
        #ax_phase.set_ylim(-2, 2)

        ax_R.cla()
        ax_R.set_title('R')
        ax_R.set_xlabel('t')
        ax_R.set_ylabel('R(t)')
        ax_R.plot(t.numpy(), true_y.numpy()[:, 0, 1], 'g-')
        ax_R.plot(t.numpy(), pred_y.numpy()[:, 0, 0, 2], 'r--')

        ax_D.cla()
        ax_D.set_title('D')
        ax_D.set_xlabel('t')
        ax_D.set_ylabel('D(t)')
        ax_D.plot(t.numpy(), true_y.numpy()[:, 0, 2], 'g-')
        ax_D.plot(t.numpy(), pred_y.numpy()[:, 0, 0, 3], 'r--')

        ax_beta.cla()
        ax_beta.set_title('Beta')
        ax_beta.set_xlabel('t')
        ax_beta.set_ylabel('Beta(t)')
        ax_beta.plot(t.numpy(), q[:, 0, 0, 0], 'r--')

        ax_cita.cla()
        ax_cita.set_title('Cita')
        ax_cita.set_xlabel('t')
        ax_cita.set_ylabel('Cita(t)')
        ax_cita.plot(t.numpy(), torch.full([len(t.numpy())], param[1]).numpy(), 'r--')

        ax_ep.cla()
        ax_ep.set_title('Ep')
        ax_ep.set_xlabel('t')
        ax_ep.set_ylabel('Epp(t)')
        ax_ep.plot(t.numpy(), torch.full([len(t.numpy())], param[2]).numpy(), 'r--')

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        #plt.draw()
        #plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        # parameters
        #self.beta = Variable(torch.abs(torch.Tensor([0.02])), requires_grad=True)
        self.alpha = Variable(torch.abs(torch.Tensor([0.02])), requires_grad=True)
        self.cita1 = Variable(torch.abs(torch.Tensor([0.05])), requires_grad=True)
        self.cita2 = Variable(torch.abs(torch.Tensor([0.05])), requires_grad=True)
        self.ep = Variable(torch.abs(torch.Tensor([0.02])), requires_grad=True)

        self.beta_net = nn.Sequential(
            nn.Linear(4, 20),
            nn.ELU(),
            nn.Linear(20, 5),
            nn.ELU(),
            #nn.ReLU(),
            nn.Linear(5, 1),
            # nn.Tanh()
        )

        for m in self.beta_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0.05)

    def parameter_get(self, pred_y):
        return self.beta_net(pred_y)

    def forward(self, t, y):
        #print(y.size()
        beta = self.beta_net(y)

        Y = torch.zeros([1,1,4])
        Y[:,:,0] = beta*(N - y[:,:,0] - y[:,:,1] - y[:,:,2]-y[:,:,3]) * (y[:, :, 0]+y[:,:,1])/N - torch.abs(self.alpha)*y[:,:,0] - torch.abs(self.cita1)*y[:,:,0]
        Y[:,:,1] = torch.abs(self.alpha)*y[:,:,0] - torch.abs(self.cita2)*y[:,:,1] - torch.abs(self.ep)*y[:,:,1]
        Y[:,:,2] = torch.abs(self.cita1)*y[:,:,0] + torch.abs(self.cita2)*y[:,:,1]
        Y[:,:,3] = torch.abs(self.ep)*y[:,:,1]
        #print(t, y, self.net(Y))
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
    params = (list(func.parameters()) + list([func.alpha, func.cita1, func.cita2, func.ep]))
    optimizer = optim.RMSprop(params, lr=0.001)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        #print(itr)
        optimizer.zero_grad()
        pred_y = odeint(func, torch.unsqueeze(true_y0, dim=0), t, method='adams')
        loss_I = torch.mean(torch.abs((pred_y[:, :, :, 0] + pred_y[:, :, :, 1]) - torch.unsqueeze(true_y[:, :, 0], dim=1)))
        loss_R = torch.mean(torch.abs(pred_y[:, :, :, 2] - torch.unsqueeze(true_y[:, :, 1], dim=1)))
        loss_D = torch.mean(torch.abs(pred_y[:, :, :, 3] - torch.unsqueeze(true_y[:, :, 2], dim=1)))
        loss = loss_I + loss_R + loss_D
        #parameter = func.parameter_get(pred_y)
        #pmter = parameter.detach().numpy()
        #print(loss)
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, torch.unsqueeze(true_y0, dim=0), all_t, method='adams')
                Q = func.parameter_get(pred_y)
                q = Q.detach().numpy()
                loss_I = torch.mean(torch.abs((pred_y[:, :, :, 0] + pred_y[:, :, :, 1]) - torch.unsqueeze(all_truey[:, :, 0], dim=1)))
                loss_R = torch.mean(torch.abs(pred_y[:, :, :, 2] - torch.unsqueeze(all_truey[:, :, 1], dim=1)))
                loss_D = torch.mean(torch.abs(pred_y[:, :, :, 3] - torch.unsqueeze(all_truey[:, :, 2], dim=1)))
                loss = loss_I + loss_R +loss_D
                #parameter = func.parameter_get(pred_y)
                ##pmter = parameter.detach().numpy()
                parameter = [func.alpha.item(), func.cita1.item(), func.cita2.item(), func.ep.item()]
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                print(parameter)
                visualize(all_truey, pred_y, parameter, q, func, ii, all_t)
                ii += 1

        end = time.time()
