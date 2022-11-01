
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
parser.add_argument('--areas', type=str, choices=['wuhan', 'UK', 'USA', 'Italy'], default='UK')
parser.add_argument('--prob', type=int, default=0.8)
parser.add_argument('--data_size', type=int, default=53)
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
# WUHAN: true_y0 = torch.tensor([[0.0502, 0.0032, 0.0038]]) #I,R,D
# ENGLAND: true_y0 = torch.tensor([[0.002, 0., 0.]]) # I, R, D

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

if args.areas == 'Italy':
    true_y0 = torch.tensor([[0.444, 0.01998, 0.00222, 0.0002, 0.0006]]) # E, M, C, R, D
    N = 6000.4670
    data = np.loadtxt('Italy_IRD.csv', delimiter=',', skiprows=1)
elif args.areas == 'wuhan':
    true_y0 = torch.tensor([[1.004, 0.04518, 0.00502, 0.0032, 0.0038]])  # E, M, C, R, D
    N = 1101.0612
    data = np.loadtxt('wuhan_IRD.csv', delimiter=',', skiprows=1)
elif args.areas == 'UK':
    true_y0 = torch.tensor([[0.004, 0.00018, 0.00002, 0.0, 0.0]])  # E, M, C, R, D
    N = 6600.0042
    data = np.loadtxt('UK_IRD.csv', delimiter=',', skiprows=1)
elif args.areas == 'USA':
    true_y0 = torch.tensor([[0.002, 0.00009, 0.00001, 0.0, 0.0]])  # E, M, C, R, D
    N = 32700.0021
    data = np.loadtxt('USA_IRD.csv', delimiter=',', skiprows=1)

args.data_size = int(args.prob*data.shape[0])
tp = data[0:args.data_size, 0] # t
true_yp = data[0:args.data_size, 1:4] # I, R, D
te_t = data[args.data_size:, 0]
te_true_yp = data[args.data_size:, 1:4]
te_T = torch.Tensor(te_t)
te_y = torch.Tensor(np.expand_dims(te_true_yp, axis=1))
print('Test:', te_T.size(), te_y.size())
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
print('Train:', t.size(), true_y.size())

all_t = torch.Tensor(data[:, 0])
all_truey = torch.Tensor(np.expand_dims(data[:, 1:4], axis=1))
print('All:', all_t.size(), all_truey.size())

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 8), facecolor='white')
    ax_I = fig.add_subplot(251, frameon=False)
    ax_R = fig.add_subplot(252, frameon=False)
    ax_D = fig.add_subplot(253, frameon=False)
    ax_M = fig.add_subplot(254, frameon=False)
    ax_C = fig.add_subplot(255, frameon=False)
    ax_beta = fig.add_subplot(256, frameon=False)
    ax_cita1 = fig.add_subplot(257, frameon=False)
    ax_cita2 = fig.add_subplot(258, frameon=False)
    ax_ep = fig.add_subplot(259, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, param, Beta_q, Cita1_q, Cita2_q, Ep_q, odefunc, itr, t):
    scale = 10000
    if args.viz:

        ax_I.cla()
        ax_I.set_title('I')
        ax_I.set_xlabel('t')
        ax_I.set_ylabel('I(t)')
        pred_I = pred_y[:, 0, 0, 1] + pred_y[:, 0, 0, 2]
        ax_I.plot(t.numpy(), true_y.numpy()[:, 0, 0]*scale, 'g-')
        ax_I.plot(t.numpy(), pred_I.numpy()*scale, 'b--')

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

        ax_beta.cla()
        ax_beta.set_title('Beta')
        ax_beta.set_xlabel('t')
        ax_beta.set_ylabel('Beta(t)')
        ax_beta.plot(t.numpy(), Beta_q[:, 0, 0, 0], 'r--')
        ax_beta.plot(t.numpy(), torch.full([len(t.numpy())], Beta_q[0, 0, 0, 0]).numpy(), 'darkred', alpha=0.6)

        ax_cita1.cla()
        ax_cita1.set_title('Cita1')
        ax_cita1.set_xlabel('t')
        ax_cita1.set_ylabel('Cita1(t)')
        ax_cita1.plot(t.numpy(), Cita1_q[:, 0, 0, 0], 'r--')
        ax_cita1.plot(t.numpy(), torch.full([len(t.numpy())], Cita1_q[0, 0, 0, 0]).numpy(), 'darkred', alpha=0.6)

        ax_cita2.cla()
        ax_cita2.set_title('Cita2')
        ax_cita2.set_xlabel('t')
        ax_cita2.set_ylabel('Cita2(t)')
        ax_cita2.plot(t.numpy(), Cita2_q[:, 0, 0, 0], 'r--')
        ax_cita2.plot(t.numpy(), torch.full([len(t.numpy())], Cita2_q[0, 0, 0, 0]).numpy(), 'darkred', alpha=0.6)

        ax_ep.cla()
        ax_ep.set_title('Ep')
        ax_ep.set_xlabel('t')
        ax_ep.set_ylabel('Ep(t)')
        ax_ep.plot(t.numpy(), Ep_q[:, 0, 0, 0], 'r--')
        ax_ep.plot(t.numpy(), torch.full([len(t.numpy())], Ep_q[0, 0, 0, 0]).numpy(), 'darkred', alpha=0.6)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        
        np.savetxt('txt/SEMCRD_NN' + args.areas + str(itr) + '.csv', pred_y.numpy()[:, 0, 0, :], delimiter=',')

        print('initial_beta:', Beta_q[0, 0, 0, 0].item())
        print('initial_cita1:', Cita1_q[0, 0, 0, 0].item())
        print('initial_cita2:', Cita2_q[0, 0, 0, 0].item())
        print('initial_ep:', Ep_q[0, 0, 0, 0].item())
        print('AveR2_I:', r2_score(pred_I.numpy()*scale, true_y.numpy()[:, 0, 0]*scale))
        print('AveR2_R:', r2_score(pred_y.numpy()[:, 0, 0, 3]*scale, true_y.numpy()[:, 0, 1]*scale))
        print('AveR2_D:', r2_score(pred_y.numpy()[:, 0, 0, 4]*scale, true_y.numpy()[:, 0, 2]*scale))
        print('AveMSE_I:', mean_squared_error(pred_I.numpy()*scale, true_y.numpy()[:, 0, 0]*scale))
        print('AveMSE_R:', mean_squared_error(pred_y.numpy()[:, 0, 0, 3]*scale, true_y.numpy()[:, 0, 1]*scale))
        print('AveMSE_D:', mean_squared_error(pred_y.numpy()[:, 0, 0, 4]*scale, true_y.numpy()[:, 0, 2]*scale))
        print('ValR2_I:', r2_score(pred_I.numpy()[args.data_size:] * scale, true_y.numpy()[args.data_size:, 0, 0] * scale))
        print('ValR2_R:', r2_score(pred_y.numpy()[args.data_size:, 0, 0, 3] * scale, true_y.numpy()[args.data_size:, 0, 1] * scale))
        print('ValR2_D:', r2_score(pred_y.numpy()[args.data_size:, 0, 0, 4] * scale, true_y.numpy()[args.data_size:, 0, 2] * scale))
        print('ValMSE_I:', mean_squared_error(pred_I.numpy()[args.data_size:] * scale, true_y.numpy()[args.data_size:, 0, 0] * scale))
        print('ValMSE_R:', mean_squared_error(pred_y.numpy()[args.data_size:, 0, 0, 3] * scale, true_y.numpy()[args.data_size:, 0, 1] * scale))
        print('ValMSE_D:', mean_squared_error(pred_y.numpy()[args.data_size:, 0, 0, 4] * scale, true_y.numpy()[args.data_size:, 0, 2] * scale))

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        # parameters
        self.gamma = Variable(torch.abs(torch.Tensor([0.15])), requires_grad=True)
        self.alpha = Variable(torch.abs(torch.Tensor([0.15])), requires_grad=True)


        self.beta_net = nn.Sequential(
            nn.Linear(5, 20),
            nn.ELU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )

        self.cita1_net = nn.Sequential(
            nn.Linear(5, 20),
            nn.ELU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )

        self.cita2_net = nn.Sequential(
            nn.Linear(5, 20),
            nn.ELU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )

        self.ep_net = nn.Sequential(
            nn.Linear(5, 20),
            nn.ELU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )

        for m in self.beta_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.002)
                nn.init.constant_(m.bias, val=0.01)

        for m in self.cita1_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=-1.02)

        for m in self.cita2_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.002)
                nn.init.constant_(m.bias, val=-1.56)

        for m in self.ep_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.002)
                nn.init.constant_(m.bias, val=-1.56)

    def parameter_get(self, pred_y):
        return self.beta_net(pred_y), self.cita1_net(pred_y)**2, self.cita2_net(pred_y)**2, self.ep_net(pred_y)**2

    def forward(self, t, y):
        beta = self.beta_net(y)
        cita1 = self.cita1_net(y)**2
        cita2 = self.cita2_net(y)**2
        ep = self.ep_net(y)**2

        Y = torch.zeros([1,1,5])
        Y[:,:,0] = beta*(N - y[:,:,0] - y[:,:,1] - y[:,:,2]-y[:,:,3]-y[:,:,4]) * (y[:, :, 1]+y[:,:,2])/N - torch.abs(self.gamma)*y[:,:,0]
        Y[:,:,1] = torch.abs(self.gamma)*y[:,:,0] - torch.abs(self.alpha)*y[:,:,1] - cita1*y[:,:,1]
        Y[:,:,2] = torch.abs(self.alpha)*y[:,:,1] - cita2*y[:,:,2] - ep*y[:,:,2]
        Y[:,:,3] = cita1*y[:,:,1] + cita2*y[:,:,2]
        Y[:,:,4] = ep*y[:,:,2]
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
    params = (list(func.parameters()) + list([func.gamma, func.alpha]))
    optimizer = optim.RMSprop(params, lr=args.lr)
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
            if itr % (25  * args.test_freq) == 0:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.95
            with torch.no_grad():
                print('Iter {:04d} | Loss {:.6f}'.format(itr, loss.item()))
                pred_y = odeint(func, torch.unsqueeze(true_y0, dim=0), all_t, method=args.method)
                Beta, Cita1, Cita2, Ep = func.parameter_get(pred_y)
                Beta_q = Beta.detach().numpy()
                Cita1_q = Cita1.detach().numpy()
                Cita2_q = Cita2.detach().numpy()
                Ep_q = Ep.detach().numpy()
                loss_I = torch.mean(torch.abs((pred_y[:, :, :, 1]+ pred_y[:,:,:,2]) - torch.unsqueeze(all_truey[:, :, 0], dim=1)))
                loss_R = torch.mean(torch.abs(pred_y[:, :, :, 3] - torch.unsqueeze(all_truey[:, :, 1], dim=1)))
                loss_D = torch.mean(torch.abs(pred_y[:, :, :, 4] - torch.unsqueeze(all_truey[:, :, 2], dim=1)))
                loss = loss_I + loss_R + loss_D
                parameter = [func.gamma.item(), func.alpha.item()]
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                pmter_0 = [np.max(list(func.parameters())[0].detach().numpy()),
                           np.min(list(func.parameters())[0].detach().numpy())]
                pmter_1 = [np.max(list(func.parameters())[1].detach().numpy()),
                           np.min(list(func.parameters())[1].detach().numpy())]
                pmter_2 = [np.max(list(func.parameters())[2].detach().numpy()),
                           np.min(list(func.parameters())[2].detach().numpy())]
                p_min = pmter_0[1] if pmter_0[1] < pmter_1[1] else pmter_1[1]
                p_min = p_min if p_min < pmter_2[1] else pmter_2[1]
                p_max = pmter_0[0] if pmter_0[0] > pmter_1[0] else pmter_1[0]
                p_max = p_max if p_max > pmter_2[0] else pmter_2[0]
                print('p(max):', p_max, 'p(min):', p_min)
                print(parameter)
                visualize(all_truey, pred_y, parameter, Beta_q, Cita1_q, Cita2_q, Ep_q, func, ii, all_t)
                ii += 1
            torch.save(func, 'func.pkl')
            print('Save pth!')
        end = time.time()
