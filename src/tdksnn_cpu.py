import torch
import numpy as np
from torch.nn.modules.activation import LeakyReLU
import utils


class TDKSNN(torch.nn.Module):
    '''Learn the KS-potential and its gradient'''

    def __init__(self, input_dim, hidden_dim, output_dim=1, nonlinearity='tanh', nn_type='lstm'):
        super(TDKSNN, self).__init__()
        device = torch.device("cpu")
        self.M = self.symplectic_matrix(input_dim).to(device)

        self.layer_dim = 1
        self.nn_type = nn_type
        self.hidden_dim = hidden_dim
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)
        # self.layer4 = torch.nn.Sequential(
        #     torch.nn.Linear(hidden_dim, hidden_dim),
        #     torch.nn.LeakyReLU(),
        # )
        for l in [self.layer1, self.layer2, self.layer3]:
            torch.nn.init.orthogonal_(l.weight)
        

        self.nonlinearity = utils.choose_nonlinearity(nonlinearity)
        # self.dropout = torch.nn.Dropout(1e-20)
        # self.rnn = torch.nn.RNN(
        #     round(input_dim / 2), hidden_dim, self.layer_dim, batch_first=True, nonlinearity=nonlinearity)
        # self.lstm = torch.nn.LSTM(
        #     round(input_dim / 2), hidden_dim, self.layer_dim, batch_first=True)

    def forward(self, x):
        # if self.nn_type == 'lstm':
        #     y = x[:, 0:101]**2 + x[:, 101:]**2
        #     y = y.unsqueeze(0)
        #     # y = self.dropout(y)
        #     h0 = torch.zeros(self.layer_dim, y.size(
        #         0), self.hidden_dim).requires_grad_()
        #     c0 = torch.zeros(self.layer_dim, y.size(
        #         0), self.hidden_dim).requires_grad_()
        #     out, (hn, cn) = self.lstm(y, (h0.detach(), c0.detach()))
        #     # print(out.size())
        #     out = out[-1, :, :]
        #     out = self.dropout(out)
        #     out = self.layer3(out)
            # print(out.size())
        # elif self.nn_type == 'fc':
        # y = x[:, 0:101].pow(2) + x[:, 101:].pow(2)
            # y = self.dropout(y)
        y = self.nonlinearity(self.layer1(x))
        # y = self.dropout(y)
        y = self.nonlinearity(self.layer2(y))
        # y = self.layer4(y)

        out = self.layer3(y)
            # out = self.dropout(out)
        return out
        # print(x.shape)

        # y, hn = self.rnn(x, h0.detach())

        # y = self.nonlinearity(self.layer2(y))

        #assert y.dim() == 2 and y.shape[1] == 2, "Output tensor should have shape [batch_size, 2]"
        # print("y shape={}".format(y.shape))

    def symplectic_matrix(self, n):
        M = torch.eye(n, dtype=torch.double)
        M = torch.cat([M[n//2:], -M[:n//2]])
        return M

    def time_derivative(self, x):
        '''NN Hamiltonian'''
        H = self.forward(x)
        # print(H.shape)
        '''Gradient of the Hamiltonian'''
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        dH_sympletic = dH @ self.M.t()
        return dH_sympletic

    def hamiltonian_time_derivative(self, x):
        H = self.forward(x)
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        ddH  = dH * x
        return dH
