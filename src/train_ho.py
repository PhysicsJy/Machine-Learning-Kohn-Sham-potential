from tdksnn import TDKSNN
import torch
import argparse
import numpy as np
import os
import sys
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--nn_type', default='lstm',
                        type=str, help='type of the nn, lstm or fc')
    parser.add_argument('--pre_trained', default=False,
                        type=bool, help='whether use a pretrained model')
    parser.add_argument('--input_dim', default=2*150, type=int,
                        help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=400,
                        type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, #1e-3, 5e-4, 1e-4, 5e-5, 1e-5
                        type=np.double, help='learning rate')
    parser.add_argument('--batch_size', default=32, #32, 64, 128
                        type=int, help='batch_size')
    parser.add_argument('--nonlinearity', default='softplus',
                        type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=1000,
                        type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=50, type=int,
                        help='number of gradient steps between prints')
    parser.add_argument('--verbose', dest='verbose',
                        action='store_true', help='verbose?')
    parser.add_argument('--name', default='ho_sftplus_15_2', type=str,
                        help='only one option right now')
    parser.add_argument('--save_dir', default=THIS_DIR,
                        type=str, help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()


class KSOrbitalDataset(Dataset):
    def __init__(self, csv_file, csv_file2, root_dir, train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        coef_path = os.path.join(root_dir, csv_file)
        coef_dt_path = os.path.join(root_dir, csv_file2)
        self.orbital_coefficients = np.genfromtxt(coef_path, dtype=complex)
        self.orbital_coefficients_dt = np.genfromtxt(coef_dt_path, dtype=complex)
        self.train = train

        shuffler = np.random.permutation(len(self.orbital_coefficients))

        self.split_ix = int(len(self.orbital_coefficients) * 0.95)

        q = np.real(self.orbital_coefficients) * np.sqrt(2)
        p = np.imag(self.orbital_coefficients) * np.sqrt(2)
        dqdt = np.real(self.orbital_coefficients_dt) * np.sqrt(2)
        dpdt = np.imag(self.orbital_coefficients_dt) * np.sqrt(2)

        q = q[shuffler]
        p = p[shuffler]
        dqdt = dqdt[shuffler]
        dpdt = dpdt[shuffler]

        self.transform = transform
        x = torch.tensor(np.hstack((q, p)),
                         requires_grad=True, dtype=torch.float32)
        dx = torch.tensor(np.hstack((dqdt, dpdt)),
                          requires_grad=True, dtype=torch.float32)

        self.train_x = x[:self.split_ix]
        self.train_dx = dx[:self.split_ix]

        self.test_x = x[self.split_ix:]
        self.test_dx = dx[self.split_ix:]

    def __len__(self):
        if self.train:
            return len(self.train_x)
        return len(self.test_x)

    def __getitem__(self, idx):
        if self.train:
            x, dx = self.train_x[idx], self.train_dx[idx]
        else:
            x, dx = self.test_x[idx], self.test_dx[idx]

        sample = (x, dx)
        if self.transform:
            sample = self.transform((x, dx))
        return sample


def train(args):
    train_dataset = KSOrbitalDataset(csv_file='ho_wf_15_2.csv',
                                     csv_file2='ho_wf_dt_15_2.csv',
                                     root_dir='./data',
                                     train=True)

    test_dataset = KSOrbitalDataset(csv_file='ho_wf_15_2.csv',
                                    csv_file2='ho_wf_dt_15_2.csv',
                                    root_dir='./data',
                                    train=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = args.batch_size
    n_iters = args.total_steps
    num_epochs = n_iters / (len(train_dataset) / batch_size)
    num_epochs = int(num_epochs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    model = TDKSNN(args.input_dim, args.hidden_dim, output_dim=1,
                   nonlinearity=args.nonlinearity, nn_type='fc')
    model = model.to(device)

    if args.pre_trained:
        path = "{}/{}.tar".format(args.save_dir, args.name)
        model.load_state_dict(torch.load(path))

    optim = torch.optim.Adam(
        model.parameters(), args.learn_rate, weight_decay=1e-4)
    loss_func = torch.nn.MSELoss()
    loss_func = loss_func.to(device)
    stats = {'train_loss': [], 'test_loss': []}
    iter = 0

    for epoch in range(num_epochs):
        for (x, dxdt) in train_loader:
            x = x.to(device)
            dxdt = dxdt.to(device)
            dxdt_hat = model.time_derivative(x).to(device)
            loss = loss_func(dxdt, dxdt_hat) 
            loss.backward()
            grad = torch.cat([p.grad.flatten()
                              for p in model.parameters()]).clone()
            optim.step()
            optim.zero_grad()
            stats['train_loss'].append(loss.item())

            if iter % args.print_every == 0:
                for test_x, test_dxdt in test_loader:
                    test_x = test_x.to(device)
                    test_dxdt = test_dxdt.to(device)
                    test_dxdt_hat = model.time_derivative(test_x)
                    test_loss = loss_func(test_dxdt, test_dxdt_hat)
                    stats['test_loss'].append(test_loss.item())

                if args.verbose:
                    print("step {}, train_loss {:.4e}, test_loss {:.4e}, grad norm {:.4e}, grad std {:.4e}".format(
                        iter, loss.item(), test_loss.item(), grad@grad, grad.std()))
            iter += 1
    return model, stats


if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    path = '{}/{}.tar'.format(args.save_dir, args.name)
    torch.save(model.state_dict(), path)
