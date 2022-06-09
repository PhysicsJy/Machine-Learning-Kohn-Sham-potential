import torch

def choose_nonlinearity(name):
  nl = None
  if name == 'tanh':
    nl = torch.tanh
  elif name == 'relu':
    nl = torch.relu
  elif name == 'sigmoid':
    nl = torch.sigmoid
  elif name == 'softplus':
    nl = torch.nn.functional.softplus
  elif name == 'selu':
    nl = torch.nn.functional.selu
  elif name == 'elu':
    nl = torch.nn.functional.elu
  elif name == 'swish':
    nl = lambda x: x * torch.sigmoid(x)
  else:
    raise ValueError("nonlinearity not recognized")
  return nl

class TDKSNN(torch.nn.Module):
    '''Learn the KS-potential and its gradient'''
    def __init__(self, input_dim, hidden_dim, output_dim=1, nonlinearity='tanh', nn_type=None):
        super(TDKSNN, self).__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.symp_mat = self.symplectic_matrix(input_dim).to(device)
        self.layer_dim = 1
        self.nn_type = nn_type
        self.hidden_dim = hidden_dim
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

        for l in [self.layer1, self.layer2, self.layer3]:
            torch.nn.init.orthogonal_(l.weight)
        
        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, x):
        y = self.nonlinearity(self.layer1(x))
        y = self.nonlinearity(self.layer2(y))
        out = self.layer3(y)

        return out

    def symplectic_matrix(self, n):
        symp_mat = torch.eye(n, dtype=torch.float32)
        symp_mat = torch.cat([symp_mat[n//2:], -symp_mat[:n//2]])
        return symp_mat

    def time_derivative(self, x):
        H = self.forward(x)
        '''Gradient of the Hamiltonian'''
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        dH_sympletic = torch.matmul(dH, self.M.t())
        return dH_sympletic
