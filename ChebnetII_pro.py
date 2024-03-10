import math
import torch

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian
import torch.nn.functional as F
from utils import cheby
from torch.nn import Parameter

def presum_tensor(h, initial_val):
    length = len(h) + 1
    temp = torch.zeros(length)
    temp[0] = initial_val
    for idx in range(1, length):
        temp[idx] = temp[idx-1] + h[idx-1]
    return temp

def preminus_tensor(h, initial_val):
    length = len(h) + 1
    temp = torch.zeros(length)
    temp[0] = initial_val
    for idx in range(1, length):
        temp[idx] = temp[idx-1] - h[idx-1]
    return temp

def reverse_tensor(h):
    temp = torch.zeros_like(h)
    length = len(temp)
    for idx in range(0, length):
        temp[idx] = h[length-1-idx]
    return temp


class ChebnetII_prop(MessagePassing):
    def __init__(self, K, **kwargs):
        super(ChebnetII_prop, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.initial_val_low = Parameter(torch.tensor(2.0), requires_grad=False)
        self.temp_low = Parameter(torch.Tensor(self.K), requires_grad=True)
        self.temp_high = Parameter(torch.Tensor(self.K), requires_grad=True)
        self.initial_val_high = Parameter(torch.tensor(0.0), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.temp_low.data.fill_(2.0/self.K)
        self.temp_high.data.fill_(2.0/self.K)


    def forward(self, x, edge_index, edge_weight=None, highpass=True):
        if highpass:
            TEMP = F.relu(self.temp_high)
            coe_tmp = presum_tensor(TEMP, self.initial_val_high)
        else:
            TEMP = F.relu(self.temp_low)
            coe_tmp = preminus_tensor(TEMP, self.initial_val_low)

        coe = coe_tmp.clone()

        for i in range(self.K + 1):
            coe[i] = coe_tmp[0] * cheby(i, math.cos((self.K + 0.5) * math.pi / (self.K + 1)))
            for j in range(1, self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                coe[i] = coe[i] + coe_tmp[j] * cheby(i, x_j)
            coe[i] = 2 * coe[i] / (self.K + 1)

        # L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                           num_nodes=x.size(self.node_dim))

        # L_tilde=L-I
        edge_index_tilde, norm_tilde = add_self_loops(edge_index1, norm1, fill_value=-1.0,
                                                      num_nodes=x.size(self.node_dim))

        Tx_0 = x
        Tx_1 = self.propagate(edge_index_tilde, x=x, norm=norm_tilde, size=None)

        out = coe[0] / 2 * Tx_0 + coe[1] * Tx_1

        for i in range(2, self.K + 1):
            Tx_2 = self.propagate(edge_index_tilde, x=Tx_1, norm=norm_tilde, size=None)
            Tx_2 = 2 * Tx_2 - Tx_0
            out = out + coe[i] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)