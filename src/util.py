import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def kldiv_diag_gaussian(mean1, var1, mean2, var2):
    return .5*torch.sum((var1 + (mean1 - mean2)**2) / var2 + torch.log(var2) - torch.log(var1) - 1)

def reparam(mu, var, sample=True):
    if sample:
        #std = 0.5*logvar.exp_() # Why is there a .5?
        std = torch.sqrt(var)
        eps = torch.FloatTensor(std.size()).normal_()
        #eps = Variable(eps)
        return mu + eps * std
    else:
        return mu

def eig_stable(A, tol=1e-10, out=None): 
    #L, V = torch.eig(A, eigenvectors=True)
    #L=L[:,0].view(-1)
    
    L, V = torch.symeig(A, eigenvectors=True)
    keep = L > tol
    #print(out)
    #print("A",A)
    print(keep)
    return(L[keep], V[:,keep])

def sample_MVG2(M,U,V):
    Lu, Pu = eig_stable(U, out="U")
    Lv, Pv = eig_stable(V, out="V")
    
    E = torch.randn((Lu.shape[0],Lv.shape[0]))
    
    #print(torch.diag(torch.sqrt(Lv)))
    #print(Pv.transpose(0,1))

    return M + Pu@torch.diag(torch.sqrt(Lu))@E@torch.diag(torch.sqrt(Lv))@Pv.transpose(0,1)

def sample_MVG(M, U, v):
    # U is a matrix, v is the diagonal of V
    # M: rxc
    # U: rxr
    # V: cxc

    A = torch.cholesky(U + 1e-3*torch.diag(torch.ones(U.shape[0])))
    #A = torch.cholesky(U)
    B = torch.diag(v.sqrt())
    E = torch.randn(M.shape)
    
    return M + torch.mm(torch.mm(A, E), B)

def gen_toy_data(num_train=100, num_val=10, seed=0, x_true=None):
    """
    :param add_noise:
    :param num_tasks:
    :param seed:
    :param num_train:
    :param num_val:
    :param generate_data: if True, generate data. If false only generate task characteristics, i.e., phase and amplitude.
    :return:
    """
    np.random.seed(seed)
    period = 0.1 * np.random.randn() + 1#np.linspace(0.5, 5, np.sqrt(num_tasks))
    phase = np.pi
    amp = 1
    x = np.sort(np.random.uniform(-5, 5, num_train)).astype(np.float32)
    x_val = np.sort(np.random.uniform(-5, 5, num_val)).astype(np.float32)
    x = torch.from_numpy(x).view(-1, 1)
    x_val = torch.from_numpy(x_val).view(-1, 1)
    y_neg = Variable(amp * torch.sin(period * x[x<0] + phase)).view(-1,1) \
            + 0.1 * torch.randn(x[x<0].view(-1,1).size())
    y_pos = Variable(amp * torch.sin(4*period * x[x>=0] + phase)).view(-1,1) \
            + 0.1 * torch.randn(x[x>=0].view(-1,1).size())
    y = torch.cat((y_neg,y_pos))
    #y_val = amp * torch.sin(period * x_val + phase) + 0.1 * torch.randn(x_val.size())
    y_val_neg = amp * torch.sin(period * x_val[x_val<0] + phase).view(-1,1) \
            + 0.1 * torch.randn(x_val[x_val<0].view(-1,1).size())
    y_val_pos = amp * torch.sin(4*period * x_val[x_val>=0] + phase).view(-1,1) \
            + 0.1 * torch.randn(x_val[x_val>=0].view(-1,1).size())
    y_val = torch.cat((y_val_neg,y_val_pos))
    y_scale = torch.std(y)

    if x_true is not None:
        y_true_neg = amp * torch.sin(period * x_true[x_true<0] + phase).view(-1,1)
        y_true_pos = amp * torch.sin(4*period * x_true[x_true>=0] + phase).view(-1,1)
        y_true = torch.cat((y_true_neg,y_true_pos))
        return x, x_val, y, y_val, y_scale, y_true

    return x, x_val, y, y_val, y_scale


