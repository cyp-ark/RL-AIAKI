import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Sampler

def make_transition(data,col_reward,target,rolling_size):
    df = pd.read_parquet(data)
    s_col = [x for x in df if x[:2]=='s:']
    a_col = [x for x in df if x[:2]=='a:']
    r_col = [x for x in df if x==col_reward]
    dict = {}
    dict['traj'] = {}

    s,a,r,s2,t  = [],[],[],[],[]
    
    for traj in tqdm(df.traj.unique()):
        df_traj = df[df['traj'] == traj]
        dict['traj'][traj] = {'s':[],'a':[],'r':[]}
        dict['traj'][traj]['s'] = df_traj[s_col].values.tolist()
        dict['traj'][traj]['a'] = df_traj[a_col].values.tolist()
        dict['traj'][traj]['r'] = df_traj[r_col].values.tolist()

        step_len = len(df_traj) - rolling_size
        for step in range(step_len):
            current_state = dict['traj'][traj]['s'][step:step+rolling_size]
            current_state[-1][-12:]=[0,0,0,0,0,0,0,0,0,0,0,0]
            s.append(current_state)
            a.append(dict['traj'][traj]['a'][step+rolling_size-1])
            reward = dict['traj'][traj]['r'][step+rolling_size]
            r.append(reward)
            next_state = dict['traj'][traj]['s'][step+1:step+1+rolling_size]
            next_state[-1][-12:]=[0,0,0,0,0,0,0,0,0,0,0,0]
            s2.append(next_state)

            if (reward == target) or (step == step_len - 1):
                t.append(1)
                break
            else : 
                t.append(0)
    
    s  = torch.FloatTensor(np.float32(s))
    a  = torch.LongTensor(np.int64(a))
    r = torch.FloatTensor(np.float32(r))
    s2 = torch.FloatTensor(np.float32(s2))
    t  = torch.FloatTensor(np.float32(t))
    Dataset = TensorDataset(s, a, r, s2, t)

    return Dataset

class CustomSampler(Sampler):
    def __init__(self, data, batch_size, ns, target):
        self.data = data
        self.batch_size = batch_size
        self.num_samples_1 = ns
        self.num_samples_0 = batch_size - ns
        self.indices = [i for i in range(len(data)) if data[i][2].item() == 0]
        self.indices_neg = [i for i in range(len(data)) if data[i][2].item() == target]
        self.used_indices_neg = []

    def __iter__(self):
        np.random.shuffle(self.indices)
        np.random.shuffle(self.indices_neg)

        batch = []
        for idx in self.indices:
            batch.append(idx)
            if len(batch) == self.num_samples_0:
                batch.extend(self._sample_indices_neg())
                yield batch
                batch = []

    def _sample_indices_neg(self, remaining=0):
        if remaining:
            num_samples_1 = self.batch_size - remaining
        else:
            num_samples_1 = self.num_samples_1

        if len(self.used_indices_neg) + num_samples_1 > len(self.indices_neg):
            self.used_indices_neg = []
            np.random.shuffle(self.indices_neg)

        indices_neg = self.indices_neg[len(self.used_indices_neg):len(self.used_indices_neg) + num_samples_1]
        self.used_indices_neg.extend(indices_neg)
        return indices_neg

    def __len__(self):
        return (len(self.indices) + len(self.indices_neg)) // self.batch_size

class imvt(torch.jit.ScriptModule):
    __constants__ = ['input_dim', 'n_units']
    def __init__(self, input_dim, output_dim, n_units, device, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.zeros(input_dim, 1, n_units)*init_std)
        self.U_i = nn.Parameter(torch.zeros(input_dim, 1, n_units)*init_std)
        self.U_f = nn.Parameter(torch.zeros(input_dim, 1, n_units)*init_std)
        self.U_o = nn.Parameter(torch.zeros(input_dim, 1, n_units)*init_std)
        self.W_j = nn.Parameter(torch.zeros(input_dim, n_units, n_units)*init_std)
        self.W_i = nn.Parameter(torch.zeros(input_dim, n_units, n_units)*init_std)
        self.W_f = nn.Parameter(torch.zeros(input_dim, n_units, n_units)*init_std)
        self.W_o = nn.Parameter(torch.zeros(input_dim, n_units, n_units)*init_std)
        self.b_j = nn.Parameter(torch.zeros(input_dim, n_units)*init_std)
        self.b_i = nn.Parameter(torch.zeros(input_dim, n_units)*init_std)
        self.b_f = nn.Parameter(torch.zeros(input_dim, n_units)*init_std)
        self.b_o = nn.Parameter(torch.zeros(input_dim, n_units)*init_std)
        self.F_alpha_n = nn.Parameter(torch.zeros(input_dim, n_units, 1)*init_std)
        self.F_alpha_n_b = nn.Parameter(torch.zeros(input_dim, 1)*init_std)
        self.F_beta = nn.Linear(2*n_units, 1)
        self.Phi = nn.Linear(2*n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim
        self.device = device
    
    @torch.jit.script_method
    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(self.device)
        c_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(self.device)
        outputs = torch.jit.annotate(List[Tensor], [])
        for t in range(x.shape[1]):
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_j) + self.b_j)
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_o) + self.b_o)
            c_tilda_t = c_tilda_t*f_tilda_t + i_tilda_t*j_tilda_t
            h_tilda_t = (o_tilda_t*torch.tanh(c_tilda_t))
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n)+self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas/torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas*outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas/torch.sum(betas, dim=1, keepdim=True)
        mean = torch.sum(betas*mu, dim=1)

        return mean, alphas, betas

class imvt2(torch.jit.ScriptModule):
    __constants__ = ['input_dim', 'n_units']
    def __init__(self, input_dim, output_dim, n_units, device, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.zeros(input_dim, 1, n_units))
        self.U_i = nn.Parameter(torch.zeros(input_dim, 1, n_units))
        self.U_f = nn.Parameter(torch.zeros(input_dim, 1, n_units))
        self.U_o = nn.Parameter(torch.zeros(input_dim, 1, n_units))
        self.W_j = nn.Parameter(torch.zeros(input_dim, n_units, n_units))
        self.W_i = nn.Parameter(torch.zeros(input_dim, n_units, n_units))
        self.W_f = nn.Parameter(torch.zeros(input_dim, n_units, n_units))
        self.W_o = nn.Parameter(torch.zeros(input_dim, n_units, n_units))
        self.b_j = nn.Parameter(torch.zeros(input_dim, n_units))
        self.b_i = nn.Parameter(torch.zeros(input_dim, n_units))
        self.b_f = nn.Parameter(torch.zeros(input_dim, n_units))
        self.b_o = nn.Parameter(torch.zeros(input_dim, n_units))
        self.F_alpha_n = nn.Parameter(torch.zeros(input_dim, n_units, 1))
        self.F_alpha_n_b = nn.Parameter(torch.zeros(input_dim, 1))
        self.F_beta = nn.Linear(2*n_units, 1)
        self.Phi = nn.Linear(2*n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim
        self.device = device
    
    @torch.jit.script_method
    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(self.device)
        c_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(self.device)
        outputs = torch.jit.annotate(List[Tensor], [])
        for t in range(x.shape[1]):
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_j) + self.b_j)
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_o) + self.b_o)
            c_tilda_t = c_tilda_t*f_tilda_t + i_tilda_t*j_tilda_t
            h_tilda_t = (o_tilda_t*torch.tanh(c_tilda_t))
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n)+self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas/torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas*outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas/torch.sum(betas, dim=1, keepdim=True)
        mean = torch.sum(betas*mu, dim=1)

        return mean, alphas, betas

def cql_loss(q, a):
    """Computes the CQL loss for a batch of Q-values and actions."""
    logsumexp = torch.logsumexp(q, dim=1, keepdim=True).mean()
    q_a = q.gather(1, a).mean()
    return logsumexp - q_a