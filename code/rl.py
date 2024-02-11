import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import timedelta
import os, math, torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.)

class QNetwork(nn.Module):
    def __init__(self, state_dim, nb_actions=None):
        super(QNetwork_64, self).__init__()

        self.state_dim = state_dim
        self.nb_actions = nb_actions
        
        self.fc = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.nb_actions)
        )
        self.fc.apply(init_weights)
        
    def forward(self, x):
        x = self.fc(x)
        return x

class IMVFullLSTM(torch.jit.ScriptModule):
    __constants__ = ["n_units", "input_dim"]
    def __init__(self, state_dim, nb_actions, n_units, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.W_i = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        self.W_f = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        self.W_o = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1)*init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1)*init_std)
        self.F_beta = nn.Linear(2*n_units, 1)
        self.Phi = nn.Linear(2*n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim
    @torch.jit.script_method
    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cuda()
        c_t = torch.zeros(x.shape[0], self.input_dim*self.n_units).cuda()
        outputs = torch.jit.annotate(List[Tensor], [])
        for t in range(x.shape[1]):
            # eq 1
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_j) + self.b_j)
            inp =  torch.cat([x[:, t, :], h_tilda_t.view(h_tilda_t.shape[0], -1)], dim=1)
            # eq 2
            i_t = torch.sigmoid(self.W_i(inp))
            f_t = torch.sigmoid(self.W_f(inp))
            o_t = torch.sigmoid(self.W_o(inp))
            # eq 3
            c_t = c_t*f_t + i_t*j_tilda_t.reshape(j_tilda_t.shape[0], -1)
            # eq 4
            h_tilda_t = (o_t*torch.tanh(c_t)).view(h_tilda_t.shape[0], self.input_dim, self.n_units)
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        # eq 8
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) +self.F_alpha_n_b)
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


class RL(object):
    def __init__(self, state_dim, nb_actions, n_units, gamma,
                 learning_rate, update_freq, rng, device):
        self.rng = rng
        self.state_dim = state_dim
        self.nb_actions = nb_actions
        self.n_units = n_units
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.update_freq = update_freq
        self.update_counter = 0
        self.device = device
        self.network = IMVFullLSTM(state_dim=self.state_dim, nb_actions=self.nb_actions, n_units=self.n_units)
        self.target_network = IMVFullLSTM(state_dim=self.state_dim, nb_actions=self.nb_actions)
        self.weight_transfer(from_model=self.network, to_model=self.target_network)
        self.network.to(self.device)
        self.target_network.to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, amsgrad=True)

    def train_on_batch(self, s, a, r, s2, t):
        s  = torch.FloatTensor(np.float32(s)).to(self.device)
        s2 = torch.FloatTensor(np.float32(s2)).to(self.device)
        a  = torch.LongTensor(np.int64(a)).to(self.device)
        r  = torch.FloatTensor(np.float32(r)).to(self.device)
        t  = torch.FloatTensor(np.float32(t)).to(self.device)

        with torch.no_grad():
            q,_,_ = self.network(s).detach()
            q2,_,_ = self.target_network(s2).detach()
            q2_net,_,_ = self.network(s2).detach()
            q2_max = q2.gather(1, torch.max(q2_net, 1)[1].unsqueeze(1)).squeeze(1)
        q_pred = q.gather(1, a.unsqueeze(1)).squeeze(1) 

        bellman_target = torch.clamp(r, max=1.0, min=0.0) + self.gamma * torch.clamp(q2_max.detach(), max=1.0, min=0.0) * (1 - t)

        loss = F.smooth_l1_loss(q_pred, bellman_target)            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def get_loss(self, s, a, r, s2, t):
        s  = torch.FloatTensor(np.float32(s)).to(self.device)
        s2 = torch.FloatTensor(np.float32(s2)).to(self.device)
        a  = torch.LongTensor(np.int64(a)).to(self.device)
        r  = torch.FloatTensor(np.float32(r)).to(self.device)
        t  = torch.FloatTensor(np.float32(t)).to(self.device)

        with torch.no_grad():
            q,_,_ = self.network(s).detach()
            q2,_,_ = self.target_network(s2).detach()
            q2_net,_,_ = self.network(s2).detach()
            q2_max = q2.gather(1, torch.max(q2_net, 1)[1].unsqueeze(1)).squeeze(1)
        q_pred = q.gather(1, a.unsqueeze(1)).squeeze(1) 
            

        bellman_target = torch.clamp(r, max=1.0, min=0.0) + self.gamma * torch.clamp(q2_max.detach(), max=1.0, min=0.0) * (1 - t)

        loss = F.smooth_l1_loss(q_pred, bellman_target)
        return loss.detach().cpu().numpy()

    def get_q(self, s):
        s = torch.FloatTensor(s).to(self.device)
        return self.network(s).detach().cpu().numpy()

    def get_max_action(self, s):
        s = torch.FloatTensor(s).to(self.device)
        q = self.network(s).detach()
        return q.max(1)[1].cpu().numpy()

    def get_action(self, states):
        return self.get_max_action(states)

    def learn(self, s, a, r, s2, term):
        """ Learning from one minibatch """
        loss = self.train_on_batch(s, a, r, s2, term)
        if self.update_counter == self.update_freq:
            self.weight_transfer(from_model=self.network, to_model=self.target_network)
            self.update_counter = 0
        else:
            self.update_counter += 1
        return loss

    def weight_transfer(from_model, to_model):
        to_model.load_state_dict(from_model.state_dict())


class DQNExperiment(object):
    def __init__(self, data_loader_train, data_loader_validation, q_network, ps, ns, folder_location, folder_name, saving_period, rng, resume):
        self.rng = rng
        self.data_loader_train = data_loader_train
        self.data_loader_validation = data_loader_validation
        self.q_network = q_network
        self.ps = ps  # num pos samples replaced in each minibatch
        self.ns = ns  # num neg samples replaced in each minibatch
        self.batch_num = 0
        self.saving_period = saving_period  # after each `saving_period` epochs, the results so far will be saved.
        self.resume = resume 
        storage_path = os.path.join(os.path.abspath(folder_location), folder_name)
        self.storage_rl = os.path.join(storage_path, 'rl_' + self.q_network.sided_Q)
        self.checkpoint_folder = os.path.join(storage_path, 'rl_' + self.q_network.sided_Q + '_checkpoints')
        if not os.path.exists(self.storage_rl):
            os.mkdir(self.storage_rl)
        if not os.path.exists(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)
        
    def do_epochs(self, number):
        '''
        Each epoch is one iteration thorugh the entire dataset.
        '''
        self.curr_epoch = 0
        self.all_epoch_steps = []
        self.all_epoch_validation_steps = []
        self.all_epoch_loss = []
        self.all_epoch_validation_loss = []
        self.data_loader_train.reset(shuffle=True, pos_samples_in_minibatch=self.ps, neg_samples_in_minibatch=self.ns)
        self.data_loader_validation.reset(shuffle=False, pos_samples_in_minibatch=0, neg_samples_in_minibatch=0)
        for epoch in range(self.curr_epoch, number):
            print()
            print('>>>>> Experiment ' + 'Q-' + self.q_network.sided_Q + ' Epoch ' + str(epoch + 1) + '/' + str(number))
            # Learn here
            epoch_done = False
            epoch_steps = 0
            epoch_loss = 0
            print('Minibatch learning within epoch')
            bar = pyprind.ProgBar(self.data_loader_train.num_minibatches_epoch)
            while not epoch_done:
                s, actions, rewards, next_s, terminals, epoch_done = self.data_loader_train.get_next_minibatch()
                epoch_steps += len(s)
                loss = self.q_network.learn(s, actions, rewards, next_s, terminals)
                epoch_loss += loss
            self.data_loader_train.reset(shuffle=True, pos_samples_in_minibatch=self.ps, neg_samples_in_minibatch=self.ns)
            self.data_loader_validation.reset(shuffle=False, pos_samples_in_minibatch=0, neg_samples_in_minibatch=0)
            self.all_epoch_loss.append(epoch_loss/epoch_steps)
            self.all_epoch_steps.append(epoch_steps)
            if (epoch + 1)% self.saving_period == 0:
                self._do_eval()
                try:
                    torch.save({
                        'epoch': epoch,
                        'rl_network_state_dict': self.q_network.network.state_dict(),
                        # 'rl_target_network_state_dict': self.q_network.target_network.state_dict(),
                        # 'rl_optimizer_state_dict': self.q_network.optimizer.state_dict(),
                        'loss': self.all_epoch_loss,
                        'validation_loss': self.all_epoch_validation_loss,
                        'epoch_steps': self.all_epoch_steps,
                        'epoch_validation_steps': self.all_epoch_validation_steps,
                    }, os.path.join(self.checkpoint_folder, 'checkpoint' + str(epoch) +'.pt'))
                    np.save(os.path.join(self.storage_rl, 'q_losses.npy'), np.array(self.all_epoch_loss))
                except:
                    print(">>> Cannot save files. On Windows: the files might be open.")
        
    def _do_eval(self):
        epoch_val_steps = 0
        epoch_val_loss = 0
        epoch_done = False
        bar = pyprind.ProgBar(self.data_loader_validation.num_minibatches_epoch)
        while not epoch_done:
            bar.update()
            s, actions, rewards, next_s, terminals, epoch_done = self.data_loader_validation.get_next_minibatch()
            epoch_val_steps += len(s)
            loss = self.q_network.get_loss(s, actions, rewards, next_s, terminals)
            epoch_val_loss += loss
        self.all_epoch_validation_loss.append(epoch_val_loss / epoch_val_steps)
        self.all_epoch_validation_steps.append(epoch_val_steps)
        try:
            np.save(os.path.join(self.storage_rl, 'q_validation_losses.npy'), np.array(self.all_epoch_validation_loss))
        except:
            pass



def train(params, rng, loader_train, loader_validation):
    qnet = RL(state_dim=params["state_dim"], nb_actions=params["num_actions"], gamma=params["gamma"], learning_rate=params["rl_learning_rate"],update_freq=params["update_freq"], rng=params["random_seed"], device=params["device"])
    print('Initializing Experiment')
    expt = DQNExperiment(data_loader_train=loader_train, data_loader_validation=loader_validation, q_network=qnet, ps=0, ns=2,
                        folder_location=params["folder_location"], folder_name=params["folder_name"], 
                        saving_period=params["exp_saving_period"], rng=rng, resume=params["rl_resume"])
    with open(os.path.join(expt.storage_rl, 'config_exp.yaml'), 'w') as y:
            yaml.safe_dump(params, y)  # saving new params for future reference
    print('Running experiment (training Q-Networks)')
    expt.do_epochs(number=params["exp_num_epochs"])
    print("Training Q-Networks finished successfully")