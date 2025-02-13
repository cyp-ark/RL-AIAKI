import pandas as pd
import numpy as np
import os, yaml, wandb, pickle, optuna, gc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader, Sampler

from rl import make_transition, imvt, imvt2, CustomSampler
from make_plot import show_AUROC, plot_alpha, plot_beta, make_transition_test, make_betas, make_transition_test_for_AKI

from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score

np.random.seed(params['random_seed'])
torch.manual_seed(params['random_seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(params['random_seed'])
    torch.cuda.manual_seed_all(params['random_seed'])

def objective(trial):
    batch_size = trial.suggest_categorical("batch_size",[32,64])
    n_units = trial.suggest_categorical("n_units",[2,4,8,16,32,64])
    
    lr = trial.suggest_categorical("learning_rate",[1e-6,5e-6,1e-5,5e-5,1e-4])
    lr_decay = trial.suggest_categorical("lr_decay",[0.75,0.8,0.85,0.9,0.95,1])
    lr_step = trial.suggest_categorical("lr_step",[2,5,10])

    ns = trial.suggest_categorical("negative_sampling",[2,4,6,8])
    
    loss = trial.suggest_categorical("loss",['smooth_l1','mse'])
    
    update_freq = trial.suggest_categorical("update_freq",[2,4,8,16,32])
    
    epochs = 50

    wandb.init(
        project='IMV_LSTM_AKI_new', name=f'trial-{trial.number}', reinit=True,
        config={
        "batch_size":batch_size,
        "n_units":n_units,
        "learning_rate":lr,
        "lr_decay":lr_decay,
        "lr_step":lr_step,
        "ns":ns,
        "loss":loss,
        "update_freq":update_freq
    })

    auroc = _train(batch_size,n_units,lr,lr_decay,lr_step,ns,loss,epochs,update_freq)

    return auroc

def _train(batch_size,n_units,lr,lr_decay,lr_step,ns,loss_type,epochs,update_freq=2):
    network = imvt2(input_dim=params['state_dim'], output_dim=params['num_actions'], n_units=n_units, device=device).to(device)
    target_network = imvt2(input_dim=params['state_dim'], output_dim=params['num_actions'], n_units=n_units, device=device).to(device)
    gamma = 1.0
    patience = 5
    best_loss = 1e6

    optimizer = optim.Adam(network.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=lr_decay)

    num_workers = 4

    sampler = CustomSampler(train_data,batch_size,ns=ns,target=target)
    train_loader = DataLoader(train_data,batch_sampler=sampler,num_workers=num_workers)
    val_loader = DataLoader(val_data,batch_size=256,shuffle=False)
    

    for epoch in range(epochs):
        train_loss = 0
        update_counter = 0
        for s,a,r,s2,t in tqdm(train_loader):            
            s = s.to(device)
            a = a.to(device)
            r = r.to(device)
            s2 = s2.to(device)
            t = t.to(device)

            q,_,_ = network(s)
            q_pred = q.gather(1, a).squeeze()
            
            with torch.no_grad():
                q2,_,_ = target_network(s2)
                q2_net,_,_ = network(s2)

            q2_max = q2.gather(1, torch.max(q2_net,dim=1)[1].unsqueeze(1)).squeeze(1).detach()

            bellman_target = torch.clamp(r, max=0.0, min=-1.0) + gamma * torch.clamp(q2_max.detach(), max=0.0, min=-1.0)*(1-t)
            if loss_type == "l1":loss = F.l1_loss(q_pred, bellman_target)
            elif loss_type == "smooth_l1":loss = F.smooth_l1_loss(q_pred, bellman_target)
            elif loss_type == "mse":loss = F.mse_loss(q_pred, bellman_target)

            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            update_counter += 1
            if update_counter == update_freq:
                target_network.load_state_dict(network.state_dict())
                update_counter = 0

        with torch.no_grad():
            val_loss = 0
            for s,a,r,s2,t in val_loader:
                s = s.to(device)
                a = a.to(device)
                r = r.to(device)
                s2 = s2.to(device)
                t = t.to(device)

                q,_,_ = network(s)
                q2,_,_ = target_network(s2)
                q2 = q2.detach()
                q_pred = q.gather(1, a).squeeze()

                q2_net,_,_ = network(s2)
                q2_net = q2_net.detach()
                q2_max = q2.gather(1, torch.max(q2_net,dim=1)[1].unsqueeze(1)).squeeze()

                bellman_target = torch.clamp(r, max=0.0, min=-1.0) + gamma * torch.clamp(q2_max.detach(), max=0.0, min=-1.0)*(1-t)
                if loss_type == "l1":loss = F.l1_loss(q_pred, bellman_target)
                elif loss_type == "smooth_l1":loss = F.smooth_l1_loss(q_pred, bellman_target)
                elif loss_type == "mse":loss = F.mse_loss(q_pred, bellman_target)
                val_loss += loss.item()

            q_value = []
            aki1 = []
            aki2 = []
            reward = []
            for s,a,r,m1,m2 in val_transition:
                s = s.to(device)
                q,_,_ = network(s.to(device))
                aki1.append(m1.detach().cpu().numpy())
                aki2.append(m2.detach().cpu().numpy())
                q_value.append(q.detach().cpu().numpy())
                reward.append(r.detach().cpu().numpy())
            aki1 = 1 - np.concatenate(aki1,axis=0)
            aki2 = 1 - np.concatenate(aki2,axis=0)
            q_value = 1 + np.concatenate(q_value,axis=0)
            reward  = 1 + np.concatenate(reward,axis=0)
            
            q_max = q_value.max(axis=1)
            q_median = np.median(q_value, axis=1)
            
            auroc      = roc_auc_score(reward,q_max)
            auroc_med  = roc_auc_score(reward,q_median)
            auroc1_max = roc_auc_score(aki1,q_max)
            auroc1_med = roc_auc_score(aki1,q_median)
            auroc2_max = roc_auc_score(aki2,q_max)
            auroc2_med = roc_auc_score(aki2,q_median)

        
        if (epoch%lr_step ==0):
            scheduler.step()
        
        if val_loss < best_loss:
            best_loss = val_loss
            counters = 0
        else :
            counters += 1

        wandb.log({"Iter:": epoch, "train:":train_loss, "val:":val_loss, "AUROC":auroc, "AUROC_median":auroc_med,"AUROC_stage_1_max":auroc1_max, "AUROC_stage_1_median":auroc1_med,"AUROC_stage_2_max":auroc2_max,"AUROC_stage_2_median":auroc2_med,"counters":counters})

        if (counters > patience)&(epoch>=20):
            break

    gc.collect()
    torch.cuda.empty_cache()
    return auroc

if __name__ == "__main__":
    train = './code/train_4hrs_mean.parquet'
    val = './code/val_4hrs_mean.parquet'
    target = -1

    train_data = make_transition(train,"r:AKI_stage3",target,rolling_size=6)
    val_data = make_transition(val,"r:AKI_stage3",target,rolling_size=6)
    val_transition = make_transition_test_for_AKI(val,rolling_size=6)
    
    device = 'cuda:0'
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1000)

    best_params = study.best_params
    best_loss = study.best_value

    print("Best Hyperparameters:", best_params)
    print("Best Validation Loss:", best_loss)