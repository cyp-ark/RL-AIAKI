import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch

def show_AUROC(df,true_label,prob):
    fpr, tpr, thresholds = roc_curve(df[true_label],df[prob])
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]

    sens, spec = tpr[ix], 1-fpr[ix]

    plt.plot(fpr, tpr)

    plt.xlabel('FP Rate')
    plt.ylabel('TP Rate')
    plt.scatter(fpr[ix],tpr[ix],marker='+',color='r', label = 'Best threshold = %.3f, \nSensitivity = %.3f, \nSpecificity = %.3f' % (best_thresh, sens, spec))
    plt.legend()
    plt.title('D-Network')

    plt.show()

    print('AUROC:',roc_auc_score(df[true_label], df[prob]))

def plot_alpha(alphas,cols):
    fig, ax = plt.subplots(figsize=(100, 100))
    im = ax.imshow(alphas)
    ax.set_xticks(np.arange(24))
    ax.set_yticks(np.arange(125))
    ax.set_xticklabels(["t-"+str(i) for i in np.arange(23, -1, -1)])
    ax.set_yticklabels(list(cols))
    for i in range(len(cols)):
        for j in range(24):
            text = ax.text(j, i, round(alphas[i, j], 3),
                        ha="center", va="center", color="w")
    ax.set_title("Importance of features and timesteps")
    #fig.tight_layout()
    plt.show()


def make_betas(betas,cols):
    #betas
    betas = pd.DataFrame([cols,betas]).transpose()
    betas.columns = ['var','betas']
    betas_non_presense = betas.copy()
    betas_non_presense = betas_non_presense[betas_non_presense['var'].isin([x for x in cols if x[:10]!='s:presense'])]

    betas['betas'] = betas['betas'] - betas['betas'].mean()
    betas.sort_values(by='betas',inplace=True)

    betas_non_presense['betas'] = betas_non_presense['betas'] - betas_non_presense['betas'].mean()
    betas_non_presense.sort_values(by='betas',inplace=True)
    return betas,betas_non_presense


def plot_beta(betas):
    plt.figure(figsize=(25, 8))
    plt.title("Feature importance")
    plt.bar(range(len(betas)), betas['betas'])
    plt.xticks(ticks=range(len(betas)), labels=betas['var'], rotation=90, size=10, fontsize=10)
    plt.show()


def make_transition_test_for_AKI(data,rolling_size=24,batch_size=256):
    df = pd.read_parquet(data)
    s_col = [x for x in df.columns if x[:2]=='s:']
    a_col = [x for x in df.columns if x[:2]=='a:']
    r_col = [x for x in df.columns if x=='r:AKI_stage3']
    m_col = [x for x in df.columns if x[:2]=='m:']
    dict = {}
    dict['traj'] = {}
    
    print(m_col)

    s  = []
    a  = []
    r  = []
    m1 = []
    m2 = []

    for traj in tqdm(df.traj.unique()):
        df_traj = df[df['traj'] == traj]
        dict['traj'][traj] = {'s':[],'a':[]}
        dict['traj'][traj]['s'] = df_traj[s_col].values.tolist()
        dict['traj'][traj]['a'] = df_traj[a_col].values.tolist()
        dict['traj'][traj]['r'] = df_traj[r_col].values.tolist()
        dict['traj'][traj]['m1'] = df_traj[m_col[0]].values.tolist()
        dict['traj'][traj]['m2'] = df_traj[m_col[1]].values.tolist()
        

        step_len = len(df_traj) - rolling_size + 1
        for step in range(step_len):
            current_state = dict['traj'][traj]['s'][step:step+rolling_size]
            current_state[-1][-12:]=[0,0,0,0,0,0,0,0,0,0,0,0]
            s.append(current_state)
            a.append(dict['traj'][traj]['a'][step+rolling_size-1])
            r.append(dict['traj'][traj]['r'][step+rolling_size-1])
            m1.append(dict['traj'][traj]['m1'][step+rolling_size-1])
            m2.append(dict['traj'][traj]['m2'][step+rolling_size-1])
    
    s = torch.FloatTensor(np.float32(s))
    a = torch.LongTensor(np.int64(a))
    r  = torch.FloatTensor(np.float32(r))
    m1 = torch.FloatTensor(np.float32(m1))
    m2 = torch.FloatTensor(np.float32(m2))

    Dataset = TensorDataset(s,a,r,m1,m2)
    rt = DataLoader(Dataset,batch_size,shuffle=False)
    return rt

def make_transition_test(data,rolling_size=24,batch_size=256):
    df = pd.read_parquet(data)
    s_col = [x for x in df if x[:2]=='s:']
    a_col = [x for x in df if x[:2]=='a:']
    r_col = [x for x in df if x[:2]=='r:']
    dict = {}
    dict['traj'] = {}

    s  = []
    a  = []
    r1 = []
    r2 = []
    r3 = []
    r1_patient = []
    r2_patient = []
    r24 = []

    for traj in tqdm(df.traj.unique()):
        df_traj = df[df['traj'] == traj]
        dict['traj'][traj] = {'s':[],'a':[]}
        dict['traj'][traj]['s'] = df_traj[s_col].values.tolist()
        dict['traj'][traj]['a'] = df_traj[a_col].values.tolist()
        dict['traj'][traj]['r1'] = df_traj[r_col[0]].values.tolist()
        dict['traj'][traj]['r2'] = df_traj[r_col[1]].values.tolist()
        dict['traj'][traj]['r3'] = df_traj[r_col[2]].values.tolist()
        if sum(df_traj[r_col[0]].values.tolist()) < 0 : r1p = 1
        else : r1p = 0
        if sum(df_traj[r_col[1]].values.tolist()) > 0 : r2p = 1
        else : r2p = 0

        step_len = len(df_traj) - rolling_size + 1
        for step in range(step_len):
            current_state = dict['traj'][traj]['s'][step:step+rolling_size]
            current_state[-1][-12:]=[0,0,0,0,0,0,0,0,0,0,0,0]
            s.append(current_state)
            a.append(dict['traj'][traj]['a'][step+rolling_size-1])
            r1.append(dict['traj'][traj]['r1'][step+rolling_size-1])
            r2.append(dict['traj'][traj]['r2'][step+rolling_size-1])
            r3.append(dict['traj'][traj]['r3'][step+rolling_size-1])
            r1_patient.append(r1p)
            r2_patient.append(r2p)
            if step <= step_len - 6:
                r24.append(0)
            else :
                r24.append(1)
    
    s = torch.FloatTensor(np.float32(s))
    a = torch.LongTensor(np.int64(a))
    r1 = torch.FloatTensor(np.float32(r1))
    r2 = torch.FloatTensor(np.float32(r2))
    r3 = torch.FloatTensor(np.float32(r3))
    r1_patient = torch.LongTensor(np.int64(r1_patient))
    r2_patient = torch.LongTensor(np.int64(r2_patient))
    r24 = torch.LongTensor(np.int64(r24))

    Dataset = TensorDataset(s,a,r1,r2,r3,r1_patient,r2_patient,r24)
    rt = DataLoader(Dataset,batch_size,shuffle=False)
    return rt