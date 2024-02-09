import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import timedelta
import os, math, torch
import torch.nn as nn
import torch.nn.functional as F

class DataLoader(object):
    def __init__(self, df, rng, device, minibatch_size, rolling_size, type):
        self.df = pd.read_parquet(df)
        self.dict = {}
        self.rng = rng
        self.device = device
        self.minibatch_size = minibatch_size
        self.rolling_size = rolling_size
        self.type = type

        self.transition = {}
        self.index_transition = None
        self.index_pos_last = []
        self.index_neg_last = []
        self.data_size = None

    def reset(self, shuffle, pos_samples_in_minibatch, neg_samples_in_minibatch):
        self.ps = pos_samples_in_minibatch
        self.ns = neg_samples_in_minibatch
        if shuffle:
            self.rng.shuffle(self.transition_indices)
            self.rng.shuffle(self.transition_indices_pos_last)
            self.rng.shuffle(self.transition_indices_neg_last)
        self.transitions_head = 0
        self.transitions_head_pos = 0
        self.transitions_head_neg = 0
        self.epoch_finished = False
        self.epoch_pos_finished = False
        self.epoch_neg_finished = False
        self.num_minibatches_epoch = int(np.floor(self.transition_data_size / self.minibatch_size)) + int(1 - self.drop_smaller_than_minibatch)

    def make_transition(self):
        self.dict['traj'] = {}
        self.transition['s'] = {}
        self.transition['a'] = {}
        self.transition['r'] = {}
        self.transition['next_s'] = {}
        self.transition['terminal'] = {}
        self.transition['pos_traj'] = []
        self.transition['neg_traj'] = []
        count = 0
        s = [x for x in self.df.columns if x[:2]=='s:']
        a = [x for x in self.df.columns if x[:2]=='a:']
        r = [x for x in self.df.columns if x==('r:'+self.type)]

        for traj in tqdm(self.df.traj.unique()):
            df_i = self.df[self.df['traj']==traj].sort_values(by='step')
            self.dict['traj'][traj] = {'s': [], 'a': [], 'r': []}
            self.dict['traj'][traj]['s'] = df_i[s]
            self.dict['traj'][traj]['a'] = df_i[a]
            self.dict['traj'][traj]['r'] = df_i[r]
            if sum(df_i[r].values) > 0:
                self.transition['pos_traj'].append(traj)
            else :
                self.transition['neg_traj'].append(traj)

            t_len = len(self.dict['traj'][traj]['s']) - self.rolling_size - 1
            for t in range(t_len):
                self.transition['s'][count] = self.dict['traj'][traj]['s'][t:t+self.rolling_size]
                self.transition['a'][count] = self.dict['traj'][traj]['a'][t+self.rolling_size-1:t+self.rolling_size]
                self.transition['r'][count] = self.dict['traj'][traj]['r'][t+self.rolling_size-1:t+self.rolling_size]
                self.transition['next_s'][count] = self.dict['traj'][traj]['s'][t+1:t+self.rolling_size+1]
                self.transition['terminal'][count] = 0
                count += 1
            tlast = t_len + 1
            self.transition['s'][count] = self.dict['traj'][traj]['s'][tlast:tlast+self.rolling_size]
            self.transition['a'][count] = self.dict['traj'][traj]['a'][tlast+self.rolling_size-1:tlast+self.rolling_size]
            self.transition['r'][count] = self.dict['traj'][traj]['r'][tlast+self.rolling_size-1:tlast+self.rolling_size]
            self.transition['next_s'][count] = self.dict['traj'][traj]['s'][tlast+1:tlast+self.rolling_size+1]
            self.transition['terminal'][count] = 1
            #if traj in self.encoded_data['pos_traj']:self.pos_last.append(count)
            #else : self.neg_last.append(count)
            count += 1
        self.data_size = count
        self.index_transition = np.arange(self.data_size)

    def get_next_minibatch(self):
        if self.epoch_finished == True:
            print('Epoch finished, please call reset() method before next call to get_next_minibatch()')
            return None
        # Getting data from dictionaries
        offset = self.ns + self.ps
        minibatch_main_index_list = list(self.transition_indices[self.transitions_head:self.transitions_head + self.minibatch_size - offset])
        minibatch_pos_last_index_list = self.transition_indices_pos_last[self.transitions_head_pos:self.transitions_head_pos + self.ps]
        minibatch_neg_last_index_list = self.transition_indices_neg_last[self.transitions_head_neg:self.transitions_head_neg + self.ns]
        self.transitions_head_pos += self.ps
        self.transitions_head_neg += self.ns
        minibatch_index_list = minibatch_main_index_list + minibatch_pos_last_index_list + minibatch_neg_last_index_list
        get_from_dict = operator.itemgetter(*minibatch_index_list)
        s_minibatch = get_from_dict(self.transition_data['s'])
        actions_minibatch = get_from_dict(self.transition_data['actions'])
        rewards_minibatch = get_from_dict(self.transition_data['rewards'])
        next_s_minibatch = get_from_dict(self.transition_data['next_s'])
        terminals_minibatch = get_from_dict(self.transition_data['terminals'])
        # Updating current data head
        self.transitions_head += self.minibatch_size
        self.epoch_finished = self.transitions_head + self.drop_smaller_than_minibatch*self.minibatch_size >= self.transition_data_size
        self.transitions_head_pos = self.transitions_head_pos % len(self.transition_indices_pos_last)
        self.transitions_head_neg = self.transitions_head_neg % len(self.transition_indices_neg_last)
        return s_minibatch, actions_minibatch, rewards_minibatch, next_s_minibatch, terminals_minibatch, self.epoch_finished


def make_data_loaders(train_data, validation_data, rng, device, minibatch_size, rolling_size, type):
    # Note that the loaders will be reset in Experiment
    loader_train = DataLoader(train_data, rng, device, minibatch_size, rolling_size, type)
    loader_validation = DataLoader(validation_data, rng, device, minibatch_size, rolling_size, type)
    loader_train.make_transition()
    loader_validation.make_transition()
    return loader_train, loader_validation