import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from datetime import timedelta
import scipy
#import torch
#import torch.nn as nn
#from torch.distributions import Normal
#import torch.nn.init as init
from sklearn.metrics import precision_recall_curve, auc
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
#from skmultilearn.model_selection import IterativeStratification


def number(df):
    print('number of subject :', len(df.drop_duplicates(subset=["subject_id"])))
    print('number of hadm :', len(df.drop_duplicates(subset=["hadm_id"])))
    print('number of stay :', len(df.drop_duplicates(subset=["stay_id"])))

def time_sort(df):
    df[['charttime', 'intime', 'outtime']] = df[['charttime', 'intime', 'outtime']].apply(pd.to_datetime)
    df = df.loc[(df['intime'] < df['charttime']) & (df['intime'] < df['outtime']) & (df['charttime'] < df['outtime'])]
    return df.sort_values(by=['stay_id', 'charttime']).reset_index(drop=True)

def filter_RRT_icu(target, RRT):
   
    if not RRT.empty:
        target['outtime'] = RRT['starttime'].iloc[0]
        target = target[target['charttime'] < target['outtime']]
        if not target.empty : target['RRT'].iloc[-1] = 1
        
    return target

def filter_RRT_hosp(target, RRT):

    if not RRT.empty:
        for _, row in RRT.iterrows():
            end_criteria = row['chartdate'] + timedelta(days=1, hours=23, minutes=59)
            start_criteria = row['chartdate']
            target = target[~((target['charttime'] > start_criteria) & (target['charttime'] < end_criteria))]

    return target

def filter_KT(target,KT):
    
    if not KT.empty:
        criteria = KT['chartdate'].min() + timedelta(days=0,hours=0,minutes=0)
        target = target[target['charttime'] < criteria]

    return target

def Pre_admission(df_icu, df_hosp):

    tqdm.pandas(desc="Processing groups")

    df_SCr = pd.concat([df_icu, df_hosp])
    df_icu['intime_7'] = df_icu['intime'] - timedelta(days=7)
    df_icu['intime_365'] = df_icu['intime'] - timedelta(days=365)

    def MDRD(df):
            
        if df['gender'] == 'F' and df['race'] == 'BLACK': df['baseline'] = (75 / (0.742 * 1.21 * 186 * df['anchor_age'] ** (-0.203))) ** (-1 / 1.154)
        elif df['race'] == 'BLACK': df['baseline'] = (75 / (1 * 1.21 * 186 * df['anchor_age'] ** (-0.203))) ** (-1 / 1.154)
        elif df['gender'] == 'F': df['baseline'] = (75 / (0.742 * 1 * 186 * df['anchor_age'] ** (-0.203))) ** (-1 / 1.154)
        else: df['baseline'] = (75 / (1 * 1 * 186 * df['anchor_age'] ** (-0.203))) ** (-1 / 1.154)
    
        return round(df['baseline'], 1)

    def operation(target_icu, target_SCr, stay_id):

        if not target_SCr.empty :

            for i in stay_id:    
                
                intime = target_icu.loc[target_icu['stay_id'] == i, 'intime'].iloc[0]
                intime_7 = target_icu.loc[target_icu['stay_id'] == i, 'intime_7'].iloc[0]

                value_SCr = target_SCr.loc[(target_SCr['charttime'] < intime) & (target_SCr['charttime'] > intime_7), 'SCr'] 

                if not value_SCr.empty : 
                    target_icu.loc[target_icu['stay_id'] == i, ['baseline', 'method']] = value_SCr.min(), 1

                else : 
                    
                    intime_365 = target_icu.loc[target_icu['stay_id'] == i, 'intime_365'].iloc[0]
                    value_SCr = target_SCr.loc[(target_SCr['charttime'] < intime_7) & (target_SCr['charttime'] > intime_365), 'SCr']

                    if not value_SCr.empty : 
                        target_icu.loc[target_icu['stay_id'] == i, ['baseline', 'method']] = np.median(value_SCr), 2

                    else : 
                        target_icu.loc[target_icu['stay_id'] == i, ['baseline', 'method']] = MDRD(target_icu.iloc[0]), 0

        else : 
            target_icu.loc[target_icu['stay_id'] == i, ['baseline', 'method']] = MDRD(target_icu.iloc[0]), 0

        return target_icu

    df_icu = df_icu.groupby('subject_id', group_keys=False).progress_apply(lambda x : operation(x, df_SCr[df_SCr['subject_id'] == x.name], x['stay_id'].drop_duplicates())).reset_index(drop=True)
    
    return df_icu

def diff(df_icu,df_hosp,mode):

    tqdm.pandas(desc="Processing groups")

    if mode == 'After' : 
        df_icu['charttime_6'] = df_icu['charttime'] + timedelta(hours=6)

    elif mode == 'Before' : 
        df_icu['min'] = np.nan
        df_icu['max'] = np.nan
        df_icu['diff'] = np.nan

    df_SCr = pd.concat([df_icu, df_hosp])
    df_SCr = df_SCr[df_SCr['SCr'].notnull()]
    
    def operation(target_icu, target_SCr):
        
        for i in range(len(target_icu)):

            if mode == 'After':
                cri = df_icu['charttime'].iloc[i] + timedelta(hours=6)
                cri_48 = cri - timedelta(days=2)

            else :
                cri = df_icu['charttime'].iloc[i]
                cri_48 = cri - timedelta(days=2)

            value_SCr = target_SCr.loc[(target_SCr['charttime'] < cri) & (target_SCr['charttime'] > cri_48),'SCr']
                                          
            if ((not value_SCr.empty) & (pd.isna(target_icu['min'].iloc[i]))) :
                target_icu['min'].iloc[i] = value_SCr.min() # case 2
                if not pd.isna(target_icu['SCr'].iloc[i]): # case 4
                    target_icu['diff'].iloc[i] = target_icu['SCr'].iloc[i] - value_SCr.min()

            elif (pd.isna(target_icu['min'].iloc[i])) :
                if not pd.isna(target_icu['SCr'].iloc[i]): # case 3
                    target_icu['min'].iloc[i] = target_icu['SCr'].iloc[i] 
                    target_icu['diff'].iloc[i] = 0

            if ((not value_SCr.empty) & (pd.isna(target_icu['max'].iloc[i]))) :
                target_icu['max'].iloc[i] = value_SCr.max() 

            elif (pd.isna(target_icu['max'].iloc[i])) :
                if not pd.isna(target_icu['SCr'].iloc[i]):
                    target_icu['max'].iloc[i] = target_icu['SCr'].iloc[i] 

        return target_icu

    df_icu = df_icu.groupby('subject_id', group_keys=False).progress_apply(lambda x : operation(x, df_SCr[df_SCr['subject_id'] == x.name])).reset_index(drop=True)
    df_icu['diff'] = round(df_icu['diff'],1)

    return df_icu

def SCr_AKI_stage(df):
    
    df['ratio'] = df['SCr'] / df['baseline']
    df['SCr_stage'] = 0

    condition_1 = ((df['ratio'] < 2) & (df['ratio'] >= 1.5)) | ((df['diff'] >= 0.3) & (df['diff'] <= 4))
    condition_2 = (df['ratio'] < 3) & (df['ratio'] >= 2)
    condition_3 = (df['ratio'] >= 3) | (df['diff'] > 4) | (df['RRT'] == 1)

    df.loc[condition_1, 'SCr_stage'] = 1
    df.loc[condition_2, 'SCr_stage'] = 2
    df.loc[condition_3, 'SCr_stage'] = 3

    df.loc[(df['SCr'].isnull()) & (df['RRT'] != 1), 'SCr_stage'] = 0

    return df

def SCr_resample(df,label):

    tqdm.pandas(desc="Processing groups")

    def resample_group(group,label):

        a = group.iloc[[0]].copy()
        b = group.iloc[[0]].copy()

        for frame in [a]:
                frame['charttime'] = frame['intime']
                frame[[label, 'diff', 'min', 'max', 'ratio','SCr_diff','SCr_stage']] = np.nan
        
        for frame in [b]:
                frame['charttime'] = frame['outtime']
                frame[[label, 'diff', 'min', 'max', 'ratio','SCr_diff','SCr_stage']] = np.nan

        group = pd.concat([a, group, b])
        group.set_index('charttime',inplace=True)
        
        Q = group.resample(rule='6H', origin = group['intime'].iloc[0], label='left').last()

        last_contribute = group[(group['charttime'] < group['outtime']) & (group['charttime'] > (group['outtime'] - timedelta(hours=6)))]

        if not last_contribute.empty :
            Q.iloc[-1] = last_contribute.iloc[-1]

        Q['timedelta'] = Q.index - group['intime'].iloc[0]
        Q['subject_id'] = Q['subject_id'].iat[0]
        Q['hadm_id'] = Q['hadm_id'].iat[0]
        Q['stay_id'] = Q['stay_id'].iat[0]
        Q['first_careunit'] = Q['first_careunit'].iat[0]
        Q['intime'] = Q['intime'].iat[0]
        Q['outtime'] = Q['outtime'].iat[0]
        Q['RRT_icu_history'] = Q['RRT_icu_history'].iat[0]
        Q['RRT_hosp_history'] = Q['RRT_hosp_history'].iat[0]
        Q['los'] = Q['los'].iat[0]
        Q['race'] = Q['race'].iat[0]
        Q['anchor_age'] = Q['anchor_age'].iat[0]
        Q['gender'] = Q['gender'].iat[0]
        Q['baseline'] = Q['baseline'].iat[0]
        Q['method'] = Q['method'].iat[0]
        Q['charttime'] = pd.to_datetime(Q.index)
        Q ['RRT'] = Q['RRT'].fillna(0)

        if Q['RRT'].sum() > 0 :
            Q['RRT'].iloc[-1] = 1

        return Q
   
    df = df.groupby('stay_id',group_keys = False).progress_apply(lambda x : resample_group(x,label))
    df['timedelta'] = pd.to_timedelta(df['timedelta'])
    
    return df

def SCr_copy(df, df_icu, df_hosp, label):

    tqdm.pandas(desc="Processing groups")

    df_SCr = pd.concat([df_icu,df_hosp])
    
    def copy(target,label):

        target[label] = target[label].fillna(method='ffill', limit=4)
        target['min'] = target['min'].fillna(method='ffill', limit=4)
        target['max'] = target['max'].fillna(method='ffill', limit=4)
        target['diff'] = target['diff'].fillna(method='ffill', limit=4)
        target['SCr_stage'] = target['SCr_stage'].fillna(method='ffill', limit=4)
        target['SCr_diff'] = target['SCr_diff'].fillna(method='ffill', limit=4)
        target['SCr_charttime_diff'] = target['SCr_charttime_diff'].fillna(method='ffill', limit=4)

        return target
    
    def copy_values(target, target_SCr, label):

        if target[label].isnull().iloc[0] == True:
            
            target_hosp = target_SCr.loc[target_SCr['subject_id'] == target['subject_id'].iloc[0]]
            
            forward = target_hosp[(target_hosp['charttime'] <target['charttime'].iloc[0]) & (target_hosp['charttime'] > (target['charttime'].iloc[0] - timedelta(days=1)))]
            forward = forward.sort_values(['charttime'])

            if not forward.empty:

                target_value = forward.iloc[-1]
                cri = target_value['charttime']
                target_2 = target[(target['charttime'] > cri) & (target['charttime'] < (cri + timedelta(days=1)))]
                target_2 = target_2[label].isnull().sum()

                if target_2 != 0:
                    target.loc[target.index[:target_2], label] = target_value[label]
                    target.loc[target.index[:target_2], 'min'] = target_value['min']
                    target.loc[target.index[:target_2], 'max'] = target_value['max']
                    target.loc[target.index[:target_2], 'diff'] = target_value['diff']
                    target.loc[target.index[:target_2], 'SCr_stage'] = target_value['SCr_stage']
                    target.loc[target.index[:target_2], 'SCr_diff'] = target_value['SCr_diff']
                    target.loc[target.index[:target_2], 'SCr_charttime_diff'] = target_value['SCr_charttime_diff']
                    
        return target

    df = df.groupby('stay_id', group_keys=False).progress_apply(lambda x : copy(x, label)).reset_index(drop=True)
    print(df[label].isnull().sum())

    df = df.groupby('stay_id', group_keys=False).progress_apply(lambda x : copy_values(x, df_SCr, label)).reset_index(drop=True)
    print(df[label].isnull().sum())

    return df

def SCr_mask(target):

    target['SCr_mask'] = target['SCr'].notna().astype(int)
    target['SCr'] = target['SCr'].fillna(1)

    target['min_mask'] = target['min'].notna().astype(int)
    target['min'] = target['min'].fillna(1)

    target['max_mask'] = target['max'].notna().astype(int)
    target['max'] = target['max'].fillna(1)

    target['diff_mask'] = target['diff'].notna().astype(int)
    target['diff'] = target['diff'].fillna(1)

    target['SCr_stage_mask'] = target['SCr_stage'].notna().astype(int)
    target['SCr_stage'] = target['SCr_stage'].fillna(1)

    target['SCr_diff_mask'] = target['SCr_diff'].notna().astype(int)
    target['SCr_diff'] = target['SCr_diff'].fillna(1)

    target['SCr_charttime_diff_mask'] = target['SCr_charttime_diff'].notna().astype(int)
    target['SCr_charttime_diff'] = target['SCr_charttime_diff'].fillna(1)

    return target

def copy_copy_values(df, df_icu, df_hosp, label):

    tqdm.pandas(desc="Processing groups")


    df_icu = df_icu.rename(columns = {'valuenum': label})
    
    df_SCr = pd.concat([df_icu,df_hosp])
    
    def copy(target,label):

        target[label] = target[label].fillna(method='ffill', limit=4)
        target[label + '_diff'] = target[label + '_diff'].fillna(method='ffill', limit=4)
        
        return target
    
    def copy_values(target, hosp, label):

        if target[label].isnull().iloc[0] == True:
            
            target_hosp = hosp.loc[hosp['subject_id'] == target['subject_id'].iloc[0]]
            
            forward = target_hosp[(target_hosp['charttime'] < target['charttime'].iloc[0]) & (target_hosp['charttime'] > (target['charttime'].iloc[0] - timedelta(days=1)))]
            forward = forward.sort_values(['charttime'])

            if not forward.empty:
               
                target_value = forward.iloc[-1]
                cri = target_value['charttime']
                target_2 = target[(target['charttime'] > cri) & (target['charttime'] < (cri + timedelta(days=1)))]
                target_2 = target_2[label].isnull().sum()

                if target_2 != 0:
                    target.loc[target.index[:target_2], label] = target_value[label]
                    target.loc[target.index[:target_2], label + '_diff'] = target_value[label + '_diff']

        return target

    df = df.groupby('stay_id', group_keys=False).progress_apply(lambda x : copy(x, label)).reset_index(drop=True)
    print(df[label].isnull().sum())

    df = df.groupby('stay_id', group_keys=False).progress_apply(lambda x : copy_values(x, df_SCr, label)).reset_index(drop=True)
    print(df[label].isnull().sum())
    
    df[label + '_mask']  = df[label].notna().astype(int)
    df[label]  = df[label].fillna(0)

    df[label + '_diff_mask']  = df[label + '_diff'].notna().astype(int)
    df[label + '_diff'] =  df[label + '_diff'].fillna(0)

    return df


def Weight(target_icu, target_Weight):

    for i in range(len(target_Weight)):
        target_icu.loc[target_icu['charttime'] > target_Weight['charttime'].iloc[i],'Weight'] = target_Weight['Weight'].iloc[i]

    return target_icu

def Urine(df):

    df = df.sort_values(by=['stay_id', 'charttime'])
    df['charttime_diff'] = df.groupby('stay_id')['charttime'].diff().fillna(timedelta(seconds=0))
    df = df.assign(**{'6h-12h': 0, '12h': 0, '24h': 0, 'anuria_12h': 0, 'Urine_stage': 0, 'Urine_output_rate': 0})

    anuria_threshold = 50.0

    def process_group(target):

        target = target.reset_index(drop=True)
        target['Urine_output_rate'] = target['Urine'] / (target['charttime_diff'].dt.total_seconds() / 3600.0) / target['Weight']
        target['cum_value'] = target['Urine'][::-1].cumsum()
        target['cum_time_diff'] = target['charttime_diff'][::-1].cumsum().dt.total_seconds() / 3600.0

        for i in range(1, len(target)):

            group = target.iloc[1:i+1]
            group['cum_value'] = group['cum_value'] - group['cum_value'].iloc[-1]
            group['cum_time_diff'] = group['cum_time_diff'] - group['cum_time_diff'].iloc[-1]

            for threshold_hours_min, threshold_hours_max, rate_threshold, column_name, stage in [
                (6, 12, 0.5, '6h-12h', 1), 
                (12, float('inf'), 0.5, '12h', 2), 
                (24, float('inf'), 0.3, '24h', 3)]:

                condition = (group['cum_time_diff'] >= threshold_hours_min) & (group['cum_time_diff'] < threshold_hours_max)
                filtered_group = group.loc[condition]

                if not filtered_group.empty:
                    urine_output_rate = filtered_group['cum_value'] / filtered_group['cum_time_diff'] / target['Weight'].iloc[i]
                    if (column_name in ['6h-12h', '12h', '24h']) and (urine_output_rate.iloc[-1] < rate_threshold):
                        target.loc[i, column_name] = 1
                        target.loc[i, 'Urine_stage'] = stage

                    if (column_name == '12h') and (filtered_group['cum_value'].iloc[-1] < anuria_threshold):
                        target.loc[i, 'anuria_12h'] = 1
                        target.loc[i, 'Urine_stage'] = 3

        return target

    tqdm.pandas(desc="Processing groups")
    df = df.groupby('stay_id', group_keys=False).progress_apply(process_group).reset_index(drop=True)

    df = df[(~df['Urine_output_rate'].isna()) & (df['Urine_output_rate'] != float('inf'))]

    df.loc[df['6h-12h'] == 0, '12h'] = 0
    df.loc[df['6h-12h'] == 0, '24h'] = 0
    df.loc[df['12h'] == 0, '24h'] = 0

    df.loc[df['6h-12h'] == 0, 'Urine_stage'] = 0
    df.loc[df['6h-12h'] == 1, 'Urine_stage'] = 1
    df.loc[df['12h'] == 1, 'Urine_stage'] = 2
    df.loc[df['24h'] == 1, 'Urine_stage'] = 3
    df.loc[df['anuria_12h'] == 1, 'Urine_stage'] = 3

    return df

def Urine_resample(df,label):

    tqdm.pandas(desc="Processing groups")

    def resample_group(group,label):

        a = group.iloc[[0]].copy()
        b = group.iloc[[0]].copy()
    
        for frame in [a]:
                frame['charttime'] = frame['intime']
                frame[[label,'Weight','6h-12h','12h','24h','anuria_12h','Urine_stage','Urine_output_rate_diff','Urine_charttime_diff']] = np.nan
        
        for frame in [b]:
                frame['charttime'] = frame['outtime']
                frame[[label,'Weight','6h-12h','12h','24h','anuria_12h','Urine_stage','Urine_output_rate_diff','Urine_charttime_diff']] = np.nan

        group = pd.concat([a, group, b])
        group.set_index('charttime',inplace=True)
        
        Q = group.resample(rule='6H', origin = group['intime'].iloc[0], label='left').last()
        Q['timedelta'] = Q.index - group['intime'].iloc[0]
        Q['subject_id'] = Q['subject_id'].iat[0]
        Q['hadm_id'] = Q['hadm_id'].iat[0]
        Q['stay_id'] = Q['stay_id'].iat[0]
        Q['intime'] = Q['intime'].iat[0]
        Q['outtime'] = Q['outtime'].iat[0]
        Q['charttime'] = pd.to_datetime(Q.index)

        return Q
   
    df = df.groupby('stay_id', group_keys=False).progress_apply(lambda x : resample_group(x,label)).reset_index(drop=True)
    df['timedelta'] = pd.to_timedelta(df['timedelta'])
    df['current_charttime'] = round((df['timedelta'] + timedelta(hours=6)).dt.total_seconds() / 3600,1)
    
    return df

def Urine_mask(target):

    target['Urine_output_rate_diff_mask'] = target['Urine_output_rate_diff'].isna().astype(int)
    target['Urine_output_rate_diff'] = target['Urine_output_rate_diff'].fillna(0)

    target['6h-12h_mask'] = target['6h-12h'].isna().astype(int)
    target['6h-12h'] = target['6h-12h'].fillna(0)

    target['12h_mask'] = target['12h'].isna().astype(int)
    target['12h'] = target['12h'].fillna(0)

    target['24h_mask'] = target['24h'].isna().astype(int)
    target['24h'] = target['24h'].fillna(0)

    target['anuria_12h_mask'] = target['anuria_12h'].isna().astype(int)
    target['anuria_12h'] = target['anuria_12h'].fillna(0)

    target['Urine_stage_mask'] = target['Urine_stage'].isna().astype(int)
    target['Urine_stage'] = target['Urine_stage'].fillna(0)

    target['Urine_charttime_diff_mask'] = target['Urine_charttime_diff'].isna().astype(int)
    target['Urine_charttime_diff'] = target['Urine_charttime_diff'].fillna(0)

    target['Weight_mask'] = target['Weight'].isna().astype(int)
    target['Weight'] = target['Weight'].fillna(0)

    return target


def GT(df,AKI,RRT):
    
    df['GT_presence_6'] = 0
    df['GT_presence_12'] = 0
    df['GT_presence_18'] = 0
    df['GT_presence_24'] = 0

    df['GT_stage_1'] = 0
    df['GT_stage_2'] = 0
    df['GT_stage_3'] = 0
    df['GT_stage_3D'] = 0

    df['charttime_6'] =  df['charttime'] + timedelta(hours=6)

    def operation(target,stage_pool,RRT_pool):

        length = len(target)        

        for j in range(length):

            cri = target['charttime_6'].iloc[j]

            area_6 = stage_pool[(stage_pool['charttime'] > cri) & (stage_pool['charttime'] < (cri + timedelta(hours=6)))]['stage'].sum()
            area_12 = stage_pool[(stage_pool['charttime'] > cri) & (stage_pool['charttime'] < (cri + timedelta(hours=12)))]['stage'].sum()
            area_18 = stage_pool[(stage_pool['charttime'] > cri) & (stage_pool['charttime'] < (cri + timedelta(hours=18)))]['stage'].sum()
            area_24 = stage_pool[(stage_pool['charttime'] > cri) & (stage_pool['charttime'] < (cri + timedelta(hours=24)))]['stage']

            RRT_6 = RRT_pool[(RRT_pool['charttime'] > cri) & (RRT_pool['charttime'] < (cri + timedelta(hours=6)))]['RRT'].sum()
            RRT_12 = RRT_pool[(RRT_pool['charttime'] > cri) & (RRT_pool['charttime'] < (cri + timedelta(hours=12)))]['RRT'].sum()
            RRT_18 = RRT_pool[(RRT_pool['charttime'] > cri) & (RRT_pool['charttime'] < (cri + timedelta(hours=18)))]['RRT'].sum()
            RRT_24 = RRT_pool[(RRT_pool['charttime'] > cri) & (RRT_pool['charttime'] < (cri + timedelta(hours=24)))]['RRT'].sum()

            if not area_24.empty :
                area_24_max = max(area_24)
            
            else : area_24_max = 0

            if RRT_24 > 0 :
                    target['GT_stage_3D'].iloc[j] = 1
                    target['GT_stage_1'].iloc[j] = 1
                    target['GT_stage_2'].iloc[j] = 1
                    target['GT_stage_3'].iloc[j] = 1
                    target['GT_presence_24'].iloc[j] = 1

            if RRT_18> 0 :
                    target['GT_presence_24'].iloc[j] = 1
                    target['GT_presence_18'].iloc[j] = 1

            if RRT_12 > 0 :
                    target['GT_presence_24'].iloc[j] = 1
                    target['GT_presence_18'].iloc[j] = 1
                    target['GT_presence_12'].iloc[j] = 1

            if RRT_6 > 0 :
                    target['GT_presence_24'].iloc[j] = 1
                    target['GT_presence_18'].iloc[j] = 1
                    target['GT_presence_12'].iloc[j] = 1
                    target['GT_presence_6'].iloc[j] = 1
                    
            if (area_24.sum() > 0) :
                target['GT_presence_24'].iloc[j] = 1
                
                if area_24_max == 1 :
                    target['GT_stage_1'].iloc[j] = 1

                elif area_24_max == 2 :
                    target['GT_stage_1'].iloc[j] = 1
                    target['GT_stage_2'].iloc[j] = 1

                elif area_24_max == 3 :
                    target['GT_stage_1'].iloc[j] = 1
                    target['GT_stage_2'].iloc[j] = 1
                    target['GT_stage_3'].iloc[j] = 1

            if (area_18 > 0) :
                target['GT_presence_18'].iloc[j] = 1

            if (area_12 > 0) :
                target['GT_presence_12'].iloc[j] = 1

            if (area_6 > 0) :
                target['GT_presence_6'].iloc[j] = 1

        return target
    
    tqdm.pandas(desc="Processing groups")
    df = df.groupby('stay_id', group_keys=False).progress_apply(lambda x : operation(x, AKI[AKI['stay_id'] == x.name], RRT[RRT['stay_id'] == x.name])).sort_values(['subject_id','stay_id','charttime']).reset_index(drop=True)

    return df

def convert_temperature(df):
    df['valuenum'] = (df['valuenum'] - 32) * 5 / 9
    df['itemid'] = 223762
    df['valueuom'] = '°C'
    return df

def Vital(df, stage, vitalsign):

    HR_items = 220045
    NI_SBP_items = 220179
    NI_DBP_items = 220180
    Art_SBP_items = 220050
    Art_DBP_items = 220051
    TP_items = 223762
    RR_items = 220210
    Art_OS_items = 220227
    NI_OS_items = 220277
    F_items = 223761

    df_FA = df[df['itemid'] == F_items].copy()
    df_FA = convert_temperature(df_FA)

    df_HR = df[df['itemid'] == HR_items].copy()
    df_NI_SBP = df[df['itemid'] == NI_SBP_items].copy()
    df_NI_DBP = df[df['itemid'] ==NI_DBP_items].copy()
    df_Art_SBP = df[df['itemid'] == Art_SBP_items].copy()
    df_Art_DBP = df[df['itemid'] == Art_DBP_items].copy()
    df_TP = df[df['itemid'] == TP_items].copy()
    df_RR = df[df['itemid'] == RR_items].copy()
    df_Art_OS = df[df['itemid'] == Art_OS_items].copy()
    df_NI_OS = df[df['itemid'] == NI_OS_items].copy()

    df_TP = pd.merge(df_TP, df_FA, how='outer')

    df_HR['valueuom'] = 'bpm'
    df_NI_SBP['valueuom'] = 'mmHg'
    df_NI_DBP['valueuom'] = 'mmHg'
    df_Art_SBP['valueuom'] = 'mmHg'
    df_Art_DBP['valueuom'] = 'mmHg'
    df_TP['valueuom'] = '°C'
    df_RR['valueuom'] = 'insp/min'
    df_Art_OS['valueuom'] = '%'
    df_NI_OS['valueuom'] = '%'

    # HR
    df_HR = df_HR[(df_HR['valuenum'] <= 300) & (df_HR['valuenum'] >= 0)]
    vitalsign.loc[(vitalsign['heartrate'] > 300) & (vitalsign['heartrate'] < 0),'heartrate'] = np.nan

    # SBP
    df_NI_SBP = df_NI_SBP[(df_NI_SBP['valuenum'] < 300) & (df_NI_SBP['valuenum'] >= 0)]
    df_Art_SBP = df_Art_SBP[(df_Art_SBP['valuenum'] < 300) & (df_Art_SBP['valuenum'] >= 0)]
    vitalsign.loc[(vitalsign['sbp'] >= 300) & (vitalsign['sbp'] < 0),'sbp'] = np.nan

    # DBP
    df_NI_DBP = df_NI_DBP[(df_NI_DBP['valuenum'] < 175) & (df_NI_DBP['valuenum'] > 10)]
    df_Art_DBP = df_Art_DBP[(df_Art_DBP['valuenum'] < 175) & (df_Art_DBP['valuenum'] > 10)]
    vitalsign.loc[(vitalsign['dbp'] >= 175) & (vitalsign['dbp'] <= 10),'dbp'] = np.nan

    # TP
    df_TP = df_TP[(df_TP['valuenum'] <= 43) & (df_TP['valuenum'] > 32)]
    vitalsign.loc[(vitalsign['temperature'] > 43) & (vitalsign['temperature'] <= 32),'temperature'] = np.nan

    # RR
    df_RR = df_RR[(df_RR['valuenum'] < 60) & (df_RR['valuenum'] >= 0)]
    vitalsign.loc[(vitalsign['resprate'] >= 60) & (vitalsign['resprate'] < 0),'resprate'] = np.nan

    # OS
    df_NI_OS = df_NI_OS[(df_NI_OS['valuenum'] <= 100) & (df_NI_OS['valuenum'] >= 0)]
    df_Art_OS = df_Art_OS[(df_Art_OS['valuenum'] <= 100) & (df_Art_OS['valuenum'] >= 0)]
    vitalsign.loc[(vitalsign['o2sat'] > 100) & (vitalsign['o2sat'] < 0),'o2sat'] = np.nan

    dataframes = [stage, df_TP, df_HR, df_NI_SBP, df_NI_DBP, df_Art_SBP, df_Art_DBP, df_RR, df_Art_OS, df_NI_OS, vitalsign]

    for df in dataframes:
        df['charttime'] = pd.to_datetime(df['charttime'])

    stage['timedelta'] = pd.to_timedelta(stage['timedelta'])

    stay_id = stage['stay_id'].drop_duplicates()
    subject_id = stage['subject_id'].drop_duplicates()

    df_TP = pd.merge(df_TP,stay_id,on='stay_id',how='inner')
    df_HR = pd.merge(df_HR,stay_id,on='stay_id',how='inner')
    df_NI_SBP = pd.merge(df_NI_SBP,stay_id,on='stay_id',how='inner')
    df_NI_DBP = pd.merge(df_NI_DBP,stay_id,on='stay_id',how='inner')
    df_Art_SBP = pd.merge(df_Art_SBP,stay_id,on='stay_id',how='inner')
    df_Art_DBP = pd.merge(df_Art_DBP,stay_id,on='stay_id',how='inner')
    df_RR = pd.merge(df_RR,stay_id,on='stay_id',how='inner')
    df_Art_OS = pd.merge(df_Art_OS,stay_id,on='stay_id',how='inner')
    df_NI_OS = pd.merge(df_NI_OS,stay_id,on='stay_id',how='inner')
    vitalsign = pd.merge(vitalsign,subject_id,on='subject_id',how='inner')

    df_Art_SBP['Art'] = 1
    df_NI_SBP['Art'] = 0
    df_SBP = pd.concat([df_Art_SBP,df_NI_SBP]).sort_values(['subject_id','charttime'])
    df_SBP = df_SBP.groupby('stay_id',group_keys=False).apply(lambda x : gap(x,'sbp')).reset_index(drop=True)
    df_Art_SBP = df_SBP[df_SBP['Art'] == 1]
    df_NI_SBP = df_SBP[df_SBP['Art'] == 0]

    df_Art_DBP['Art'] = 1
    df_NI_DBP['Art'] = 0
    df_DBP = pd.concat([df_Art_DBP,df_NI_DBP]).sort_values(['subject_id','charttime'])
    df_DBP = df_DBP.groupby('stay_id',group_keys=False).apply(lambda x : gap(x,'dbp')).reset_index(drop=True)
    df_Art_DBP = df_DBP[df_DBP['Art'] == 1]
    df_NI_DBP = df_DBP[df_DBP['Art'] == 0]

    df_Art_OS['Art'] = 1
    df_NI_OS['Art'] = 0
    df_OS = pd.concat([df_Art_OS,df_NI_OS]).sort_values(['subject_id','charttime'])
    df_OS = df_OS.groupby('stay_id',group_keys=False).apply(lambda x : gap(x,'o2sat')).reset_index(drop=True)
    df_Art_OS = df_OS[df_OS['Art'] == 1]
    df_NI_OS = df_OS[df_OS['Art'] == 0]

    df_TP['valuenum'] = round(df_TP['valuenum'],1)
    df_TP = df_TP.groupby('stay_id',group_keys=False).apply(lambda x : gap(x,'temperature')).reset_index(drop=True)

    df_HR = df_HR.groupby('stay_id',group_keys=False).apply(lambda x : gap(x,'heartrate')).reset_index(drop=True)

    df_RR = df_RR.groupby('stay_id',group_keys=False).apply(lambda x : gap(x,'resprate')).reset_index(drop=True)

    vitalsign_TP = vitalsign[vitalsign['temperature'].notna()][['subject_id','charttime','temperature']].sort_values(['subject_id','charttime'])
    vitalsign_TP = vitalsign_TP.groupby('subject_id',group_keys=False).apply(lambda x : gap(x,'temperature')).reset_index(drop=True)

    vitalsign_HR = vitalsign[vitalsign['heartrate'].notna()][['subject_id','charttime','heartrate']].sort_values(['subject_id','charttime'])
    vitalsign_HR = vitalsign_HR.groupby('subject_id',group_keys=False).apply(lambda x : gap(x,'heartrate')).reset_index(drop=True)

    vitalsign_RR = vitalsign[vitalsign['resprate'].notna()][['subject_id','charttime','resprate']].sort_values(['subject_id','charttime'])
    vitalsign_RR = vitalsign_RR.groupby('subject_id',group_keys=False).apply(lambda x : gap(x,'resprate')).reset_index(drop=True)

    vitalsign_SBP = vitalsign[vitalsign['sbp'].notna()][['subject_id','charttime','sbp']].sort_values(['subject_id','charttime'])
    vitalsign_SBP = vitalsign_SBP.groupby('subject_id',group_keys=False).apply(lambda x : gap(x,'sbp')).reset_index(drop=True)
    
    vitalsign_DBP = vitalsign[vitalsign['dbp'].notna()][['subject_id','charttime','dbp']].sort_values(['subject_id','charttime'])
    vitalsign_DBP = vitalsign_DBP.groupby('subject_id',group_keys=False).apply(lambda x : gap(x,'dbp')).reset_index(drop=True)

    vitalsign_OS = vitalsign[vitalsign['o2sat'].notna()][['subject_id','charttime','o2sat']].sort_values(['subject_id','charttime'])
    vitalsign_OS = vitalsign_OS.groupby('subject_id',group_keys=False).apply(lambda x : gap(x,'o2sat')).reset_index(drop=True)

    return (
        df_HR, df_NI_SBP, df_NI_DBP, df_Art_SBP, df_Art_DBP, df_TP, df_RR, df_Art_OS, df_NI_OS, df_SBP, df_DBP, df_OS, vitalsign, stage, vitalsign_TP , vitalsign_HR , vitalsign_RR ,vitalsign_SBP , vitalsign_DBP, vitalsign_OS 
    )

def Mapping(df,data,label):

    df[label] = np.nan
    df[label + '_diff'] = np.nan

    def operation(target,target_data,label):
  
        for i in range(len(target)):

            target_lower = target['charttime'].iloc[i]
            target_upper = target_lower  + timedelta(hours=6)
            target_value = target_data[(target_data['charttime'] < target_upper) & (target_data['charttime'] > target_lower)].sort_values(['charttime'])

            if not target_value.empty: 
                target[label].iloc[i] = target_value['valuenum'].iloc[-1]
                target[label + '_diff'].iloc[i] = target_value[label + '_diff'].iloc[-1]

        return target
    
    tqdm.pandas(desc="Processing groups")
    df = df.groupby('stay_id',group_keys=False).progress_apply(lambda x : operation(x,data[data['stay_id'] == x.name], label)).reset_index(drop=True)
    df['timedelta'] = pd.to_timedelta(df['timedelta'])
    print(df[label].isnull().sum())

    return df

def Presence_Mapping(df,data,label):

    df[label] = np.nan

    def operation(target,target_data,label):
  
        for i in range(len(target)):

            target_lower = target['charttime'].iloc[i]
            target_upper = target_lower  + timedelta(hours=6)
            target_value = target_data[(target_data['charttime'] < target_upper) & (target_data['charttime'] > target_lower)].sort_values(['charttime'])

            if not target_value.empty: 
                target[label].iloc[i] = target_value['valuenum'].iloc[-1]

        return target
    
    tqdm.pandas(desc="Processing groups")
    df = df.groupby('stay_id',group_keys=False).progress_apply(lambda x : operation(x,data[data['stay_id'] == x.name], label)).reset_index(drop=True)
    df['timedelta'] = pd.to_timedelta(df['timedelta'])
    print(df[label].isnull().sum())

    return df
    
def gap(target,label):

    if label in target.columns:
        if label != 'charttime':
            target[label] = round(target[label],1)
            target[label + '_diff']  = target[label].diff()
        else :
            target[label + '_diff']  = target[label].diff()
        
    else :
        target['valuenum'] = round(target['valuenum'],1)
        target[label + '_diff']  = target['valuenum'].diff()

    return target

def MAX_AKI(df):
    df_pre = []
    stay_id = df['stay_id'].drop_duplicates()

    for i in tqdm(stay_id):

        target = df[df['stay_id'] == i]
        target['max_stage'] = target[['SCr_stage', 'Urine_stage']].max(axis=1)
        
        df_pre.append(target)
    df_pre = pd.concat(df_pre)
    return df_pre


def step_ROC(dataloader):

    y_true = []
    y_scores = []
    
    for batch in dataloader.dataset:
        
        X, Y = batch.tensors

        X = X.cpu().detach().numpy().tolist()
        y_scores.extend(X[0])

        Y = Y.cpu().detach().numpy().tolist()
        y_true.extend(Y[0])

    return y_true, y_scores

def step_ROC(dataloader):

    y_true = []
    y_scores = []
    
    for batch in dataloader.dataset:
        
        X, Y = batch.tensors

        X = X.cpu().detach().numpy().tolist()
        y_scores.extend(X[0])

        Y = Y.cpu().detach().numpy().tolist()
        y_true.extend(Y[0])

    return y_true, y_scores

def AKI_step_ROC(dataloader):

    sum = 0
    y_true = []
    y_scores = []
    
    for batch in dataloader.dataset:
        X, Y = batch.tensors
        if torch.sum(Y) > 0:
            Y = Y.cpu().detach().numpy().tolist()
            y_true.extend(Y[0])

            X = X.cpu().detach().numpy().tolist()
            y_scores.extend(X[0])

            sum+= 1
    
    print(sum)

    return y_true, y_scores

def AKI_first_ROC(dataloader):

    sum = 0
    y_true = []
    y_scores = []
    values = float(1)

    for batch in dataloader.dataset:
        X, Y = batch.tensors
        if torch.sum(Y) > 0:
            Y = Y.cpu().detach().numpy().tolist()
            index = Y[0].index(values)
            y_true.extend(Y[0][:index+1])

            X = X.cpu().detach().numpy().tolist()
            y_scores.extend(X[0][:index+1])

            sum+= 1

    print(sum)
    
    return y_true, y_scores

def AUROC(y_true,y_scores):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    # Compute AUROC
    auroc = roc_auc_score(y_true, y_scores)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (AUROC: {:.3f})'.format(auroc))
    #plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()
    return fpr, tpr, thresholds

def AUPRC(y_true,y_scores):
    # Compute precision, recall, and threshold values
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # Compute AUPRC
    auprc = auc(recall, precision)

    # Plot Precision-Recall curve
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (AUPRC: {:.3f})'.format(auprc))
    plt.plot([0, 1], [1, 0], 'k--')  # Random guessing line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower left')
    plt.show()
    return recall, precision, thresholds

def calculate_confusion_matrix(y_true, y_scores, threshold):
    # 이진 분류를 위한 임계값(threshold) 기준으로 예측 결과 계산
    y_pred = [1 if score >= threshold else 0 for score in y_scores]

    # True Positive(TP), False Positive(FP), True Negative(TN), False Negative(FN) 초기화
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # Confusion Matrix 계산
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == 1 and pred_label == 1:
            TP += 1
        elif true_label == 0 and pred_label == 1:
            FP += 1
        elif true_label == 0 and pred_label == 0:
            TN += 1
        elif true_label == 1 and pred_label == 0:
            FN += 1

    evaluation(TP,FP,TN,FN)
    
    return TP, FP, TN, FN

def evaluation(TP,FP,TN,FN):
    Sensitivity = TP / (TP+FN) # Recall
    Specitivity = TN / (FP+TN) 
    Accuaracy = (TP+TN) / (TP+TN+FP+FN)
    Precision = TP / (TP+FP)
    F1 = (2 * Precision * Sensitivity) / (Precision+Sensitivity)

    print('Accuracy :',round(Accuaracy,3)*100,'%')
    print('Precision :',round(Precision,3)*100,'%')
    print('Sensitivity :',round(Sensitivity,3)*100,'%')
    print('Specitivity :',round(Specitivity,3)*100,'%')
    print('F1 score :',round(F1,3))

def ICD(df, icd_list):
    df_icd = df[df['icd_code'].str.startswith(tuple(icd_list))]
    return df_icd

def check_ICD(stage,df9,df10,label):

    check = pd.concat([df9,df10])
    check.drop_duplicates(subset='hadm_id',inplace=True)
    check[label] = 1
    check = check[['hadm_id',label]]
    check = pd.merge(stage,check,on='hadm_id',how='left')
    check[label] = check[label].fillna(0)

    print(check[label].isnull().sum())
    print(check[label].value_counts())
    print(check.drop_duplicates(subset='hadm_id')[label].value_counts())

    return check 

def length(df):
    df_pre = []

    stay_id = df['stay_id'].drop_duplicates()
    
    for i in tqdm(stay_id):
        target = df[df['stay_id'] == i]
        length = len(target)
        target['length'] = length
        
        df_pre.append(target)

    df_pre = pd.concat(df_pre)
    return df_pre

def check_label(df):
    n = len(df.columns)
    for i in range(1,n):
        print(df.iloc[:,i].value_counts(normalize=True) * 100)
        print(df.iloc[:,i].value_counts())

def iterative_split(df, test_size, stratify_columns):
    """Custom iterative train test split which
    'maintains balanced representation with respect
    to order-th label combinations.'

    From https://madewithml.com/courses/mlops/splitting/#stratified-split
    """
    # One-hot encode the stratify columns and concatenate them
    one_hot_cols = [pd.get_dummies(df[col]) for col in stratify_columns]
    one_hot_cols = pd.concat(one_hot_cols, axis=1).to_numpy()
    stratifier = IterativeStratification(
        n_splits=2, order=len(stratify_columns), sample_distribution_per_fold=[test_size, 1-test_size])
    train_indices, test_indices = next(stratifier.split(df.to_numpy(), one_hot_cols))
    # Return the train and test set dataframes
    train, test = df.iloc[train_indices], df.iloc[test_indices]
    return train, test

def Remaing_LOS(df):
    df_pre = []
    stay_id = df['stay_id'].drop_duplicates()

    for i in tqdm(stay_id):
        target = df[df['stay_id'] == i]

        target = target.iloc[:-4]

        #for j in range(len(target)):
            #if target['stage'].iloc[j] > 0 :
                #target = target.iloc[:j+29]
                #break

        df_pre.append(target)
    df_pre = pd.concat(df_pre)
    return df_pre

def MAX_AKI(df,Urine_icu,SCr_baseline):

    df_pre = []
    stay_id = df['stay_id'].drop_duplicates()

    for i in tqdm(stay_id):

        target = df[df['stay_id'] == i]
        target_Urine = Urine_icu[Urine_icu['stay_id'] == i]
        target_SCr = SCr_baseline[SCr_baseline['stay_id'] == i]

        target['max_stage'] = max(max(target_Urine['Urine_stage']),max(target_SCr['SCr_stage']))
        if target['RRT'].sum() > 0 :
            target['max_stage'] = 3
        
        df_pre.append(target)
    df_pre = pd.concat(df_pre)
    return df_pre

def Result(dataloader):

    y_true, y_scores = step_ROC(dataloader)
    fpr, tpr, AUROC_thresholds = AUROC(y_true,y_scores)
    recall, precision, AUPRC_thresholds = AUPRC(y_true,y_scores)

    for i in np.arange(0,1,0.05):
        print(i)
        y_pred = [1 if score >= i else 0 for score in y_scores]

        # 정확도 계산
        accuracy = accuracy_score(y_true, y_pred)
        print(f'Accuracy: {accuracy}')

        # 정밀도 계산
        precision = precision_score(y_true, y_pred)
        print(f'Precision: {precision}')

        # 재현율 계산
        recall = recall_score(y_true, y_pred)
        print(f'Recall: {recall}')

        # F1 점수 계산
        f1 = f1_score(y_true, y_pred)
        print(f'F1 Score: {f1}')

        # 오차 행렬(confusion matrix) 계산
        #cm = confusion_matrix(y_true, y_pred)
        #print('Confusion Matrix:')
        #print(cm)
    
    y_true, y_scores = AKI_step_ROC(dataloader)
    fpr, tpr, AUROC_thresholds = AUROC(y_true,y_scores)
    recall, precision, AUPRC_thresholds = AUPRC(y_true,y_scores)

    for i in np.arange(0,1,0.05):
        print(i)
        y_pred = [1 if score >= i else 0 for score in y_scores]

        # 정확도 계산
        accuracy = accuracy_score(y_true, y_pred)
        print(f'Accuracy: {accuracy}')

        # 정밀도 계산
        precision = precision_score(y_true, y_pred)
        print(f'Precision: {precision}')

        # 재현율 계산
        recall = recall_score(y_true, y_pred)
        print(f'Recall: {recall}')

        # F1 점수 계산
        f1 = f1_score(y_true, y_pred)
        print(f'F1 Score: {f1}')

        # 오차 행렬(confusion matrix) 계산
        #cm = confusion_matrix(y_true, y_pred)
        #print('Confusion Matrix:')
        #print(cm)

    y_true, y_scores = AKI_first_ROC(dataloader)
    fpr, tpr, AUROC_thresholds = AUROC(y_true,y_scores)
    recall, precision, AUPRC_thresholds = AUPRC(y_true,y_scores)

    for i in np.arange(0,1,0.05):
        print(i)
        y_pred = [1 if score >= i else 0 for score in y_scores]

        # 정확도 계산
        accuracy = accuracy_score(y_true, y_pred)
        print(f'Accuracy: {accuracy}')

        # 정밀도 계산
        precision = precision_score(y_true, y_pred)
        print(f'Precision: {precision}')

        # 재현율 계산
        recall = recall_score(y_true, y_pred)
        print(f'Recall: {recall}')

        # F1 점수 계산
        f1 = f1_score(y_true, y_pred)
        print(f'F1 Score: {f1}')

        # 오차 행렬(confusion matrix) 계산
        #cm = confusion_matrix(y_true, y_pred)
        #print('Confusion Matrix:')
        #print(cm)

def find_indices_with_condition(lst, condition):
        return [index for index, element in enumerate(lst) if condition(element)]

def tomasev_AU(main_dataloader,sub_dataloader):

    y_true_main, y_scores_main = step_ROC(main_dataloader)
    y_true_sub, y_scores_sub= step_ROC(sub_dataloader)

    condition = lambda x: x == 0

    # 조건을 만족하는 원소의 인덱스 찾기
    indices_main = find_indices_with_condition(y_true_main, condition)
    y_true_main_target = [y_true_main[index] for index in indices_main]
    y_scores_main_target = [y_scores_main[index] for index in indices_main]

    condition = lambda x: x == 1

    # 조건을 만족하는 원소의 인덱스 찾기
    indices_sub = find_indices_with_condition(y_true_sub, condition)
    y_true_sub_target = [y_true_main[index] for index in indices_sub]
    y_scores_sub_target = [y_scores_main[index] for index in indices_sub]

    y_true = y_true_main_target + y_true_sub_target
    y_scores = y_scores_main_target + y_scores_sub_target

    fpr, tpr, AUROC_thresholds = AUROC(y_true,y_scores)
    recall, precision, AUPRC_thresholds = AUPRC(y_true,y_scores)

def get_unique_common_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    common_elements = set1.intersection(set2)
    unique_common_elements = list(common_elements)
    return unique_common_elements

def xavier_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)

