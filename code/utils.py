import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from datetime import timedelta
import os

def baseline_SCr(chartevents_SCr, icustays, labevents) :
    subject_id_SCr = chartevents_SCr['subject_id'].unique()
    stay_id_SCr = chartevents_SCr['stay_id'].unique()

    icustays_SCr = icustays[icustays['stay_id'].isin(stay_id_SCr)]
    icustays_SCr['intime'] = pd.to_datetime(icustays_SCr['intime'])

    labevents_SCr = labevents[labevents['itemid'].isin([
    50912, # Creatinine, Blood, Chemistry
    52024, # Creatinine, Whole Blood, Blood, Chemistry
    52546  # Creatinine, Blood, Chemistry
    ])]

    labevents_SCr = labevents_SCr[['subject_id','hadm_id','charttime','valuenum']]
    labevents_SCr['charttime'] = pd.to_datetime(labevents_SCr['charttime'])

    baseline_SCr_list = []

    for i in tqdm(subject_id_SCr) : 
        tmp_hosp_labevents_sCr = labevents_SCr[labevents_SCr['subject_id'] == i]
        for j in icustays_SCr[icustays_SCr['subject_id'] == i].stay_id : 
            tmp_icu_icustays_intime = icustays_SCr[icustays_SCr['stay_id'] == j].iloc[0]
            tmp_icu_icustays_intime_7days = tmp_icu_icustays_intime['intime'] - timedelta(days=7)
            tmp_icu_icustays_intime_1yr = tmp_icu_icustays_intime['intime'] - timedelta(days=365)

            baseline_sCr = 0
            
            tmp_hosp_labevents_sCr_list = tmp_hosp_labevents_sCr[tmp_hosp_labevents_sCr['charttime'] < tmp_icu_icustays_intime['intime']]
            tmp_hosp_labevents_sCr_list = tmp_hosp_labevents_sCr_list[tmp_hosp_labevents_sCr_list['charttime'] > tmp_icu_icustays_intime_7days]

            if not tmp_hosp_labevents_sCr_list.empty :
                baseline_sCr = tmp_hosp_labevents_sCr_list['valuenum'].min()

            else : 
                tmp_hosp_labevents_sCr_list = tmp_hosp_labevents_sCr_list[tmp_hosp_labevents_sCr_list['charttime'] < tmp_icu_icustays_intime_7days]
                tmp_hosp_labevents_sCr_list = tmp_hosp_labevents_sCr_list[tmp_hosp_labevents_sCr_list['charttime'] > tmp_icu_icustays_intime_1yr]
                baseline_sCr = tmp_hosp_labevents_sCr_list['valuenum'].median()

            baseline_SCr_list.append([i, j, baseline_sCr])

    df_baseline_sCr_list = pd.DataFrame(baseline_SCr_list, columns=['subject_id','stay_id','valuenum'])
    return df_baseline_sCr_list

def resample_ffill(resample_):
    resample_['valuenum'] = resample_.groupby('stay_id')['valuenum'].ffill()
    return resample_

def resample_bfill(resample_):
    resample_['valuenum'] = resample_.groupby('stay_id')['valuenum'].bfill()
    return resample_    

def resample_fill(resample_):
    resample_['valuenum'] = resample_.groupby('stay_id')['valuenum'].ffill()
    resample_['valuenum'] = resample_.groupby('stay_id')['valuenum'].bfill()
    return resample_