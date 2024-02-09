import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import timedelta
import os, math

def columns_to_datetime(df):
    if 'charttime' in df.columns:
        df['charttime'] = pd.to_datetime(df['charttime'])
    if 'intime' in df.columns:
        df['intime'] = pd.to_datetime(df['intime'])
    if 'outtime' in df.columns:
        df['outtime'] = pd.to_datetime(df['outtime'])
    if 'starttime' in df.columns:
        df['starttime'] = pd.to_datetime(df['starttime'])
    if 'endtime' in df.columns:
        df['endtime'] = pd.to_datetime(df['endtime'])
    if 'chartdate' in df.columns:
        df['chartdate'] = pd.to_datetime(df['chartdate'])
    return df

# Demographic
def cal_gender(patients):
    df = patients[['subject_id','gender']]
    df.loc[df.gender == 'F' , 'gender'] = 1
    df.loc[df.gender == 'M' , 'gender'] = 0
    return df

def cal_age(icustays,patients):
    icustays_intime = icustays[['subject_id','hadm_id','stay_id','intime']]
    patients_age = patients[['subject_id','anchor_age','anchor_year']]

    rt = pd.merge(icustays_intime, patients_age, on = 'subject_id', how = 'left')

    rt['anchor_age_delta'] = pd.to_timedelta(rt['anchor_age']*365.25, unit='D')
    rt['anchor_year'] = pd.to_datetime(rt['anchor_year'],format="%Y")

    rt['delta'] = rt['intime'] - rt['anchor_year']
    rt['age'] = ((rt['anchor_age_delta'] + rt['delta'])/365.25).dt.days

    rt = rt[['subject_id','hadm_id','stay_id','age']]
    return rt

def cal_race(admissions):
    rt = admissions[['subject_id','race']].copy()
    rt['race'].replace(['ASIAN - ASIAN INDIAN', 'ASIAN - CHINESE','ASIAN - KOREAN', 'ASIAN - SOUTH EAST ASIAN'],'ASIAN',inplace=True)
    rt['race'].replace(['BLACK/AFRICAN AMERICAN','BLACK/AFRICAN','BLACK/CAPE VERDEAN','BLACK/CARIBBEAN ISLAND'],'BLACK',inplace=True)
    rt['race'].replace(['HISPANIC/LATINO - CENTRAL AMERICAN','HISPANIC/LATINO - COLUMBIAN','HISPANIC/LATINO - CUBAN','HISPANIC/LATINO - DOMINICAN','HISPANIC/LATINO - GUATEMALAN','HISPANIC/LATINO - HONDURAN','HISPANIC/LATINO - MEXICAN',
                                'HISPANIC/LATINO - PUERTO RICAN','HISPANIC/LATINO - SALVADORAN','PORTUGUESE','SOUTH AMERICAN'],'HISPANIC OR LATINO',inplace=True)
    rt['race'].replace(['NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER','AMERICAN INDIAN/ALASKA NATIVE'],'OTHER',inplace=True)
    rt['race'].replace(['UNABLE TO OBTAIN','PATIENT DECLINED TO ANSWER'],'UNKNOWN',inplace=True)
    rt['race'].replace(['WHITE - BRAZILIAN','WHITE - EASTERN EUROPEAN','WHITE - OTHER EUROPEAN','WHITE - RUSSIAN'],'WHITE',inplace=True)

    rt = rt.drop_duplicates()

    multiple = rt.subject_id.value_counts().loc[lambda x : x > 1].to_frame()
    multiple.reset_index(inplace = True)
    multiple = multiple.subject_id.unique()

    hosp_race_multiple = rt[rt['subject_id'].isin(multiple)]

    sol = []
    multi = []
    for i in hosp_race_multiple.subject_id.unique() :
        tmp = hosp_race_multiple[hosp_race_multiple['subject_id'] == i]
        if (tmp['race'] == 'UNKNOWN').any() :
            tmp = tmp[tmp['race'] != 'UNKNOWN']
        if len(tmp) <2 :
            sol.append(tmp)
        else : 
            multi.append(tmp)
    sol = pd.concat(sol)
    multi = pd.concat(multi)

    multi['race'] = 'MULTIPLE RACE/ETHNICITY'
    multi.drop_duplicates(inplace=True)

    rt = rt[~rt['subject_id'].isin(sol.subject_id.unique())]
    rt = rt[~rt['subject_id'].isin(multi.subject_id.unique())]

    rt = pd.concat([rt,sol,multi])
    rt[['WHITE','BLACK','HISPANIC OR LATINO','ASIAN','MULTIPLE RACE','OTHER','UNKNOWN']] = 0
    rt.loc[rt['race']=='WHITE','WHITE'] = 1
    rt.loc[rt['race']=='BLACK','BLACK'] = 1
    rt.loc[rt['race']=='HISPANIC OR LATINO','HISPANIC OR LATINO'] = 1
    rt.loc[rt['race']=='ASIAN','ASIAN'] = 1
    rt.loc[rt['race']=='MULTIPLE RACE','MULTIPLE RACE'] = 1
    rt.loc[rt['race']=='OTHER','OTHER'] = 1
    rt.loc[rt['race']=='UNKNOWN','UNKNOWN'] = 1
    rt.drop('race',axis=1,inplace=True)
    return rt

def cal_height(icustays,chartevents,omr):
    # Height(inch)
    chartevents_height_inch = chartevents[chartevents['itemid'].isin([226707])]
    chartevents_height_inch = chartevents_height_inch[['subject_id','hadm_id','stay_id','charttime','itemid','valuenum']]
    chartevents_height_inch['valuenum'] = (chartevents_height_inch['valuenum']*2.54).round(1)

    # Height(cm)
    chartevents_height_cm = chartevents[chartevents['itemid'].isin([226730])]
    chartevents_height_cm = chartevents_height_cm[['subject_id','hadm_id','stay_id','charttime','itemid','valuenum']]

    chartevents_height = pd.concat([chartevents_height_cm,chartevents_height_inch])
    chartevents_height['charttime'] = pd.to_datetime(chartevents_height['charttime'])
    chartevents_height.sort_values(by=['subject_id','hadm_id','stay_id','charttime'],inplace=True)
    chartevents_height = chartevents_height[(chartevents_height['valuenum']>0)&(chartevents_height['valuenum']<240)]

    omr_height = omr[omr['result_name'].isin(['Height (Inches)', 'Height'])]
    omr_height['result_value'] = omr_height['result_value'].astype('float64')
    omr_height['result_value'] = (omr_height['result_value']*2.54).round(1)
    omr_height = omr_height[['subject_id','chartdate','result_value']]
    omr_height['chartdate'] = pd.to_datetime(omr_height['chartdate'])

    rt = []
    for subject_id in tqdm(icustays.subject_id.unique()):
        tmp_omr_weight = omr_height[omr_height['subject_id']==subject_id]
        for stay_id in icustays[icustays['subject_id']==subject_id].stay_id.unique():
            hadm_id = icustays[icustays['stay_id']==stay_id].hadm_id.values[0]
            icustays_intime = icustays[icustays['stay_id'] == stay_id]['intime'].iloc[0]

            admission_weight = 0

            chartevents_weight_list = chartevents_height[chartevents_height['stay_id']==stay_id]
            chartevents_weight_list = chartevents_weight_list[chartevents_weight_list['charttime'] >= icustays_intime]       
            omr_weight_list = tmp_omr_weight[tmp_omr_weight['chartdate'] <= icustays_intime]
            
            if not chartevents_weight_list.empty:
                admission_weight = chartevents_weight_list['valuenum'].iloc[0]
            elif not omr_weight_list.empty:
                admission_weight = omr_weight_list['result_value'].iloc[-1]
            else : 
                admission_weight = np.nan

            rt.append([subject_id,hadm_id,stay_id,admission_weight])
    rt = pd.DataFrame(rt, columns = ['subject_id','hadm_id','stay_id','height'])
    return rt



def cal_weight(icustays,chartevents,inputevents,omr):
    #Admission Weight(Kg)
    chartevents_weight_kg = chartevents[chartevents['itemid'].isin([226512])]
    chartevents_weight_kg = chartevents_weight_kg[['subject_id','hadm_id','stay_id','charttime','valuenum']]

    #Admission Weight(lbs.)
    chartevents_weight_lbs = chartevents[chartevents['itemid'].isin([226531])]
    chartevents_weight_lbs = chartevents_weight_lbs[['subject_id','hadm_id','stay_id','charttime','valuenum']]

    chartevents_weight_lbs['valuenum'] = (chartevents_weight_lbs['valuenum'] * 0.453592).round(1)
    chartevents_weight = pd.concat([chartevents_weight_kg,chartevents_weight_lbs])
    chartevents_weight.sort_values(by=['subject_id','hadm_id','stay_id','charttime'],inplace=True)
    chartevents_weight = chartevents_weight[(chartevents_weight['valuenum']>0)&(chartevents_weight['valuenum']<250)]

    omr_weight = omr[omr['result_name'].isin(['Weight','Weight (Lbs)'])]
    omr_weight['result_value'] = omr_weight['result_value'].astype('float64')
    omr_weight['result_value'] = (omr_weight['result_value']*0.453592).round(1)
    omr_weight = omr_weight[(omr_weight['result_value']>0)&(omr_weight['result_value']<250)]
    omr_weight = omr_weight[['subject_id','chartdate','result_value']]

    rt = []
    for subject_id in tqdm(icustays.subject_id.unique()):
        tmp_omr_weight = omr_weight[omr_weight['subject_id']==subject_id]
        for stay_id in icustays[icustays['subject_id']==subject_id].stay_id.unique():
            hadm_id = icustays[icustays['stay_id']==stay_id].hadm_id.values[0]
            icustays_intime = icustays[icustays['stay_id'] == stay_id]['intime'].iloc[0]

            admission_weight = 0

            chartevents_weight_list = chartevents_weight[chartevents_weight['stay_id']==stay_id]
            chartevents_weight_list = chartevents_weight_list[chartevents_weight_list['charttime'] >= icustays_intime]       
            omr_weight_list = tmp_omr_weight[tmp_omr_weight['chartdate'] <= icustays_intime]
            inputevents_list = inputevents[inputevents['stay_id']==stay_id]
            inputevents_list = inputevents_list[inputevents_list['starttime']>=icustays_intime].sort_values(by=['subject_id','hadm_id','stay_id','starttime'])

            if not chartevents_weight_list.empty:
                admission_weight = chartevents_weight_list['valuenum'].iloc[0]
            elif not omr_weight_list.empty:
                admission_weight = omr_weight_list['result_value'].iloc[-1]
            elif not inputevents_list.empty:
                admission_weight = inputevents_list['patientweight'].iloc[0]
            else : 
                admission_weight = np.nan

            rt.append([subject_id,hadm_id,stay_id,admission_weight])
    rt = pd.DataFrame(rt, columns = ['subject_id','hadm_id','stay_id','weight'])
    return rt

def cal_comorbidities(icustays,comorbidities,diagnoses_icd):
    for i,idx in enumerate(tqdm(comorbidities.abbreviation)):
        icd9 = comorbidities.iloc[i].ICD9
        icd9 = icd9.replace(" ","")
        icd9 = icd9.split(",")
        icd10 = comorbidities.iloc[i].ICD10
        icd10 = icd10.replace(" ","")
        icd10 = icd10.split(",")

        tmp_icd9 = diagnoses_icd[(diagnoses_icd['icd_code'].str.startswith(tuple(icd9)))&(diagnoses_icd['icd_version']==9)]
        tmp_icd10 = diagnoses_icd[(diagnoses_icd['icd_code'].str.startswith(tuple(icd10)))&(diagnoses_icd['icd_version']==10)]

        tmp = pd.concat([tmp_icd9,tmp_icd10])
        tmp[idx] = 1
        tmp = tmp[['subject_id','hadm_id',idx]]
        
        if i == 0 :
            rt = pd.merge(icustays,tmp, on=['subject_id','hadm_id'],how='left')
        else : 
            rt = pd.merge(rt,tmp, on=['subject_id','hadm_id'],how='left')
    rt = rt.fillna(0)
    rt = rt[['subject_id','hadm_id','stay_id']+list(comorbidities.abbreviation)]
    rt.drop_duplicates(inplace=True)
    return rt

def cal_BP(resample_ABPs, resample_NBPs, isSBP):
    # stay_id와 charttime을 기준으로 두 DataFrame을 병합
    if isSBP : ART, NI, BP=  'ABPs', 'NBPs', 'SBP'
    else : ART, NI, BP=  'ABPd', 'NBPd', 'DBP'
    
    merged_df = resample_ABPs.merge(resample_NBPs, on=['subject_id','hadm_id','stay_id', 'charttime'], how='left')

    # valuenum 열에서 NaN을 대체
    merged_df[BP] = np.where(pd.isna(merged_df[ART]), merged_df[NI], merged_df[ART])

    # 원래의 열만 남기고 반환
    return merged_df[['subject_id','hadm_id','stay_id', 'charttime', BP]]

def cal_MAP(resample_DBP,resample_SBP):
    resample_MAP = pd.merge(resample_DBP,resample_SBP,on=['subject_id','hadm_id','stay_id','charttime'],how='left')
    resample_MAP['MAP'] = (resample_MAP['SBP'] + 2*resample_MAP['DBP'])/3
    resample_MAP = resample_MAP[['subject_id','hadm_id','stay_id','charttime','MAP']]
    return resample_MAP


def cal_baseline_SCr(labevents,icustays,patients_gender,icustays_age,admissions_race) :
    admissions_black = admissions_race[['subject_id','BLACK']]

    labevents_SCr = labevents[labevents['itemid'].isin([
    50912, # Creatinine, Blood, Chemistry
    52024, # Creatinine, Whole Blood, Blood, Chemistry
    52546  # Creatinine, Blood, Chemistry
    ])]

    labevents_SCr = labevents_SCr[['subject_id','hadm_id','charttime','valuenum']]
    labevents_SCr['charttime'] = pd.to_datetime(labevents_SCr['charttime'])

    icustays['intime'] = pd.to_datetime(icustays['intime'])

    rt = []

    for subject_id in tqdm(icustays.subject_id.unique()) : 
        tmp_labevents = labevents_SCr[labevents_SCr['subject_id'] == subject_id]
        for stay_id in icustays[icustays['subject_id']==subject_id].stay_id.unique() : 
            hadm_id = icustays[icustays['stay_id'] == stay_id].hadm_id.values[0]
            icustays_intime = icustays[icustays['stay_id'] == stay_id].intime.values[0]
            icustays_intime_7days = icustays_intime - np.timedelta64(7,'D')
            icustays_intime_1yr = icustays_intime - np.timedelta64(365,'D')

            gender = patients_gender[patients_gender['subject_id']==subject_id].gender.values[0]
            age = icustays_age[icustays_age['stay_id']==stay_id].age.values[0]
            black =  admissions_black[admissions_black['subject_id']==subject_id].BLACK.values[0]

            baseline_SCr = 0
            MDRD = 0

            labevents_SCr_7days = tmp_labevents[(tmp_labevents['charttime'] < icustays_intime)&(tmp_labevents['charttime'] > icustays_intime_7days)&(~tmp_labevents['hadm_id'].isna())]['valuenum'].min()
            labevents_SCr_1yr   = tmp_labevents[(tmp_labevents['charttime'] <= icustays_intime_7days)&(tmp_labevents['charttime'] > icustays_intime_1yr)&(~tmp_labevents['hadm_id'].isna())]['valuenum'].median()

            if not math.isnan(labevents_SCr_7days):
                baseline_SCr = labevents_SCr_7days
                MDRD = 1
            elif not math.isnan(labevents_SCr_1yr):
                baseline_SCr = labevents_SCr_1yr
                MDRD = 2
            else: 
                baseline_SCr = (np.exp(5.228/1.154-0.203/1.154*np.log(age)-0.299/1.154*gender+0.192/1.154*black-np.log(75)/1.154)).round(1)
                MDRD = 3


            rt.append([subject_id, hadm_id, stay_id, baseline_SCr, MDRD])

    rt = pd.DataFrame(rt, columns=['subject_id','hadm_id','stay_id','baseline_SCr', 'MDRD'])
    return rt

def cal_uo(outputevents):
    outputevents_uo = outputevents[outputevents['itemid'].isin([
    226557, 226558, #Ureteral Stent, 요관스텐트
    226559, #Foley, 도뇨관(소변줄)
    226560, 226561, 226563, 226564, 226565, 226566, 226567, 226584, 226627, 226631, 226632
    ])]

    guirrigant_input = outputevents[outputevents['itemid'].isin([227488])]
    guirrigant_output = outputevents[outputevents['itemid'].isin([227489])]

    guirrigant = pd.merge(guirrigant_input,guirrigant_output[['subject_id','hadm_id','stay_id','charttime','value']],on=['subject_id','hadm_id','stay_id','charttime'])
    guirrigant['value'] =  guirrigant['value_y'] - guirrigant['value_x']
    guirrigant = guirrigant[guirrigant['value'] >= 0]
    guirrigant = guirrigant[['subject_id','hadm_id','stay_id','charttime','itemid','value']]

    outputevents_uo = outputevents_uo[['subject_id','hadm_id','stay_id','charttime','value']]
    outputevents_uo = pd.concat([outputevents_uo,guirrigant])

    outputevents_uo.sort_values(['subject_id','hadm_id','stay_id','charttime'],inplace=True)
    outputevents_uo.reset_index(inplace=True,drop=True)
    outputevents_uo.rename(columns={'value':'valuenum'},inplace=True)
    return outputevents_uo



def extract_labvalues(chartevents,labevents,labvalues,is_in_icu):
    lb, ub, lb_cond, ub_cond = labvalues.lb, labvalues.ub, labvalues.lb_cond, labvalues.ub_cond
    def is_itemid_list(itemid):
        if isinstance(itemid, str):
            itemid = itemid.replace(" ","")
            itemid = itemid.split(",")
        if isinstance(itemid, list):
            itemid = [int(x) for x in itemid]
        else : itemid = [int(itemid)]
        return itemid
    if is_in_icu :
        itemid = is_itemid_list(labvalues.itemid_icu)
        df = chartevents[chartevents['itemid'].isin(itemid)][['subject_id','hadm_id','stay_id','charttime','valuenum']]
    else :
        itemid = is_itemid_list(labvalues.itemid_hosp)
        df = labevents[labevents['itemid'].isin(itemid)][['subject_id','hadm_id','charttime','valuenum']]

    if lb_cond == 'ge' : 
        df = df[df['valuenum'] >= lb]
    elif lb_cond == 'gt' : 
        df = df[df['valuenum'] > lb]

    if ub_cond == 'le' : 
        df = df[df['valuenum'] <= ub]
    elif ub_cond == 'lt' : 
        df = df[df['valuenum'] < ub]

    df.sort_values(by=['subject_id','hadm_id','charttime'],ascending=True,inplace=True)
    df.reset_index(inplace=True,drop=True)

    return df



# resample
def resample_fill(resample_):
    resample_['valuenum'] = resample_.groupby('stay_id')['valuenum'].ffill()
    resample_['valuenum'] = resample_.groupby('stay_id')['valuenum'].bfill()
    return resample_


def resample_inputrates(icustays, inputevents, name):
    resampled_data = []
    # inputevents에 있는 모든 stay_id를 미리 확인
    inputevents_stay_ids = set(inputevents['stay_id'].unique())
    notrate = False
    if not 'rate' in inputevents.columns:
        inputevents['rate'] = 1
        notrate = True
    for stay_id in tqdm(icustays.stay_id.unique()):
        intime = icustays[icustays['stay_id']==stay_id].intime.values[0]
        outtime = icustays[icustays['stay_id']==stay_id].outtime.values[0]
        subject_id = icustays[icustays['stay_id']==stay_id].subject_id.values[0]
        hadm_id = icustays[icustays['stay_id']==stay_id].hadm_id.values[0]

        # 1시간 단위로 시간대 생성
        time_range = pd.date_range(start=intime, end=outtime, freq='H')

        # inputevents에 stay_id가 있는지 확인
        if stay_id in inputevents_stay_ids:
            # 해당 stay_id의 inputevents 데이터 필터링
            stay_inputevents = inputevents[inputevents['stay_id'] == stay_id]

            for timestamp in time_range:
                end_time = timestamp + pd.Timedelta(hours=1)

                # 해당 시간대에 해당하는 inputevents의 rate 합계 계산
                rates = stay_inputevents[(stay_inputevents['starttime'] < end_time) & 
                                         (stay_inputevents['endtime'] >= timestamp)]['rate']
                total_rate = rates.sum() if not rates.empty else 0

                resampled_data.append({'subject_id':subject_id,'hadm_id':hadm_id,'stay_id': stay_id, 'charttime': timestamp, name: total_rate})
        else:
            # inputevents에 stay_id가 없는 경우, 모든 rate를 0으로 설정
            for timestamp in time_range:
                resampled_data.append({'subject_id':subject_id,'hadm_id':hadm_id,'stay_id': stay_id, 'charttime': timestamp, name: 0})
    resampled_data = pd.DataFrame(resampled_data)
    if notrate:
        resampled_data.loc[resampled_data[name]>0, name] = 1
    return pd.DataFrame(resampled_data)


def resample_labvalues(chartevents_,labevents_,icustays,valuename):
    rt = []
    icustays_intime = icustays[['subject_id','hadm_id','stay_id','intime']]
    icustays_intime = icustays_intime.rename(columns={'intime':'charttime'})
    icustays_outtime = icustays[['subject_id','hadm_id','stay_id','outtime']]
    icustays_outtime = icustays_outtime.rename(columns={'outtime':'charttime'})
    for i in tqdm(icustays.hadm_id.unique()):
        tmp_hosp = labevents_[labevents_['hadm_id']==i]
        tmp_hosp = tmp_hosp[['subject_id','hadm_id','charttime','valuenum']]
        tmp_hosp.sort_values('charttime',ascending=True, inplace=True)

        for i in icustays[icustays['hadm_id']==i].stay_id.unique():
            tmp = chartevents_[chartevents_['stay_id']==i]
            tmp_id = icustays[icustays['stay_id']==i][['subject_id','hadm_id','stay_id']].iloc[0].to_list()
            tmp_intime = icustays_intime[icustays_intime['stay_id']==i]
            tmp_outtime = icustays_outtime[icustays_outtime['stay_id']==i]

            tmp = pd.concat([tmp, tmp_intime, tmp_outtime])

            tmp = tmp[(tmp['charttime'].values >= tmp_intime.charttime.values)&(tmp['charttime'].values <= tmp_outtime.charttime.values)]
            tmp.index = pd.DatetimeIndex(tmp['charttime'])

            tmp = pd.DataFrame(tmp['valuenum'].resample(rule='H', origin='start').last())
            tmp.reset_index(drop=False,inplace=True)

            if tmp.iloc[0].isna().any():
                tmp_hosp = tmp_hosp[(tmp_hosp['charttime'].values < tmp_intime.charttime.values)]
                if not tmp_hosp.empty:
                    tmp.iloc[0]['valuenum'] = tmp_hosp.iloc[-1].valuenum
            tmp[['subject_id','hadm_id','stay_id']] = tmp_id
            rt.append(tmp)
    rt = pd.concat(rt)
    rt = rt[['subject_id','hadm_id','stay_id','charttime','valuenum']]
    rt = resample_fill(rt)
    rt.rename(columns={'valuenum':valuename},inplace=True)
    return rt

def resample_vitals(chartevents_,icustays,valuename):
    rt = []
    icustays_intime = icustays[['subject_id','hadm_id','stay_id','intime']]
    icustays_intime = icustays_intime.rename(columns={'intime':'charttime'})

    icustays_outtime = icustays[['subject_id','hadm_id','stay_id','outtime']]
    icustays_outtime = icustays_outtime.rename(columns={'outtime':'charttime'})
    for i in tqdm(icustays.stay_id.unique()):
        tmp = chartevents_[chartevents_['stay_id']==i]
        tmp_id = icustays[icustays['stay_id']==i][['subject_id','hadm_id','stay_id']].iloc[0].to_list()
        tmp_intime = icustays_intime[icustays_intime['stay_id']==i]
        tmp_outtime = icustays_outtime[icustays_outtime['stay_id']==i]

        tmp = pd.concat([tmp, tmp_intime, tmp_outtime])
        tmp = tmp[(tmp['charttime'].values >= tmp_intime.charttime.values)&(tmp['charttime'].values <= tmp_outtime.charttime.values)]
        tmp.index = pd.DatetimeIndex(tmp['charttime'])
        tmp = pd.DataFrame(tmp['valuenum'].resample(rule='H', origin='start').last())
        tmp.reset_index(drop=False,inplace=True)
        tmp[['subject_id','hadm_id','stay_id']] = tmp_id
        rt.append(tmp)
    rt = pd.concat(rt)
    rt = rt[['subject_id','hadm_id','stay_id','charttime','valuenum']]
    rt = resample_fill(rt)
    rt.rename(columns={'valuenum':valuename},inplace=True)
    return rt

def resample_urine(chartevents_,icustays,valuename):
    rt = []
    icustays_intime = icustays[['subject_id','hadm_id','stay_id','intime']]
    icustays_intime = icustays_intime.rename(columns={'intime':'charttime'})

    icustays_outtime = icustays[['subject_id','hadm_id','stay_id','outtime']]
    icustays_outtime = icustays_outtime.rename(columns={'outtime':'charttime'})
    for i in tqdm(icustays.stay_id.unique()):
        tmp = chartevents_[chartevents_['stay_id']==i]
        tmp_id = icustays[icustays['stay_id']==i][['subject_id','hadm_id','stay_id']].iloc[0].to_list()
        tmp_intime = icustays_intime[icustays_intime['stay_id']==i]
        tmp_outtime = icustays_outtime[icustays_outtime['stay_id']==i]

        tmp = pd.concat([tmp, tmp_intime, tmp_outtime])
        tmp = tmp[(tmp['charttime'].values >= tmp_intime.charttime.values)&(tmp['charttime'].values <= tmp_outtime.charttime.values)]
        tmp.index = pd.DatetimeIndex(tmp['charttime'])
        tmp = pd.DataFrame(tmp['valuenum'].resample(rule='H', origin='start').sum())
        tmp.reset_index(drop=False,inplace=True)
        tmp[['subject_id','hadm_id','stay_id']] = tmp_id
        rt.append(tmp)
    rt = pd.concat(rt)
    rt = rt[['subject_id','hadm_id','stay_id','charttime','valuenum']]
    rt = resample_fill(rt)
    rt.rename(columns={'valuenum':valuename},inplace=True)
    return rt


def AKI_UO_annotation(resample_uo,admission_weight):
    for i in tqdm(range(6,49,1)):
        rolling_avg=resample_uo.groupby('stay_id').rolling(window=str(i)+'H', on='charttime',min_periods=i)['uo'].mean().reset_index()
        rolling_avg.rename(columns={'uo':'roll_%iH' % i},inplace=True)
        resample_uo = pd.merge(resample_uo,rolling_avg,on=['stay_id','charttime'],how='left')
    df = pd.merge(resample_uo,admission_weight,on=['subject_id','stay_id'],how='left')
    df['6-12H_min'] = df[df.columns[6:13]].min(axis=1)/df['valuenum']
    df['12H-_min'] = df[df.columns[12:49]].min(axis=1)/df['valuenum']
    df['24H-_min'] = df[df.columns[24:49]].min(axis=1)/df['valuenum']
    df['AKI_UO']=0
    df.loc[df['6-12H_min']<0.5, 'AKI_UO'] = 1
    df.loc[df['12H-_min']<0.5, 'AKI_UO'] = 2
    df.loc[df['12H-_min']==0, 'AKI_UO'] = 3
    df.loc[df['24H-_min']<0.3, 'AKI_UO'] = 3
    df = df[['subject_id','hadm_id','stay_id','charttime','AKI_UO']]
    return df


def AKI_SCr_annotation(resample_SCr,baseline_SCr,resample_rrt):
    df = resample_SCr.copy()
    df = pd.merge(df,baseline_SCr,on=['subject_id','hadm_id','stay_id'],how='left')
    df = pd.merge(df,resample_rrt,on=['subject_id','hadm_id','stay_id','charttime'],how='left')
    SCr_48hrs_min = df.groupby('stay_id').rolling(window='48H', on='charttime')['SCr'].min().reset_index()
    SCr_48hrs_min.rename(columns={'SCr':'SCr_48hrs_min'},inplace=True)
    df = pd.merge(df,SCr_48hrs_min,on=['stay_id','charttime'],how='left')
    
    df['AKI_SCr'] = 0
    df.loc[((df['SCr']>=1.5*df['baseline_SCr'])&(df['SCr']<2.0*df['baseline_SCr']))|(df['SCr']>=0.3+df['SCr_48hrs_min']), 'AKI_SCr'] = 1
    df.loc[(df['SCr']>=2.0*df['baseline_SCr'])&(df['SCr']<3.0*df['baseline_SCr']), 'AKI_SCr'] = 2
    df.loc[(df['SCr']>=3.0*df['baseline_SCr'])|((df['SCr']>=0.3+df['SCr_48hrs_min'])&(df['SCr']>=4.0))|(df['RRT']>0), 'AKI_SCr'] = 3
    df = df[['subject_id','hadm_id','stay_id','charttime','AKI_SCr']]
    return df