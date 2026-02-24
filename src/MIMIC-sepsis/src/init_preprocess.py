"""
This script performs the secondary stage of data preprocessing for a sepsis cohort 
extracted from the MIMIC-IV database. It takes the intermediate CSV files, cleans 
the data, and aligns clinical events to identify Sepsis-3 infection onset times.

Key Operations:
* Memory-Optimized Loading: Selectively loads the necessary files (demographics, 
  antibiotics, microbiology, and labs).
* Data Cleaning & Standardization: Combines and cleans lab and microbiology data, 
  handles missing variables, and calculates 30-day hospital readmission flags.
* Stay ID Imputation: Uses time-window logic (+/- 48 hours) 
  to map unassociated clinical events to their correct ICU stay IDs.
* Sepsis-3 Onset Calculation: Determines the presumed onset of infection by evaluating 
  the temporal relationship between antibiotic administration and bacteriological sampling 
  (Abx given <= 24 hours before or <= 72 hours after a culture).

Outputs (saved to 'processed_files/'):
* onset.csv: The final presumed infection onset times for target patients.
* bacterio_processed.csv: Cleaned and ID-mapped microbiology/culture events.
* demog_processed.csv: Cleaned demographic data with readmission flags.
* labu.csv: Consolidated lab events (chart and lab events).
* abx_processed.csv: Cleaned antibiotic administration events.
"""

import argparse
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_raw", action='store_true', help="If specified, additionally save trajectories without normalized features")
    parser.add_argument("--save_intermediate", action='store_true', default=True, help="If specified, save off intermediate tables used to construct final patient table")
    return parser.parse_args()

def load_processed_files():
    print('Loading required processed files...')
    files = {
        'abx': 'abx.csv', 
        'culture': 'culture.csv',
        'microbio': 'microbio.csv',
        'demog': 'demog.csv',
    }
    
    data = {}
    for key, filename in files.items():
        data[key] = pd.read_csv(f'processed_files/{filename}', sep='|', low_memory=False)
        
    print('Loading and combining lab data...')
    labs_ce = pd.read_csv('processed_files/labs_ce.csv', sep='|', low_memory=False)
    labs_le = pd.read_csv('processed_files/labs_le.csv', sep='|', low_memory=False)
    labs_le = labs_le.rename(columns={'timestp': 'charttime'})
    data['labU'] = pd.concat([labs_ce, labs_le], sort=False, ignore_index=True)
    
    return data

def process_microbio_data(microbio, culture):
    ''''
    Process microbiology data by filling missing charttime (exact time of event) 
    with chartdate (date of event) and combining with culture data.
    '''
    microbio['charttime'] = microbio['charttime'].fillna(microbio['chartdate'])
    microbio = microbio.drop(columns=['chartdate'])
    return pd.concat([microbio, culture], sort=False, ignore_index=True)

def process_demog_data(demog):
    ''''
    Process demographic data by filling missing mortality and 
    comorbidity index values with 0, and dropping duplicates.
    '''
    demog['morta_90'] = demog['morta_90'].fillna(0)
    demog['morta_hosp'] = demog['morta_hosp'].fillna(0) 
    demog['charlson_comorbidity_index'] = demog['charlson_comorbidity_index'].fillna(0)
    return demog.drop_duplicates(subset=['admittime','dischtime'], keep='first').copy()

def calculate_readmissions(demog, cutoff=3600*24*30):
    '''
    Add a binary 're_admission' column to the demographic data indicating 
    whether a patient was readmitted to icu within 30 days of their previous discharge.
    As the rows are sorted by subject_id and admittime, this is done 
    by grouping the subject_ids and taking away the previous discharge time from
    the current admission time, and checking if the 
    difference is <= 30 days (in seconds).
    '''
    print('Calculating readmissions...')
    # vectorised approach to calculate readmissions within 30 days
    demog = demog.sort_values(['subject_id', 'admittime'])
    demog['prev_dischtime'] = demog.groupby('subject_id')['dischtime'].shift(1)
    
    demog['re_admission'] = 0
    mask = (demog['admittime'] - demog['prev_dischtime']) <= cutoff
    demog.loc[mask, 're_admission'] = 1
    
    return demog.drop(columns=['prev_dischtime'])

def fill_missing_icustay_ids(bacterio, demog, abx):
    '''
    Fill in missing ICU stay IDs in the bacteriology and antibiotic (abx) data by
    matching the event times to the ICU stay time windows (intime to outtime)
    for the corresponding subject and hostpital admitance ID (hadmid). 
    An event is associated with a stay if it occurs within 48 hours before 
    or after the stay, or if the patient only has one stay in the dataset.
    '''

    print('Filling-in missing ICUSTAY IDs in bacterio (Vectorized)')
    
    # vectorised bacterio update
    # get missing bacterio rows
    missing_bact = bacterio[bacterio['stay_id'].isna()].copy()
    missing_bact['idx'] = missing_bact.index 
    
    # drop the empty stay_id column so merge doesn't create stay_id_x and stay_id_y
    missing_bact = missing_bact.drop(columns=['stay_id']) 
    
    # prepare demographics with stay counts per subject
    demog_sub = demog[['subject_id', 'stay_id', 'intime', 'outtime']].dropna(subset=['subject_id']).copy()
    stay_counts = demog_sub.groupby('subject_id').size().reset_index(name='stay_count')
    demog_sub = demog_sub.merge(stay_counts, on='subject_id')
    
    # merge and filter based on within 48h or only 1 stay in icu for the subject
    merged_bact = missing_bact.merge(demog_sub, on='subject_id', how='inner')
    time_mask = (merged_bact['charttime'] >= merged_bact['intime'] - 48*3600) & (merged_bact['charttime'] <= merged_bact['outtime'] + 48*3600)
    valid_bact = merged_bact[time_mask | (merged_bact['stay_count'] == 1)]

    # NOTE: this could be wrong!! It follows the logic of the original script
    # but I believe they didn't mean to overwrite the stay_id for duplicates
    bacterio.loc[valid_bact['idx'], 'stay_id'] = valid_bact['stay_id'].values

    print('Filling-in missing ICUSTAY IDs in ABx')
    
    # vectorised abx update
    abx['idx'] = abx.index
    
    # prepare demographics with stay counts per hadm_id
    demog_abx_sub = demog[['hadm_id', 'stay_id', 'intime', 'outtime']].dropna(subset=['hadm_id']).copy()
    abx_counts = demog_abx_sub.groupby('hadm_id').size().reset_index(name='stay_count')
    demog_abx_sub = demog_abx_sub.merge(abx_counts, on='hadm_id')
    
    # merge and filter based on within 48h or only 1 stay in icu for the subject
    merged_abx = abx.merge(demog_abx_sub, on='hadm_id', how='inner')
    time_mask_abx = (merged_abx['starttime'] >= merged_abx['intime'] - 48*3600) & (merged_abx['starttime'] <= merged_abx['outtime'] + 48*3600)
    valid_abx = merged_abx[time_mask_abx | (merged_abx['stay_count'] == 1)]

    # NOTE: this could be wrong!! It follows the logic of the original script
    # but I believe they didn't mean to overwrite the stay_id for duplicates
    valid_abx = valid_abx.drop_duplicates(subset=['idx'], keep='last')
    
    # assign the stay_id back to the original abx dataframe
    abx.loc[valid_abx['idx'], 'stay_id'] = valid_abx['stay_id'].values
    abx = abx.drop(columns=['idx'])
                
    return bacterio, abx

def find_infection_onset(abx, bacterio):
    '''
    Find the presumed onset of infection according to Sepsis-3 guidelines.
    An infection onset is identified if:
    1) An antibiotic is given within 24 hours before a culture is taken, or
    2) An antibiotic is given within 72 hours after a culture is taken.
    The onset time is assigned as the time of antibiotic administration if condition 1 is met,
    or the time of culture if condition 2 is met. Only the earliest valid onset per stay is kept.
    '''

    print('Finding presumed onset of infection according to sepsis3 guidelines')
    
    # filter columns by stay_id and medical administration time
    abx_sub = abx[['stay_id', 'starttime']].dropna(subset=['stay_id']).copy()
    abx_sub['abx_idx'] = abx_sub.index # to preserve original processing order
    
    # filter columns by stay_id and culture charttime (time of culture collection)
    bact_sub = bacterio[['stay_id', 'subject_id', 'charttime']].dropna(subset=['stay_id']).copy()
    
    # find combinations matching the same stay_id
    merged = pd.merge(abx_sub, bact_sub, on='stay_id', how='inner')
    
    # find the closest bacterio event for each abx (antibiotic administration)
    merged['diff_hr'] = (merged['starttime'] - merged['charttime']).abs() / 3600
    merged = merged.sort_values(['abx_idx', 'diff_hr'])
    closest = merged.drop_duplicates(subset=['abx_idx']).copy()
    
    # apply conditions: 
    # time_diff <= 24 and abx takes place before bact 
    # OR time_diff <= 72 and abx takes place after bact
    cond1 = (closest['diff_hr'] <= 24) & (closest['starttime'] <= closest['charttime'])
    cond2 = (closest['diff_hr'] <= 72) & (closest['starttime'] >= closest['charttime'])

    valid = closest[cond1 | cond2].copy()
    
    # from the valid infection onsets, we want to find at what time 
    # the infection onset occurs. Accoding to sepsis3 guidelines,
    # if starttime of abx is within 24h before charttime of bact, then onset is at starttime of abx
    # if starttime of abx is within 72h after charttime of bact, then onset is at charttime of bact
    # so here we get the rows where the first condition is met
    valid_cond1 = (valid['diff_hr'] <= 24) & (valid['starttime'] <= valid['charttime'])
    
    # based on this first condition, we assignt the onset to starttime if it is true
    # and to charttime if it is false (which means the second condition must be true)
    valid['onset_time'] = np.where(valid_cond1, valid['starttime'], valid['charttime'])
    
    # only keep the earliest valid matching onset for the stay
    valid = valid.sort_values('abx_idx').drop_duplicates(subset=['stay_id'], keep='first')
    
    onset_df = valid[['subject_id', 'stay_id', 'onset_time']].copy()
    print(f'Number of preliminary, presumed septic trajectories: {len(onset_df)}')
    return onset_df

def main():
    args = parse_args()
    data = load_processed_files()
    
    # process microbio data
    bacterio = process_microbio_data(data['microbio'], data['culture'])
    
    # process demographics and add readmission column
    demog = process_demog_data(data['demog'])
    demog = calculate_readmissions(demog)
    
    # fill in missing stay_ids in bacterio and abx using demographics
    bacterio, data['abx'] = fill_missing_icustay_ids(bacterio, demog, data['abx'])
    
    # find the earliest infection onset according to sepsis3 guidelines
    onset = find_infection_onset(data['abx'], bacterio)
    
    # save processed data if requested
    if args.save_intermediate:
        print('Saving processed files...')
        onset.to_csv('processed_files/onset.csv', sep='|', index=False)
        bacterio.to_csv('processed_files/bacterio_processed.csv', sep='|', index=False)
        demog.to_csv('processed_files/demog_processed.csv', sep='|', index=False)
        data['labU'].to_csv('processed_files/labu.csv', sep='|', index=False)
        data['abx'].to_csv('processed_files/abx_processed.csv', sep='|', index=False)
        
    print('Processing complete!')
    return onset, bacterio, demog, data

if __name__ == "__main__":
    main()