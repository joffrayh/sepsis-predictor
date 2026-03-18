import argparse
import json
import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
from sklearn.impute import KNNImputer

import math
from tqdm.auto import tqdm
import warnings 
warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="processed_files",
                       help="Directory to save processed files")
    parser.add_argument("--missing_threshold", type=float, default=0.8,
                       help="Threshold for dropping columns with missing values (default: 0.7)")
    parser.add_argument("--timestep", type=int, default=4,
                       help="Size of timestep in hours (default: 4)")
    parser.add_argument("--window_before", type=int, default = 24,
                       help="Hours to include before onset time (default: 24)")
    parser.add_argument("--window_after", type=int, default= 72,
                       help="Hours to include after onset time (default: 72)")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Number of subjects to sample for testing (default: None, use all subjects)")
    return parser.parse_args()

def load_measurement_mappings():
    print('Loading measurement mappings')
    with open(f"{BASE_DIR}/ReferenceFiles/measurement_mappings.json", "r") as f:
        measurements = json.load(f)
    
    code_to_concept = {}
    for concept, info in measurements.items():
        for code in info['codes']:
            code_to_concept[code] = concept
            
    hold_times = {}
    for concept, info in measurements.items():
        if 'hold_time' in info:
            hold_times[concept] = info['hold_time']
            
    return measurements, code_to_concept, hold_times

def load_and_filter_chunked(filename, valid_stays, onset_df=None, time_col=None, winb4=24, winaft=72, itemid_filter=None):
    """
    Reads large CSV files in chunks and filters out rows not belonging 
    to our target patients or target time windows. 
    This saves huge RAM compared to old function. 
    """
    filepath = f'processed_files/{filename}'
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return pd.DataFrame()

    CHUNKSIZE = 1000000

    with open(filepath, 'rb') as f:
        total_chunks = (sum(1 for _ in f) -1) // CHUNKSIZE + 1

    chunks = []
    # read in row chunks to keep memory usage small
    for chunk in tqdm(
            pd.read_csv(
                filepath, 
                sep='|', 
                chunksize=CHUNKSIZE, 
                low_memory=False
            ), 
            total=total_chunks, 
            desc=f'\tProcessing {filename}',
            ncols=100

        ):

        # get valid patients
        chunk = chunk[chunk['stay_id'].isin(valid_stays)]
        if chunk.empty: continue
            
        # filter by item IDs if provided (e.g. for chartevents)
        if itemid_filter is not None and 'itemid' in chunk.columns:
            chunk = chunk[chunk['itemid'].astype(str).isin(itemid_filter)]
            if chunk.empty: continue
                
        # filter by time window if time column is provided
        if onset_df is not None and time_col is not None and time_col in chunk.columns:
            chunk = chunk.merge(onset_df[['stay_id', 'anchor_time']], on='stay_id', how='inner')
            mask = (chunk[time_col] >= chunk['anchor_time'] - winb4*3600) & \
                (chunk[time_col] < chunk['anchor_time'] + winaft*3600)
            chunk = chunk[mask].drop(columns=['anchor_time'])
            
        chunks.append(chunk)

    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return pd.DataFrame()

    
def process_patient_measurements_vectorized(ce_df, lab_df, mv_df, code_to_concept, batch_size=500, output_dir='processed_files'):
    print('Pivoting chartevents, lab events and mechvent...')

    if not ce_df.empty:
        print('\tMapping chartevents itemids to concepts...')
        ce_df['concept'] = ce_df['itemid'].astype(str).map(code_to_concept)
    else:
        print("WARNING: chartevents dataframe is empty.")

    if not lab_df.empty:
        print('\tMapping labevents itemids to concepts...')
        lab_df['concept'] = lab_df['itemid'].astype(str).map(code_to_concept)
    else:
        print("WARNING: labevents dataframe is empty.")

    print('\tJoining chartevents and labevents...')
    ce_lab = pd.concat([
        ce_df[['stay_id', 'charttime', 'concept', 'valuenum']] if not ce_df.empty else pd.DataFrame(),
        lab_df[['stay_id', 'charttime', 'concept', 'valuenum']] if not lab_df.empty else pd.DataFrame()
    ])
    ce_lab = ce_lab.dropna(subset=['concept'])

    # free the input frames now they're combined
    del ce_df, lab_df

    all_stay_ids = ce_lab['stay_id'].unique()
    batches = [all_stay_ids[i:i + batch_size] for i in range(0, len(all_stay_ids), batch_size)]
    print(f'\tPivoting in {len(batches)} batches of up to {batch_size} stays...')

    temp_dir = os.path.join(output_dir, '_pivot_temp')
    os.makedirs(temp_dir, exist_ok=True)
    batch_paths = []

    for i, batch_stays in enumerate(tqdm(batches, desc='\tPivoting batches', ncols=100)):
        batch = ce_lab[ce_lab['stay_id'].isin(batch_stays)]
        if batch.empty:
            continue

        wide = batch.pivot_table(
            index=['stay_id', 'charttime'],
            columns='concept',
            values='valuenum',
            aggfunc='last'
        ).reset_index()
        wide.columns.name = None

        path = os.path.join(temp_dir, f'batch_{i}.parquet')
        wide.to_parquet(path, compression='zstd')
        batch_paths.append(path)

        # explicitly free batch memory before next iteration
        del wide, batch

    del ce_lab

    print('\tConcatenating batches from disk...')
    wide_data = pd.concat(
        [pd.read_parquet(p) for p in tqdm(batch_paths, desc='\tReading batches', ncols=100)],
        ignore_index=True
    )

    # clean up temp files
    for p in batch_paths:
        os.remove(p)
    os.rmdir(temp_dir)

    # add mechvent
    if not mv_df.empty:
        print('\tMerging mechvent with wide data...')
        mv_clean = mv_df[['stay_id', 'charttime', 'mechvent']].drop_duplicates(
            subset=['stay_id', 'charttime'], keep='last'
        )
        wide_data = pd.merge(wide_data, mv_clean, on=['stay_id', 'charttime'], how='outer')
    else:
        print("WARNING: mechvent dataframe is empty.")
        wide_data['mechvent'] = np.nan

    return wide_data

def handle_outliers(df):
    print('Handling outliers in patient timeseries data')
    
    # outlier replacement
    outlier_bounds = {
        'weight_kg': (None, 300),
        'weight_lb': (None, 660),
        'heart_rate': (None, 250),
        'sbp_arterial': (None, 300),
        'map': (0, 200),
        'dbp_arterial': (0, 200),
        'respiratory_rate': (None, 80),
        'oxygen_flow': (None, 70),
        'peep': (0, 40),
        'tidal_volume': (None, 1800),
        'minute_volume': (None, 50),
        'potassium': (1, 15),
        'sodium': (95, 178),
        'chloride': (70, 150),
        'glucose': (1, 1000),
        'creatinine': (None, 150),
        'magnesium': (None, 10),
        'calcium_total': (None, 20),
        'calcium_ionized': (None, 5),
        'total_co2': (None, 120),
        'ast': (None, 10000),
        'alt': (None, 10000),
        'hemoglobin': (None, 20),
        'hematocrit': (None, 65),
        'wbc': (None, 500),
        'platelets': (None, 2000),
        'inr': (None, 20),
        'ph_arterial': (6.7, 8.0),
        'arterial_o2_pressure': (None, 700),
        'arterial_co2_pressure': (None, 200),
        'arterial_base_excess': (-50, None),
        'lactic_acid': (None, 30),
        'bilirubin_total': (None, 30)
    }

    # apply the standard bounds
    for col, (min_val, max_val) in tqdm(
            outlier_bounds.items(), 
            desc="\tHandling standard bounded outliers",
            ncols = 100
        ):
        if col in df.columns:
            if min_val is not None:
                df.loc[df[col] < min_val, col] = np.nan
            if max_val is not None:
                df.loc[df[col] > max_val, col] = np.nan

    # handle SpO2 special case
    print('\tHandling SpO2 outliers with unique logic...')
    if 'spo2' in df.columns:
        df.loc[df['spo2'] > 150, 'spo2'] = np.nan
        df.loc[df['spo2'] > 100, 'spo2'] = 100
    
    # handle temperature special case
    print('\tHandling temperature outliers with unique logic...')
    if 'temp_C' in df.columns:
        if 'temp_F' in df.columns:
            mask = (df['temp_C'] > 90) & (df['temp_F'].isna())
            df.loc[mask, 'temp_F'] = df.loc[mask, 'temp_C']
        df.loc[df['temp_C'] > 90, 'temp_C'] = np.nan
    
    # handle FiO2 special case
    print('\tHandling FiO2 outliers with unique logic...')
    if 'fio2' in df.columns:
        df.loc[df['fio2'] > 100, 'fio2'] = np.nan
        df.loc[df['fio2'] < 1, 'fio2'] *= 100
        df.loc[df['fio2'] < 20, 'fio2'] = np.nan
            
    return df

def estimate_gcs_from_rass(df):
    '''
    estimate GCS (Glascow Coma Scale. which measures level of consciousness)  
    from RASS (Richmond Agitation-Sedation Scale, which measures level of
    agitation or sedation) when GCS is missing, based on certain mappings.
    '''

    print('Estimating GCS from RASS when GCS is missing...')
    if 'gcs' not in df.columns:
        print('\tGCS column not found in dataframe.')
        print('\tSetting it to NaN and estimating from RASS where possible.')
        df['gcs'] = np.nan

    mappings = {
        0: 15,
        1: 15,
        2: 15,
        3: 15,
        4: 15,
        -1: 14,
        -2: 12,
        -3: 11,
        -4: 6,
        -5: 3
    }

    mask = df['gcs'].isna()
    df.loc[mask, 'gcs'] = df.loc[mask, 'richmond_ras'].map(mappings)

    print('\tCompleted GCS estimation from RASS.')

    return df

def estimate_fio2(df):
    '''
    estimate FiO2 (fraction of inspired oxygen) when missing, based on oxygen 
    flow rates and device types.
    '''

    print('Estimating FiO2...')
    
    flow_columns = ['oxygen_flow', 'oxygen_flow_cannula_rate', 'oxygen_flow_rate']
    df['combined_o2_flow'] = df[flow_columns].bfill(axis=1).iloc[:, 0]
    
    # function to set fio2 based on flow thresholds for a given mask and device type
    def set_fio2_by_flow(mask, flow_thresholds, fio2_values):
        df_subset = df[mask].copy()
        df_subset['fio2'] = None 
        # for each threshold, set fio2 for rows with flow <= threshold, 
        # starting from highest threshold
        for threshold, fio2 in zip(flow_thresholds, fio2_values):
            flow_mask = df_subset['combined_o2_flow'] <= threshold
            df_subset.loc[flow_mask, 'fio2'] = fio2
        df.loc[mask, 'fio2'] = df_subset['fio2']


    # for simple flow devices (e.g. nasal cannula), use these flow-to-FiO2 mappings based on clinical estimates
    mask = (df['fio2'].isna()) & (df['combined_o2_flow'].notna()) & (df['oxygen_flow_device'].isin(['0', '2'])) 
    if mask.any():
        set_fio2_by_flow(mask, [15, 12, 10, 8, 6, 5, 4, 3, 2, 1], [70, 62, 55, 50, 44, 40, 36, 32, 28, 24])

    # for patients with missing FiO2 but no flow recorded, 
    # if device is simple (e.g. nasal cannula) we can assume room air (21% FiO2)
    mask = (df['fio2'].isna()) & (df['combined_o2_flow'].isna()) & (df['oxygen_flow_device'].isin(['0', '2']))
    df.loc[mask, 'fio2'] = 21

    # for face mask devices, use these flow-to-FiO2 mappings based on clinical estimates
    face_mask_types = ['3', '4', '5', '6', '8', '9', '10', '11', '12'] 
    mask = (df['fio2'].isna()) & (df['combined_o2_flow'].notna()) & (df['oxygen_flow_device'].isin(face_mask_types))
    if mask.any():
        set_fio2_by_flow(mask, [15, 12, 10, 8, 6, 4], [75, 69, 66, 58, 40, 36])

    # for high flow devices, use these flow-to-FiO2 mappings based on clinical estimates
    mask = (df['fio2'].isna()) & (df['combined_o2_flow'].notna()) & (df['oxygen_flow_device'] == '7') 
    if mask.any():
        df_subset = df[mask].copy()
        flow = df_subset['combined_o2_flow']
        df_subset.loc[flow >= 15, 'fio2'] = 100
        df_subset.loc[(flow >= 10) & (flow < 15), 'fio2'] = 90
        df_subset.loc[(flow < 10) & (flow > 8), 'fio2'] = 80
        df_subset.loc[(flow <= 8) & (flow > 6), 'fio2'] = 70
        df_subset.loc[flow <= 6, 'fio2'] = 60
        df.loc[mask, 'fio2'] = df_subset['fio2']

    # for high flow face mask devices, use these flow-to-FiO2 mappings based on clinical estimates
    mask = (df['fio2'].isna()) & (df['combined_o2_flow'].notna()) & (df['oxygen_flow_device'] == '13')
    if mask.any():
        df_subset = df[mask].copy()
        flow = df_subset['combined_o2_flow']
        df_subset.loc[flow >= 15, 'fio2'] = 100
        df_subset.loc[(flow >= 10) & (flow < 15), 'fio2'] = 80
        df_subset.loc[flow < 10, 'fio2'] = 60
        df.loc[mask, 'fio2'] = df_subset['fio2']

    # for simple face mask devices, use these flow-to-FiO2 mappings based on clinical estimates
    mask = (df['fio2'].isna()) & (df['combined_o2_flow'].notna()) & (df['oxygen_flow_device'] == '14')
    if mask.any():
        df_subset = df[mask].copy()
        flow = df_subset['combined_o2_flow']
        df_subset.loc[flow >= 10, 'fio2'] = 80
        df_subset.loc[(flow >= 5) & (flow < 10), 'fio2'] = 60
        df_subset.loc[flow < 5, 'fio2'] = 40
        df.loc[mask, 'fio2'] = df_subset['fio2']

    df = df.drop('combined_o2_flow', axis=1)
    return df

def handle_unit_conversions(df):
    '''
    Handle unit conversions for temperature, hemoglobin/hematocrit and bilirubin.

    The function uses empirical conversion formulas to fill in missing values
    when one measurement is available but its corresponding unit/related measurement is not.

    - Temperature conversions use standard F-C formula: C = (F - 32) / 1.8
    - Hemoglobin-hematocrit relationship: Hct = (Hgb * 2.862) + 1.216
    - Bilirubin relationship: Dir = (Total * 0.6934) - 0.1752
    '''
    print('Converting units...')
    mask = (df['temp_F'] > 25) & (df['temp_F'] < 45)  
    if mask.any():
        df.loc[mask, 'temp_C'] = df.loc[mask, 'temp_F']
        df.loc[mask, 'temp_F'] = None

    mask = df['temp_C'] > 70
    if mask.any():
        df.loc[mask, 'temp_F'] = df.loc[mask, 'temp_C']
        df.loc[mask, 'temp_C'] = None

    mask = (~df['temp_C'].isna()) & (df['temp_F'].isna())
    if mask.any():
        df.loc[mask, 'temp_F'] = df.loc[mask, 'temp_C'] * 1.8 + 32

    mask = (~df['temp_F'].isna()) & (df['temp_C'].isna())
    if mask.any():
        df.loc[mask, 'temp_C'] = (df.loc[mask, 'temp_F'] - 32) / 1.8

    mask = (~df['hemoglobin'].isna()) & (df['hematocrit'].isna())
    if mask.any():
        df.loc[mask, 'hematocrit'] = (df.loc[mask, 'hemoglobin'] * 2.862) + 1.216

    mask = (~df['hematocrit'].isna()) & (df['hemoglobin'].isna())
    if mask.any():
        df.loc[mask, 'hemoglobin'] = (df.loc[mask, 'hematocrit'] - 1.216) / 2.862

    mask = (~df['bilirubin_total'].isna()) & (df['bilirubin_direct'].isna())
    if mask.any():
        df.loc[mask, 'bilirubin_direct'] = (df.loc[mask, 'bilirubin_total'] * 0.6934) - 0.1752

    mask = (~df['bilirubin_direct'].isna()) & (df['bilirubin_total'].isna())
    if mask.any():
        df.loc[mask, 'bilirubin_total'] = (df.loc[mask, 'bilirubin_direct'] + 0.1752) / 0.6934

    return df

def sample_and_hold(df, vitalslab_hold):
    print('Performing sample and hold interpolation...')

    # ensure correct ordering before forward-fill logic
    df = df.sort_values(['stay_id', 'charttime']).reset_index(drop=True)

    cols_to_process = [col for col in vitalslab_hold if col in df.columns
                       and np.issubdtype(df[col].dtype, np.number)]

    for col in tqdm(cols_to_process, desc='\tProcessing columns', ncols=100):
        hold_period_secs = vitalslab_hold[col] * 3600

        # for each row, record the charttime of the last valid (non-NaN) measurement
        # within the same stay, then only carry the value forward if it's within
        # the hold period
        last_valid_time = (
            df[['stay_id', 'charttime', col]]
            .where(df[col].notna())  # NaN out rows where col is missing
            .groupby(df['stay_id'])['charttime']
            .transform(lambda x: x.ffill())  # forward fill the charttime of last valid
        )

        last_valid_value = (
            df.groupby('stay_id')[col]
            .transform(lambda x: x.ffill())  # forward fill the value itself
        )

        # only apply the held value where:
        # 1. the original value is NaN (it's a gap)
        # 2. the last valid measurement is within the hold period
        within_hold = (df['charttime'] - last_valid_time) <= hold_period_secs
        gap_mask = df[col].isna()

        df.loc[gap_mask & within_hold, col] = last_valid_value[gap_mask & within_hold]

    return df

def standardize_patient_trajectories(init_traj, data_dict, onset, timestep=4,
                                      window_before=24, window_after=72,
                                      output_dir='processed_files', flush_every=5000):
    '''
    Standardise patient trajectories to fixed time steps (e.g. every 4 hours)
    relative to ICU admission time (anchor_time).
    For each time step, we take the mean of all measurements within that time step.
    Processed data is flushed to disk every flush_every rows to avoid OOM errors.
    '''
    print('Standardising trajectories to fixed time step...')

    # create hash maps for the secondary tables
    fluid_grp = {k: v for k, v in data_dict['fluid'].groupby('stay_id')} if 'fluid' in data_dict else {}
    vaso_grp  = {k: v for k, v in data_dict['vaso'].groupby('stay_id')}  if 'vaso'  in data_dict else {}
    uo_grp    = {k: v for k, v in data_dict['UO'].groupby('stay_id')}    if 'UO'    in data_dict else {}
    abx_grp   = {k: v for k, v in data_dict['abx'].groupby('stay_id')}   if 'abx'   in data_dict else {}
    demog_dict       = data_dict['demog'].set_index('stay_id').to_dict('index') if 'demog' in data_dict else {}
    anchor_dict      = onset.set_index('stay_id')['anchor_time'].to_dict()
    onset_time_dict  = onset.set_index('stay_id')['onset_time'].to_dict()

    columns = [
        'timestep', 'stay_id', 'onset_time', 'timestamp', 'gender', 'age', 'charlson_comorbidity_index',
        're_admission', 'los', 'morta_hosp', 'morta_90',
        *[col for col in init_traj.columns if col not in ['timestep', 'stay_id', 'charttime']],
        'fluid_total', 'fluid_step', 'uo_total', 'uo_step', 'balance', 'vaso_median', 'vaso_max',
        'abx_given', 'hours_since_first_abx', 'num_abx'
    ]

    processed_data = []
    flush_count = 0
    temp_dir = os.path.join(output_dir, '_std_temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_paths = []

    grouped_traj = init_traj.groupby('stay_id')

    # for each patient trajectory, create fixed time steps relative to
    # ICU admission time (anchor_time) and estimate measurements within each step
    for stay_id, patient_traj in tqdm(grouped_traj, desc='\tStandardising per stay_id', ncols=150):

        start_time = anchor_dict.get(stay_id, 0)
        onset_time = onset_time_dict.get(stay_id, np.nan)
        demographics = demog_dict.get(stay_id, {})

        fluid_data = fluid_grp.get(stay_id, pd.DataFrame())
        vaso_data  = vaso_grp.get(stay_id,  pd.DataFrame())
        uo_data    = uo_grp.get(stay_id,    pd.DataFrame())
        abx_data   = abx_grp.get(stay_id,   pd.DataFrame())

        patient_times = sorted(patient_traj['charttime'].unique())
        if not patient_times:
            continue

        first_time = max(patient_times[0], start_time - window_before * 3600)
        last_time  = min(patient_times[-1], start_time + window_after * 3600)

        num_timesteps = math.ceil((last_time - first_time) / (timestep * 3600))

        for timestep_idx in range(num_timesteps):
            window_start = first_time + (timestep_idx * timestep * 3600)
            window_end   = window_start + (timestep * 3600)

            if window_end < first_time or window_start > last_time:
                continue

            # measurements: mean of all readings within the window, else NaN
            mask = (patient_traj['charttime'] >= window_start) & (patient_traj['charttime'] < window_end)
            window_data = patient_traj[mask]
            if len(window_data) == 0:
                measurements = {col: np.nan for col in patient_traj.columns if col not in ['stay_id', 'charttime']}
            else:
                measurements = window_data.mean(axis=0, skipna=True).to_dict()

            # fluids
            fluid_total, fluid_step = 0, 0
            if not fluid_data.empty:
                fluid_step  = fluid_data[(fluid_data['starttime'] < window_end) & (fluid_data['endtime'] >= window_start)]['amount'].sum()
                fluid_total = fluid_data[fluid_data['endtime'] < window_end]['amount'].sum()

            # vasopressors
            vaso_median, vaso_max = 0, 0
            if not vaso_data.empty:
                v_mask = (vaso_data['starttime'] <= window_end) & (vaso_data['endtime'] >= window_start)
                w_vaso = vaso_data[v_mask]
                if len(w_vaso) > 0:
                    vaso_median = w_vaso['rate_std'].median()
                    vaso_max    = w_vaso['rate_std'].max()

            # urine output
            uo_total, uo_step = 0, 0
            if not uo_data.empty:
                uo_step  = uo_data[(uo_data['charttime'] >= window_start) & (uo_data['charttime'] < window_end)]['value'].sum()
                uo_total = uo_data[uo_data['charttime'] < window_end]['value'].sum()

            # antibiotics
            abx_given, hrs_first_abx, num_abx = 0, None, 0
            if not abx_data.empty:
                abx_mask = (abx_data['starttime'] <= window_end) & (abx_data['stoptime'] >= window_start)
                w_abx = abx_data[abx_mask]
                if len(w_abx) > 0:
                    abx_given = 1
                    num_abx   = len(w_abx['drug'].unique())
                first_abx_time = abx_data['starttime'].min()
                if pd.notna(first_abx_time):
                    hrs_first_abx = (window_end - first_abx_time) / 3600

            timestep_data = {
                'timestep':   timestep_idx + 1,
                'stay_id':    stay_id,
                'timestamp':  window_start,
                'onset_time': onset_time,
                **demographics,
                **measurements,
                'fluid_total': fluid_total, 'fluid_step': fluid_step,
                'uo_total':    uo_total,    'uo_step':    uo_step,
                'balance':     fluid_total - uo_total,
                'vaso_median': vaso_median, 'vaso_max':   vaso_max,
                'abx_given':   abx_given,   'hours_since_first_abx': hrs_first_abx, 'num_abx': num_abx
            }

            timestep_data.pop('charttime', None)
            processed_data.append(timestep_data)

        # flush to disk once we've accumulated enough rows
        if len(processed_data) >= flush_every:
            path = os.path.join(temp_dir, f'chunk_{flush_count}.parquet')
            pd.DataFrame(processed_data, columns=columns).to_parquet(path, compression='zstd')
            temp_paths.append(path)
            processed_data = []
            flush_count += 1

    # flush any remaining rows
    if processed_data:
        path = os.path.join(temp_dir, f'chunk_{flush_count}.parquet')
        pd.DataFrame(processed_data, columns=columns).to_parquet(path, compression='zstd')
        temp_paths.append(path)

    print('\tConcatenating standardized chunks...')
    result = pd.concat(
        [pd.read_parquet(p) for p in tqdm(temp_paths, desc='\tReading chunks', ncols=100)],
        ignore_index=True
    )

    for p in temp_paths:
        os.remove(p)
    os.rmdir(temp_dir)

    return result


def fixgaps(x: np.ndarray) -> np.ndarray:
    '''
    fix gaps in a 1D array using linear interpolation. Only fills NaNs that are
    between valid measurements, leaves leading and trailing NaNs as they are.
    '''
    y = np.copy(x)
    nan_mask = np.isnan(x)
    valid_indices = np.arange(len(x))[~nan_mask]
    
    if len(valid_indices) == 0: return y
        
    nan_mask[:valid_indices[0]] = False
    nan_mask[valid_indices[-1]+1:] = False
    
    y[nan_mask] = interp1d(valid_indices, x[valid_indices])(np.arange(len(x))[nan_mask])
    return y


def handle_missing_values(df, missing_threshold=0.8):
    '''
    Handle missing values by:
    1. Dropping columns with missingness above missing_threshold.
    2. Linear interpolation (per patient) for columns with low missingness (< 5%).
    3. KNN imputation for columns with moderate missingness, in patient-aligned
       chunks to avoid imputing across patient boundaries.
    '''
    print('Handling missing values...')

    # ensure correct ordering before any imputation
    df = df.sort_values(['stay_id', 'timestamp']).reset_index(drop=True)

    # columns to exclude from missingness analysis — these are labels,
    # metadata, or treatment variables that should never be imputed
    non_measurement_cols = [
        'timestep', 'stay_id', 'onset_time', 'timestamp', 'gender', 'age',
        'charlson_comorbidity_index', 're_admission', 'los',
        'morta_hosp', 'morta_90', 'fluid_total', 'fluid_step',
        'uo_total', 'uo_step', 'balance', 'vaso_median', 'vaso_max',
        'abx_given', 'hours_since_first_abx', 'num_abx'
    ]
    measurement_cols = [col for col in df.columns if col not in non_measurement_cols]

    # 1. drop columns with too many missing values
    miss = df[measurement_cols].isna().sum() / len(df)
    cols_to_keep = miss[miss < missing_threshold].index
    df = df[df.columns[~df.columns.isin(measurement_cols)].tolist() + cols_to_keep.tolist()]

    # 2. linear interpolation per patient for low-missingness columns
    low_missing_cols = miss[(miss > 0) & (miss < 0.05)].index
    for col in tqdm(low_missing_cols, desc='\tLinear interpolation', ncols=100):
        if col in df.columns:
            df[col] = df.groupby('stay_id')[col].transform(
                lambda x: pd.Series(fixgaps(x.values), index=x.index)
            )

    # 3. KNN imputation for moderate-missingness columns
    cols_for_knn = [c for c in cols_to_keep if c not in low_missing_cols and c in df.columns]

    if cols_for_knn:
        ref = df[cols_for_knn].values.copy()

        # build patient-aligned chunks so KNN never imputes across patient boundaries
        # each chunk contains whole patients and grows until it reaches chunk_size rows
        print('\tBuilding patient-aligned chunks for KNN...')
        patient_groups = df.groupby('stay_id', sort=False).apply(lambda x: x.index.tolist())
        chunks = []
        current_chunk = []
        chunk_size = 9999
        for stay_indices in patient_groups:
            current_chunk.extend(stay_indices)
            if len(current_chunk) >= chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
        if current_chunk:
            chunks.append(current_chunk)

        print(f'\tRunning KNN imputation across {len(chunks)} patient-aligned chunks...')
        for chunk_indices in tqdm(chunks, desc='\tKNN imputation', ncols=100):
            imputer = KNNImputer(n_neighbors=1, keep_empty_features=True)
            ref[chunk_indices, :] = imputer.fit_transform(ref[chunk_indices, :])

        df[cols_for_knn] = ref
        print(f'\tRemaining missing values after KNN: {df.isna().sum().sum()}')

    return df

def calculate_derived_variables(df):
    '''
    calculate derived variables such as P/F ratio (arterial oxygen pressure / FiO2),
    Shock Index (heart rate / systolic blood pressure), SOFA score and SIRS criteria.
    '''

    df['gender'] = df['gender'] - 1
    df.loc[df['age'] > 150, 'age'] = 91.4
    df['mechvent'] = df['mechvent'].fillna(0)
    df.loc[df['mechvent'] > 0, 'mechvent'] = 1
    df['charlson_comorbidity_index'] = df['charlson_comorbidity_index'].fillna(df['charlson_comorbidity_index'].median())
    df['vaso_median'] = df['vaso_median'].fillna(0)
    df['vaso_max'] = df['vaso_max'].fillna(0)

    df['pf_ratio'] = df['arterial_o2_pressure'] / (df['fio2'] / 100)
    df['shock_index'] = df['heart_rate'] / df['sbp_arterial']
    df.loc[np.isinf(df['shock_index']), 'shock_index'] = np.nan
    df['shock_index'] = df['shock_index'].fillna(df['shock_index'].mean())

    # SOFA respiratory — vectorised pd.cut
    pf = df['pf_ratio']
    df['sofa_resp'] = pd.cut(pf, bins=[-np.inf, 100, 200, 300, 400, np.inf],
                              labels=[4, 3, 2, 1, 0], right=False).astype(float).fillna(0).astype(int)

    # SOFA coagulation
    plt = df['platelets']
    df['sofa_coag'] = pd.cut(plt, bins=[-np.inf, 20, 50, 100, 150, np.inf],
                              labels=[4, 3, 2, 1, 0], right=False).astype(float).fillna(0).astype(int)

    # SOFA liver
    bili = df['bilirubin_total']
    df['sofa_liver'] = pd.cut(bili, bins=[-np.inf, 1.2, 2.0, 6.0, 12.0, np.inf],
                               labels=[0, 1, 2, 3, 4], right=False).astype(float).fillna(0).astype(int)

    # SOFA cardiovascular — depends on both MAP and vaso, use np.select
    map_ = df['map']
    vaso = df['vaso_max']
    cv_conditions = [
        map_.isna() & vaso.isna(),
        map_ >= 70,
        (map_ >= 65) & (map_ < 70),
        map_ < 65,
        vaso <= 0.1,
        vaso > 0.1,
    ]
    df['sofa_cv'] = np.select(cv_conditions, [0, 0, 1, 2, 3, 4], default=0)

    # SOFA CNS
    gcs = df['gcs']
    df['sofa_cns'] = pd.cut(gcs, bins=[-np.inf, 6, 9, 12, 14, np.inf],
                             labels=[4, 3, 2, 1, 0], right=False).astype(float).fillna(0).astype(int)

    # SOFA renal — creatinine takes priority over UO
    cr = df['creatinine']
    uo = df['uo_step']
    cr_score = pd.cut(cr, bins=[-np.inf, 1.2, 2.0, 3.5, 5.0, np.inf],
                      labels=[0, 1, 2, 3, 4], right=False).astype(float)
    uo_score = np.select([uo >= 84, uo >= 34, uo < 34], [0, 3, 4], default=0)
    df['sofa_renal'] = np.where(cr.notna(), cr_score.fillna(0), 
                                np.where(uo.notna(), uo_score, 0)).astype(int)

    df['sofa_score'] = (df['sofa_resp'] + df['sofa_coag'] + df['sofa_liver'] +
                        df['sofa_cv'] + df['sofa_cns'] + df['sofa_renal'])

    # SIRS — vectorised
    sirs = pd.DataFrame(index=df.index)
    sirs['temp'] = ((df['temp_C'] >= 38) | (df['temp_C'] <= 36)).astype(int)
    sirs['hr']   = (df['heart_rate'] > 90).astype(int)
    sirs['rr']   = ((df['respiratory_rate'] >= 20) | (df['arterial_co2_pressure'] <= 32)).astype(int)
    sirs['wbc']  = ((df['wbc'] >= 12) | (df['wbc'] < 4)).astype(int)
    df['sirs_score'] = sirs.sum(axis=1)

    return df

def apply_exclusion_criteria(df):
    '''
    apply exclusion criteria to filter out patients who do not meet the study
    criteria. Everything is vectorised. The exclusion criteria include:
    1. Extreme UO (>12000 ml in a 4h window)
    2. Extreme Fluid (>10000 ml in a 4h window)
    3. Early Death (death within 24 hours of first measurement)
    '''
    print('Applying exclusion criteria...')
    excluded_counts = {}
    
    # 1. Extreme UO
    extreme_uo_stays = df[df['uo_step'] > 12000]['stay_id'].unique()
    df = df[~df['stay_id'].isin(extreme_uo_stays)]
    excluded_counts['extreme_uo'] = len(extreme_uo_stays)
    
    # 2. Extreme Fluid
    extreme_fluid_stays = df[df['fluid_step'] > 10000]['stay_id'].unique()
    df = df[~df['stay_id'].isin(extreme_fluid_stays)]
    excluded_counts['extreme_fluid'] = len(extreme_fluid_stays)
    
    # 3. Early Death (vectorised)
    first_times = df.groupby('stay_id')['timestamp'].min()
    last_times = df.groupby('stay_id')['timestamp'].max()
    morta = df.groupby('stay_id')['morta_hosp'].first()
    time_to_death = (last_times - first_times) / 3600
    early_deaths = morta[(morta == 1) & (time_to_death <= 24)].index
    df = df[~df['stay_id'].isin(early_deaths)]
    excluded_counts['early_death'] = len(early_deaths)
    
    print("\nExclusion Statistics:")
    print("-" * 50)
    for reason, count in excluded_counts.items(): print(f"Excluded due to {reason}: {count}")
    print("-" * 50)
    
    return df

def add_septic_shock_flag(df):
    '''
    Add septic shock flags based on the Sepsis-3 criteria, which defines septic 
    shock as sepsis with persistent hypotension requiring vasopressors to 
    maintain MAP ≥ 65 mm Hg, and having a serum lactate level > 2 mmol/L despite 
    adequate volume resuscitation.
    We will use the following logic to determine septic shock onset and censoring:
    1. Patient has infection (onset_time is not NaN) 
    2. Calculate rolling fluids over a 12-hour window (3 time steps if using 4h timesteps).
    3. Identify time steps where rolling fluids ≥ 2000 ml, MAP < 65 mm Hg,
         and lactic acid > 2 mmol/L.
    4. Mark the first time step meeting these criteria as septic shock onset (flag = 1).
    5. Mark all subsequent time steps for that patient as septic shock (flag = 2) 
        to indicate censoring after onset.
    '''

    print('Adding septic shock flags...')
    df = df.sort_values(['stay_id', 'timestamp'])

    WINDOW_STEPS = max(1, 12 // 4)
    
    df['rolling_fluid'] = df.groupby('stay_id')['fluid_step'].rolling(
        window=WINDOW_STEPS, min_periods=1
    ).sum().reset_index(level=0, drop=True)
    
    # Sepsis-3: haemodynamic criteria AND confirmed infection (onset_time is not NaN)
    has_infection = df['onset_time'].notna()
    shock_condition = (
        has_infection &
        (df['rolling_fluid'] >= 2000) &
        (df['map'] < 65) &
        (df['lactic_acid'] > 2)
    )
    
    df['septic_shock'] = 0
    first_shock = df[shock_condition].groupby('stay_id').head(1).index
    df.loc[first_shock, 'septic_shock'] = 1
    
    df['has_shocked'] = df.groupby('stay_id')['septic_shock'].cumsum()
    censored_mask = (df['has_shocked'] == 1) & (df['septic_shock'] == 0)
    df.loc[censored_mask, 'septic_shock'] = 2
    
    return df.drop(columns=['rolling_fluid', 'has_shocked'])

def add_sepsis_flag(df):
    '''
    Add sepsis flags based on Sepsis-3 criteria: SOFA >= 2 AND confirmed
    infection (onset_time is not NaN). Non-septic patients (onset_time is NaN)
    will always have sepsis = 0.
    Flags: 0 = no sepsis, 1 = onset timestep, 2 = post-onset (censored).
    '''
    print('Adding sepsis flags...')
    df = df.sort_values(['stay_id', 'timestamp'])

    # Both infection confirmation AND SOFA >= 2 required
    has_infection = df['onset_time'].notna()
    sepsis_condition = has_infection & (df['sofa_score'] >= 2)

    df['sepsis'] = 0
    first_sepsis = df[sepsis_condition].groupby('stay_id').head(1).index
    df.loc[first_sepsis, 'sepsis'] = 1

    df['has_sepsis'] = df.groupby('stay_id')['sepsis'].cumsum()
    censored_mask = (df['has_sepsis'] == 1) & (df['sepsis'] == 0)
    df.loc[censored_mask, 'sepsis'] = 2

    return df.drop(columns=['has_sepsis'])

def main():

    args = parse_args()
    measurements, code_to_concept, hold_times = load_measurement_mappings()
    
    # 1. Load Onset first so we know exactly which patients/times to keep
    print("Loading onset data...")
    onset = pd.read_csv('processed_files/cohort.csv', sep='|')    
    if args.sample_size is not None:
        print(f'Sampling {args.sample_size} subjects for testing')
        onset = onset.sample(n=args.sample_size, random_state=42)
    valid_stays = set(onset['stay_id'])
    
    # # 2. Load the rest of the data using chunking and filtering
    # print("Loading secondary files dynamically...")
    data_dict = {
        'demog': load_and_filter_chunked('demog_processed.csv', valid_stays),
        'abx': load_and_filter_chunked('abx_processed.csv', valid_stays),
        'fluid': load_and_filter_chunked('fluid.csv', valid_stays),
        'vaso': load_and_filter_chunked('vaso.csv', valid_stays),
        'UO': load_and_filter_chunked('uo.csv', valid_stays),
    }

    # memory intensive files loaded strictly against the item ID dict & -24h to +72h time windows
    valid_items = set(code_to_concept.keys())
    ce_df = load_and_filter_chunked('chartevents.csv', valid_stays, onset, time_col='charttime', itemid_filter=valid_items)
    lab_df = load_and_filter_chunked('labu.csv', valid_stays, onset, time_col='charttime', itemid_filter=valid_items)
    mv_df = load_and_filter_chunked('mechvent.csv', valid_stays, onset, time_col='charttime')

    # 3. Vectorised pivoting (replaces the nested row-by-row loops)
    init_traj = process_patient_measurements_vectorized(ce_df, lab_df, mv_df, code_to_concept)
    
    if init_traj.empty:
        print("No valid trajectories found matching criteria.")
        return
        
    # free memory
    del ce_df, lab_df, mv_df 

    print('Saving intermediate trajectory after initial processing...')
    init_traj.to_parquet(f"{args.output_dir}/init_traj_step1_temp_v2.zst", compression='zstd')
    init_traj = pd.read_parquet(f"{args.output_dir}/init_traj_step1_temp_v2.zst")



    # 4. Standard processing
    init_traj = handle_outliers(init_traj)
    init_traj = estimate_gcs_from_rass(init_traj)
    init_traj = estimate_fio2(init_traj)
    init_traj = handle_unit_conversions(init_traj)
    init_traj = sample_and_hold(init_traj, hold_times) 

    print('Saving intermediate trajectory after outlier handling and sample-and-hold...')
    init_traj.to_parquet(f"{args.output_dir}/init_traj_step2_outliers_and_sample_and_hold_v2.zst", compression='zstd')
    init_traj = pd.read_parquet(f"{args.output_dir}/init_traj_step2_outliers_and_sample_and_hold_v2.zst")


    # 5. standardise the time steps 
    # optimised heavily with pre-computed dictionaries compared to the original 
    # nested loops approach
    init_traj = standardize_patient_trajectories(
        init_traj, 
        data_dict,
        onset,
        timestep=args.timestep,
        window_before=args.window_before,
        window_after=args.window_after,
        output_dir=args.output_dir 
    )

    print('Saving intermediate trajectory after standardizing to fixed time grids...')
    init_traj.to_parquet(f"{args.output_dir}/init_traj_step3_standardized_grids_v2.zst", compression='zstd')
    init_traj = pd.read_parquet(f"{args.output_dir}/init_traj_step3_standardized_grids_v2.zst")


    # 6. Handle missing values
    init_traj = handle_missing_values(init_traj, args.missing_threshold)
    print('Saving intermediate trajectory after handling missing value...')
    init_traj.to_parquet(f"{args.output_dir}/init_traj_step4_handle_missing_vals_v2.zst", compression='zstd')
    init_traj = pd.read_parquet(f"{args.output_dir}/init_traj_step4_handle_missing_vals_v2.zst")

    # 7. Derived variables & vectorised exclusion criteria & septic shock/sepsis flags
    init_traj = calculate_derived_variables(init_traj)    
    init_traj = apply_exclusion_criteria(init_traj)
    init_traj = add_septic_shock_flag(init_traj)
    init_traj = add_sepsis_flag(init_traj)

    # 8. Save
    output_path = f"{args.output_dir}/patient_timeseries_v4_v2.parquet"
    os.makedirs(args.output_dir, exist_ok=True)
    init_traj.to_parquet(output_path, compression='zstd')
    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    main()
