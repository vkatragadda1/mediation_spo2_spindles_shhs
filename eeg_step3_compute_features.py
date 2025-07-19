# Calculates sleep macrostructures, bandpower, and spindle/SO coupling features. Saves to CSV for further analysis. 

# Load packages 
from collections import defaultdict
from itertools import groupby, product
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import linregress
from tqdm import tqdm
import mne
from fooof import FOOOF
#from data_functions import *
#from feature_extraction import *

# Load Luna 
import lunapi as lp
proj = lp.proj()
proj.silence()

# Computes sleep macrostructure metrics from epoch-level sleep stage data, including durations, latencies, stage percentages, transitions, and bout lengths.
def get_macrostructures(sleep_stages, epoch_time=30):
    # Assumes sleep_stages is between lights off to lights on only
    # Assums epoch level
    r = {}
    r['tib'] = len(sleep_stages)*epoch_time/3600
    sleep_ids = np.where(np.in1d(sleep_stages, [1,2,3,4]))[0]
    r['tst'] = len(sleep_ids)*epoch_time/3600
    r['se']  = r['tst']/r['tib']*100
    if len(sleep_ids)>0:
        r['waso'] = np.sum(sleep_stages[sleep_ids[0]:sleep_ids[-1]+1]==5)*epoch_time/60
        r['sl'] = sleep_ids[0]*epoch_time/60
    else:
        r['waso'] = np.nan
        r['sl'] = np.nan
    rem_ids = np.where(sleep_stages==4)[0]
    if len(rem_ids)>0:
        r['rl'] = rem_ids[0]*epoch_time/60
    else:
        r['rl'] = np.nan

    # Sleep stages
    r['w_time'] = (sleep_stages==5).sum()*epoch_time/60
    r['r_time'] = (sleep_stages==4).sum()*epoch_time/60
    r['n1_time'] = (sleep_stages==3).sum()*epoch_time/60
    r['n2_time'] = (sleep_stages==2).sum()*epoch_time/60
    r['n3_time'] = (sleep_stages==1).sum()*epoch_time/60
    r['r_perc'] = (sleep_stages==4).mean()*100
    r['n1_perc'] = (sleep_stages==3).mean()*100
    r['n2_perc'] = (sleep_stages==2).mean()*100
    r['n3_perc'] = (sleep_stages==1).mean()*100
    
    # Transition probability
    transmat = np.zeros((5,5))
    for i in range(len(sleep_stages)-1):
        s1 = sleep_stages[i]
        s2 = sleep_stages[i+1]
        if s1 in [1,2,3,4,5] and s2 in [1,2,3,4,5]:
            transmat[int(s1)-1,int(s2)-1] += 1
    transmat = transmat/transmat.sum(axis=1, keepdims=True)
    r['n3_continue_prob'] = transmat[0,0]
    r['n2_continue_prob'] = transmat[1,1]
    r['n1_continue_prob'] = transmat[2,2]
    r['r_continue_prob'] = transmat[3,3]
    r['w_continue_prob'] = transmat[4,4]
    
    # Bout duration
    ss = np.array(sleep_stages)
    ss[np.isnan(ss)|np.isinf(ss)] = -1
    r['n1_bout_dur'] = [0]
    r['n2_bout_dur'] = [0]
    r['n3_bout_dur'] = [0]
    r['r_bout_dur'] = [0]
    for k, l in groupby(ss):
        if k==1:
            r['n3_bout_dur'].append(len(list(l))*epoch_time/60)
        elif k==2:
            r['n2_bout_dur'].append(len(list(l))*epoch_time/60)
        elif k==3:
            r['n1_bout_dur'].append(len(list(l))*epoch_time/60)
        elif k==4:
            r['r_bout_dur'].append(len(list(l))*epoch_time/60)
    r['n1_bout_dur'] = max(r['n1_bout_dur'])
    r['n2_bout_dur'] = max(r['n2_bout_dur'])
    r['n3_bout_dur'] = max(r['n3_bout_dur'])
    r['r_bout_dur'] = max(r['r_bout_dur'])

    return r


# Computes absolute and relative bandpower, 1/f exponent, and delta slope across sleep stages and channels, excluding artifact-affected epochs.
def get_bandpower(spec_, freq, sleep_stages, artifact_indicator, ch_names):
    spec = np.array(spec_)
    # Apply artifact_indicator to spec
    for ii in range(artifact_indicator.shape[0]):
        for jj in range(artifact_indicator.shape[1]):
            if artifact_indicator[ii,jj]:
                spec[ii,jj] = np.nan
                    
    # In age norm paper, total power is 0.5-20Hz
    freq_ids = (freq>=0.5)&(freq<=30)
    freq = freq[freq_ids]
    spec = spec[...,freq_ids]

    channels = [x[0] for x in ch_names[::2]]  # Assumes left right
    channel_ids = [[x*2,x*2+1] for x in range(len(channels))]
    
    res = {}
    dfreq = freq[1]-freq[0]
    for sn, st in zip([1,2,3,4,5,5],['N3','N2','N1','R','WBSO','WASO']):#,'WAM','SO5min'
        if st=='WBSO':
            kk = np.arange(len(sleep_stages))
            so = np.where(sleep_stages<5)[0][0]
            ssids = (kk<so)&(sleep_stages==5)
        elif st=='SO5min':
            kk = np.arange(len(sleep_stages))
            so = np.where(sleep_stages<5)[0][0]
            ssids = (kk<so)&(sleep_stages==5)
            ssids[10:] = False
        elif st=='WASO':
            _ = np.where(sleep_stages<5)[0]
            so = _[0]; se = _[-1]+1
            ssids = sleep_stages==5
            ssids[:so] = False
            ssids[se:] = False
        elif st=='WAM':
            wake_mask = sleep_stages==5
            ssids = np.zeros(len(sleep_stages), dtype=bool)
            for k,l in groupby(wake_mask[::-1]):
                if k:
                    ssids[-len(list(l)):] = True
                break
        else:
            ssids = sleep_stages==sn
        Nss = np.sum(ssids)
        for chid, ch in zip(channel_ids, channels):
            if Nss==0:
                res[f'total_dbs_{st}_{ch}'] = np.nan
            else:
                psd = spec[ssids][:,chid]
                psd_db = 10*np.log10(np.nansum(psd, axis=-1)*dfreq)
                psd_db[np.isinf(psd_db)] = np.nan
                res[f'total_dbs_{st}_{ch}'] = np.nanmean(psd_db)
                
                # FOOF features
                psd_ = np.nanmean(psd, axis=(0,1))
                if np.isnan(psd_).all():
                    res[f'1_f_exponent_{st}_{ch}'] = np.nan
                else:
                    fm = FOOOF()
                    fm.fit(freq, psd_)
                    res[f'1_f_exponent_{st}_{ch}'] = fm.get_params('aperiodic_params', 'exponent')
    
            for band_freq, band in zip([[1,4],[4,8],[8,12],[13,30]], ['delta', 'theta', 'alpha', 'beta']):
                if Nss<=1:
                    res[f'{band}_dbs_{st}_{ch}'] = np.nan
                    res[f'{band}_rel_{st}_{ch}'] = np.nan
                else:
                    freq_ids = (freq>=band_freq[0])&(freq<band_freq[1])
                    bp_abs = np.nansum(psd[...,freq_ids], axis=-1)*dfreq
                    bp_abs_db = 10*np.log10(bp_abs)
                    bp_abs_db[np.isinf(bp_abs_db)] = np.nan
                    res[f'{band}_dbs_{st}_{ch}'] = np.nanmean(bp_abs_db)
                    res[f'{band}_rel_{st}_{ch}'] = np.nanmean(bp_abs/(np.nansum(psd, axis=-1)*dfreq))
    
    n2n3_ids = np.in1d(sleep_stages, [1,2])
    freq_ids = (freq>=0.5)&(freq<4)
    t = np.arange(len(sleep_stages))[n2n3_ids]*30/3600
    
    for chid, ch in zip(channel_ids, channels):
        delta_power = np.nansum(spec[n2n3_ids][:,chid][...,freq_ids], axis=-1)*dfreq
        delta_power_db = np.nanmean(10*np.log10(delta_power), axis=1)
        ids = (~np.isnan(t))&(~np.isnan(delta_power_db))&(~np.isinf(delta_power_db))
        if ids.sum()>5:
            res_ = linregress(t[ids], delta_power_db[ids])
            res[f'delta_slope_N2N3_{ch}'] = res_.slope
        else:
            res[f'delta_slope_N2N3_{ch}'] = np.nan
    
    #for ch in channels:
    #    res['beta_diff_WBSO_WAM_'+ch] = res['beta_dbs_WBSO_'+ch]-res['beta_dbs_WAM_'+ch]
    
    return res

# Extracts overnight spindle and SO coupling metrics from Luna `.db` files, including waveform features, SO-phase relationships, and spindle features 
def get_spindle_so(luna_db_paths, ch_groups, ch_names, Fs):
    """
7   SPINDLES       CH_F_SPINDLE_mysp --> NO, per spindle info
10  SPINDLES               CH_N_mysp --> NO, per SO info

9   SPINDLES               CH_F_mysp --> YES, overnight spindle summary stats
13  SPINDLES                 CH_mysp --> YES, overnight SO summary stats

5   SPINDLES         CH_F_PHASE_mysp --> YES, SOPL_CWT: SO phase vs spindle wavelet power
11  SPINDLES           CH_PHASE_mysp --> YES, SOPL_EEG: SO phase vs EEG
8   SPINDLES            CH_F_SP_mysp --> YES, SOTL_CWT: SO time vs spindle wavelet power
12  SPINDLES              CH_SP_mysp --> YES, SOTL_EEG: SO time vs EEG

6   SPINDLES        CH_F_RELLOC_mysp --> YES, IF within spindle, 5 bins

4   SPINDLES  CH_F_PHASE_RELLOC_mysp --> NO, IF by PHASE by RELLOC, too much, don't know how to compare
3   SPINDLES               CH_E_mysp --> NO, stats per epoch, not useful
    """
    so_phases = np.arange(10,360,20)
    sp_types = ['all', 'slow', 'fast']
    sp_rellocs = [1,2,3,4,5]
    so_samples = np.arange(-int(Fs), int(Fs)+1)
    
    res = {}
    cols = ['AMP', 'CDENS', 'CHIRP', 'COUPL_ANCHOR', 'COUPL_ANGLE', 'COUPL_MAG',
       'COUPL_OVERLAP', 'DENS', 'DISPERSION', 'DUR', 'FFT', 'ISA_M', 'ISA_S',
       'ISA_T', 'MINS', 'N', 'NE', 'NOSC', 'R_PHASE_IF', 'SEC_AMP', 'SEC_P2P',
       'SEC_TROUGH', 'SYMM', 'SYMM2', 'SYMM_AMP', 'SYMM_TROUGH', 'UDENS']
    for ch_g, ch_n in zip(ch_groups, ch_names):
        res_ = defaultdict(list)
        for ch in ch_g:
            if not os.path.exists(luna_db_paths[ch]):
                continue
            proj.import_db(luna_db_paths[ch])
            
            # Overnight spindle summary stats
            df = proj.table('SPINDLES', 'CH_F_mysp')
            if len(df)>0:
                for t in sp_types:
                    ids = df.mysp==t
                    nn = ids.sum()
                    for p in cols:
                        if nn>0 and p in df.columns:
                            res_[f'SP_{p}_{t}'].append( df[p][ids].iloc[0] )
                
            # Overnight SO summary stats
            df = proj.table('SPINDLES', 'CH_mysp')
            if len(df)>0:
                df = df.drop(columns=['ID', 'CH', 'mysp', 'SO'])
                df = pd.DataFrame(data=np.nanmean(df.values.astype(float), axis=0, keepdims=True), columns=df.columns)
                for p in df.columns:
                    res_[p].append( df[p].iloc[0] )
            
            # SO phase vs spindle wavelet power
            df = proj.table('SPINDLES', 'CH_F_PHASE_mysp')
            if len(df)>0:
                for t,p in product(sp_types, so_phases):
                    ids = (df.mysp==t)&(df.PHASE==p)
                    if ids.sum()>0:
                        res_[f'SP_WAVELET_{t}_AT_SO_PHASE{p:.0f}'].append( df.SOPL_CWT[ids].iloc[0] )
            df = proj.table('SPINDLES', 'CH_PHASE_mysp')
            if len(df)>0:
                for t,p in product(sp_types, so_phases):
                    ids = (df.mysp==t)&(df.PHASE==p)
                    if ids.sum()>0:
                        res_[f'SO_EEG_{t}_AT_SO_PHASE{p:.0f}'].append( df.SOPL_EEG[ids].iloc[0] )
                
            df = proj.table('SPINDLES', 'CH_F_SP_mysp')
            if len(df)>0:
                for t,p in product(sp_types, so_samples):
                    ids = (df.mysp==t)&(df.SP==p)
                    if ids.sum()>0:
                        res_[f'SP_WAVELET_{t}_AT_SO_SAMPLE{p:.0f}'].append( df.SOTL_CWT[ids].iloc[0] )
            df = proj.table('SPINDLES', 'CH_SP_mysp')
            if len(df)>0:
                for t,p in product(sp_types, so_samples):
                    ids = (df.mysp==t)&(df.SP==p)
                    if ids.sum()>0:
                        res_[f'SO_EEG_{t}_AT_SO_SAMPLE{p:.0f}'].append( df.SOTL_EEG[ids].iloc[0] )

            # IF within spindle, 5 bins
            df = proj.table('SPINDLES', 'CH_F_RELLOC_mysp')
            if len(df)>0:
                df['RELLOC'] = df.RELLOC.astype(int)
                for t,p in product(sp_types, sp_rellocs):
                    ids = (df.mysp==t)&(df.RELLOC==p)
                    if ids.sum()>0:
                        res_[f'SP_IF_{t}_AT_BIN{p}'].append( df.IF[ids].iloc[0] )

        res |= {f'{k}_{ch_n}':pd.Series(v).mean() for k,v in res_.items()}

    return res


# Identifies and removes the final continuous block of wake (stage 5) from the end of the sleep stage sequence.
def remove_final_wake(sleep_stages):
    sleep_stages2 = np.array(sleep_stages)
    sleep_stages2[np.isnan(sleep_stages2)] = 5
    stop = 0
    for k,l in groupby(sleep_stages2[::-1]):
        if k==5:
            stop = len(list(l))
        break
    stop = len(sleep_stages)-stop
    return np.arange(stop)

# Aggregates macrostructure, bandpower, and spindle/SO coupling features from SHHS subjects and exports to CSV for analysis
def main():
    spec_dir = 'data_spectrogram'
    edf_dir = 'clean_edf'
    sp_so_dir = 'detection_luna_q=0.69'
    epoch_time = 30
    stages_txt = ['N3', 'N2', 'N1', 'R', 'W']
    stages_num = [1,2,3,4,5]
    stages_txt2num = {k:v for k,v in zip(stages_txt, stages_num)}
    ch_groups = [['C3M2', 'C4M1']]
    ch_names2 = ['C']
    
    visit = 'shhs1'
    #df = pd.read_excel('MaYan/Full_subject_list.xlsx')
    #df = pd.read_excel('ARIC_Zach/mastersheet_ARIC.xlsx')
    #df = pd.read_excel('FHS_proteomics/mastersheet_FHS.xlsx')
    df = pd.read_excel('../shhs_mastersheet2.xlsx')
    df = df[df.visitnumber==visit].reset_index(drop=True)

    #idsss = np.array_split(np.arange(len(df)), 12)
    #idssss = idsss[kk]
    #for i in tqdm(idssss):
    for i in tqdm(range(len(df))):
        sid = str(df.nsrrid.iloc[i])
        #try:

        # get band power features
        path = os.path.join(spec_dir, f'spec_{visit}-{sid}.npz')
        if os.path.exists(path):
            with np.load(path, allow_pickle=True) as res:
                sleep_stages = res['sleep_stages']
                feat0 = get_macrostructures(sleep_stages)
                
                ids = remove_final_wake(sleep_stages) # since mostly artifact
                spec_db = res['spec_db'][ids].astype(float)
                spec_db[np.isinf(spec_db)] = np.nan
                spec = np.power(10, spec_db/10)
                freq = res['freq']
                artifact_indicator = res['artifact_indicator'][ids]
                sleep_stages = sleep_stages[ids]
                ch_names = np.char.strip(res['ch_names'])
                Fs = float(res['Fs'])
            feat1 = get_bandpower(spec, freq, sleep_stages, artifact_indicator, ch_names)
        else:
            feat0 = {}
            feat1 = {}
        
        # Get spindle/SO features
        sp_so_paths = {ch:os.path.join(sp_so_dir, f'luna_detection_{sid}_{ch}.db') for ch in ch_names}
        feat2 = get_spindle_so(sp_so_paths, ch_groups, ch_names2, Fs)

        feats = feat0|feat1|feat2
        for k,v in feats.items():
            if k not in df.columns:
                df[k] = np.nan
            df.loc[i, k] = v
            
        #if i%5==0:
        #    df.to_csv('features_eeg-SHHS.csv', index=False)
        
        #except Exception as ee:
        #    print(f'{pid}: {str(ee)}')
    
    df = df.drop(columns=['Channels','Frequencies','Units','EDFPath','AnnotPath'])
    #df.to_csv(f'features_eeg-SHHS-{kk}.csv.zip', index=False, compression='zip')
    df.to_csv('features_eeg-SHHS.csv.zip', index=False, compression='zip')
        

# Run code 
if __name__=='__main__':
    main()#int(sys.argv[1]))

