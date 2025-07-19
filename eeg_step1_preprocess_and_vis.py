# Step 1: For pre-processing and visualization of PSG data 

# Load packages 
from itertools import groupby
import datetime
import os
import numpy as np
import pandas as pd
from scipy.stats import mode
import mne
from tqdm import tqdm
import pyedflib
import xml2dict
from scipy.signal import peak_prominences
from neurokit2 import ecg_peaks
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn
seaborn.set_style('ticks')

# Script uses Luna software for sleep structure calculations 
import lunapi as lp
proj = lp.proj()

# Function to plot data 
# Optional inputs: lights_off_time, lights_on_time
def plot(age, sex, sleep_stages, spec_db, freq, ch_names, start_time, artifact_indicator=None, save_path=None):
    # Initiate plot
    # lights_off = (lights_off_time-start_time).total_seconds()/3600
    # lights_on = (lights_on_time-start_time).total_seconds()/3600
    tt = np.arange(len(sleep_stages))*30/3600
    tmin = tt.min() #lights_off 
    tmax = tt.max() #lights_on
    xticks = np.arange(0, int(tmax)+1)
    xticklabels = [(start_time+datetime.timedelta(hours=int(x))).strftime('%H:%M') for x in xticks]

    plt.close()
    fig = plt.figure(figsize=(17,9))
    gs = GridSpec(1+len(ch_names), 1, height_ratios=[1]+[3]*len(ch_names))
    
    # Plot hypnogram
    ax = fig.add_subplot(gs[0]); ax0 = ax
    ax.step(tt, sleep_stages, where='post', color='k')
    #ax.axvline(lights_off, c='r', ls='--', lw=2)
    #ax.axvline(lights_on, c='r', ls='--', lw=2)
    ax.text(0.005, 1.01, f'{age:.0f}{sex}', ha='left', va='bottom', transform=ax.transAxes)#, fontsize=12)
    #, diag={diagnosis}\nmed={med}'
    ax.set_yticks([1,2,3,4,5], labels=['N3','N2','N1','R','W'])
    ax.set_ylim(0.9,5.1)
    ax.yaxis.grid(True)
    ax.set_xlim(tmin, tmax)
    seaborn.despine()
    plt.setp(ax.get_xticklabels(), visible=False)

    # Plot spectrograms
    ypad = 0.5
    for chi, ch in enumerate(ch_names):
        ax = fig.add_subplot(gs[1+chi], sharex=ax0)
        ax.imshow(spec_db[:,chi].T, aspect='auto', origin='lower',
            cmap='turbo', vmin=-5, vmax=20,
            extent=(tmin, tmax, freq.min(), freq.max()))
        if artifact_indicator is not None:
            es = artifact_indicator[:,chi].astype(float)
            es[es==0] = np.nan
            ax.step(tt, es*freq.max()+ypad/2, where='post', color='k', lw=3)
        #ax.axvline(lights_off, c='r', ls='--', lw=2)
        #ax.axvline(lights_on, c='r', ls='--', lw=2)
        ax.set_xlim(tmin, tmax)
        ax.set_ylim(freq.min(), freq.max()+ypad)
        ax.set_ylabel(ch, rotation=0, ha='right')
        seaborn.despine()
        if chi==len(ch_names)-1:
            ax.set_xticks(xticks, labels=xticklabels)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)

    """
    # plot spo2
    tt_spo2 = np.arange(len(spo2))/Fs/3600
    ax = fig.add_subplot(gs[1+len(eeg_ch_names2)], sharex=ax0)
    ax.plot(tt_spo2, spo2, c='k', lw=0.5)
    ax.set_ylabel('SpO2 (%)')
    ax.set_yticks([80,90,100])
    ax.set_ylim(80,100)
    ax.yaxis.grid(True)
    ax.set_xlim(tmin, tmax)
    ax.set_xticks(xticks, labels=xticklabels)
    seaborn.despine()
    """
    # Save and show figure 
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')

# Preprocesses ECG signal to identify usable segments
# Determines if signal polarity should be flipped based on R-peaks and heart rate 
def preprocess_ecg(ecg_, Fs, line_freq=60):
    # Returns True/False to indicate if ecg should be flipped
    # Returns None if ecg has: (1) <10 minutes non-nan signal OR (2) overall heart rate is abnormal
    
    # Find the longest non-nan and non-flat part
    nan_mask = np.isnan(ecg_)|np.isinf(ecg_)|(np.abs(ecg_)>5000)
    cc = 0; maxll = 0; start = -1
    for k,l in groupby(nan_mask):
        ll = len(list(l))
        if (not k) and ecg_[cc:cc+ll].std()>1 and ll>maxll:
            maxll = ll
            start = cc
        cc += ll
    if start<0:
        return None
    ecg_seg = ecg_[start:start+maxll]
    if len(ecg_seg)<Fs*30:
        return None
    # Find middle two hours, so that the signal is short
    if len(ecg_seg)>Fs*7200:
        start = int(len(ecg_seg)//2-Fs*3600)
        end = int(len(ecg_seg)//2+Fs*3600)
        ecg_seg = ecg_seg[start:end]
    if line_freq<Fs/2:
        ecg_seg = mne.filter.notch_filter(ecg_seg, Fs, line_freq, verbose=False)
    ecg_seg = mne.filter.filter_data(ecg_seg, Fs, 5, 70 if 70<Fs/2 else None, verbose=False)
    try:
        #rpeaks1 = detect_heartbeats(ecg_seg, Fs)
        #rpeaks2 = detect_heartbeats(-ecg_seg, Fs)
        rpeaks1 = ecg_peaks(ecg_seg, sampling_rate=Fs)[1]['ECG_R_Peaks']
        rpeaks2 = ecg_peaks(-ecg_seg, sampling_rate=Fs)[1]['ECG_R_Peaks']
    except Exception as ee:
        print(str(ee))
        return None
    
    peakness1 = np.mean(peak_prominences(ecg_seg, rpeaks1)[0])
    peakness2 = np.mean(peak_prominences(-ecg_seg, rpeaks2)[0])
    if peakness1<peakness2:
        hr = len(rpeaks2)/(len(ecg_seg)/Fs/60)
    else:
        hr = len(rpeaks1)/(len(ecg_seg)/Fs/60)
    if 30<hr<100:
        return peakness1<peakness2
    else:
        return None

# Loads EEG/ECG signals and sleep stage annotations from SHHS EDF and XML files
# Returns signal data, sleep stage array, annotations, and metadata 
def read_shhs_dataset(edf_path, annot_path, eeg=True, ecg=True):
    edf = mne.io.read_raw_edf(edf_path, verbose=False)
    ch_names = edf.ch_names
    start_time = edf.info['meas_date'].replace(tzinfo=None)
    params = {'start_time':start_time}
    
    ch_names2 = []
    ch_names3 = []
    if eeg:
        if 'EEG2' in ch_names:
            eeg_ch_names = ['EEG2', 'EEG']
        elif 'EEG 2' in ch_names:
            eeg_ch_names = ['EEG 2', 'EEG']
        elif 'EEG(SEC)' in ch_names:
            eeg_ch_names = ['EEG(SEC)', 'EEG']
        elif 'EEG(sec)' in ch_names:
            eeg_ch_names = ['EEG(sec)', 'EEG']
        elif 'EEG sec' in ch_names:
            eeg_ch_names = ['EEG sec', 'EEG']
        else:
            raise ValueError(f'EEG channels not found within {ch_names}')
        assert np.in1d(eeg_ch_names, ch_names).all()
        ch_names2.extend(eeg_ch_names)
        ch_names3.extend(['C3M2', 'C4M1'])
        
    if ecg:
        ecg_ch_name = 'ECG'
        assert ecg_ch_name in ch_names
        ch_names2.append(ecg_ch_name)
        ch_names3.append('ECG')
    
    edf = mne.io.read_raw_edf(edf_path, verbose=False, exclude=[x for x in ch_names if x not in ch_names2])
    signals = edf.get_data(picks=ch_names2)*1e6
    Fs = edf.info['sfreq']
    params['Fs'] = Fs
    params['ch_names_edf'] = ch_names2
    params['ch_names'] = ch_names3
    
    with open(annot_path, 'r') as ff:
        annot = xml2dict.parse(ff.read())
    
    epoch_time = float(annot['PSGAnnotation']['EpochLength'])
    assert epoch_time==30
    annot = pd.DataFrame(annot['PSGAnnotation']['ScoredEvents']['ScoredEvent'])
    annot['Start'] = annot.Start.astype(float)
    annot['Duration'] = annot.Duration.astype(float)
    start_time_annot = annot.ClockTime[annot.EventConcept.str.contains('start\s*time', case=False)]
    if len(start_time_annot)>0:
        assert start_time.strftime('%H.%M.%S') in str(start_time_annot.iloc[0])
    
    mapping = {'Wake|0':5., 'Stage 1 sleep|1':3., 'REM sleep|5':4.,
            'Stage 2 sleep|2':2., 'Stage 3 sleep|3':1., 'Stage 4 sleep|4':1.,
            'Unscored|9':np.nan, 'Movement|6':np.nan}
    sleep_stages = np.zeros(signals.shape[1], dtype=float)+np.nan
    annot2 = annot[annot.EventType=='Stages|Stages'].reset_index(drop=True)
    for i,r in annot2.iterrows():
        start = max(0, int(round(r.Start*Fs)))
        end   = min(len(sleep_stages), int(round((r.Start+r.Duration)*Fs)))
        if start<end:
            sleep_stages[start:end] = mapping[r.EventConcept]
    
    return signals, sleep_stages, annot, params

# Writes EEG/ECG signals and sleep stage annotations to an EDF+ file with proper channel headers and optional additional annotations.
def write_edfplus(signals, sleep_stages_by_epoch, Fs, output_path, ch_names, start_time, df_annot=None):
    """
    """
    ch_names = [x.replace('-', '').upper() for x in ch_names]

channel_info = [
        {'label': ch,
         'dimension': 'uV',
         'sample_rate': Fs,
         'physical_max': 32767,
         'physical_min': -32768,
         'digital_max': 32767,
         'digital_min': -32768,
         'transducer': 'E',
         'prefilter': ''}
    for ch in ch_names]

    mapping = {5:'W', 4:'R', 3:'N1', 2:'N2', 1:'N3'}
    
    annot = []
    for si, ss in enumerate(sleep_stages_by_epoch):
        annot.append({'start':si*30, 'duration':30, 'description':mapping.get(ss,'?')})
    if df_annot is not None:
        for ii, r in df_annot.iterrows():
            annot.append({'start':r.start, 'duration':r.duration, 'description':r.description})
    annot = pd.DataFrame(annot).sort_values('start', ignore_index=True)

    with pyedflib.EdfWriter(output_path, len(signals), file_type=pyedflib.FILETYPE_EDFPLUS) as ff:
        ff.setSignalHeaders(channel_info)
        ff.setStartdatetime(start_time)
        ff.writeSamples(signals)
        for i,r in annot.iterrows():
            ff.writeAnnotation(r.start, r.duration, r.description)

# Filters, resamples, and removes artifacts from EEG/ECG signals
# Saves cleaned data with annotations using Luna’s artifact detection pipeline.
def preprocess_luna(sid, eeg, ecg, sleep_stages_epoch, annot, Fs, line_freq, eeg_ch_names, ecg_ch_name, start_time, save_path, epoch_time=30):
    epoch_size = int(round(epoch_time*Fs))
    
    # Filtering
    eeg = eeg-np.nanmean(eeg,axis=-1,keepdims=True)
    eeg[np.isnan(eeg)] = 0
    if line_freq<Fs/2:
        eeg = mne.filter.notch_filter(eeg, Fs, line_freq, verbose=False)
    eeg = mne.filter.filter_data(eeg, Fs, 0.3, 35 if 35<Fs/2 else None, verbose=False)
    if ecg is not None:
        ecg = ecg-np.nanmean(ecg)
        ecg[np.isnan(ecg)] = 0
        if line_freq<Fs/2:
            ecg = mne.filter.notch_filter(ecg, Fs, line_freq, verbose=False)
        ecg = mne.filter.filter_data(ecg, Fs, 0.3, 70 if 70<Fs/2 else None, verbose=False)
        flip_ecg = preprocess_ecg(ecg, Fs)
    
    # Resampling
    newFs = 125
    if Fs!=newFs:
        eeg = mne.filter.resample(eeg, down=Fs/newFs, npad='auto')
        if ecg is not None:
            ecg = mne.filter.resample(ecg, down=Fs/newFs, npad='auto')
        Fs = newFs
        epoch_size = int(round(epoch_time*Fs))
    epoch_num = eeg.shape[1]//epoch_size
    
    # Save edf+
    if ecg is not None and flip_ecg is not None:
        if flip_ecg:
            ecg = -ecg
        signals = np.vstack([eeg,ecg])
        ch_names = eeg_ch_names+[ecg_ch_name]
    else:
        signals = eeg
        ch_names = eeg_ch_names
    
    cwd = os.getcwd()
    edf_path_tmp = os.path.join(cwd, f'tmp_{sid}.edf')
    assert ' ' not in edf_path_tmp
    write_edfplus(signals, sleep_stages_epoch, Fs, edf_path_tmp, ch_names, start_time)

    # Run luna artifact code
    p = proj.inst( sid )
    p.attach_edf(edf_path_tmp)
    
    edf_path_tmp2 = os.path.join(cwd, f'tmp_{sid}_2.edf')
    assert ' ' not in edf_path_tmp2
    eeg_ch_names_txt = ','.join(eeg_ch_names)
    cmd = [
        'EPOCH',
        f'uV sig={eeg_ch_names_txt}  % make sure the unit is uV for movement artifact threshold',
        f'CHEP-MASK chep-th=3 max=500,0.001 flat=0.5,0.1 sig={eeg_ch_names_txt}',
        'CHEP dump',]
    if ecg is not None:
        cmd.append( f'SUPPRESS-ECG sig={eeg_ch_names_txt} ecg={ecg_ch_name}' )
    cmd.extend( [
        f'SIGNALS keep={eeg_ch_names_txt}  % drop ECG',
        'MASK none  % to keep same length as original',
        f'WRITE edf={edf_path_tmp2}', ] )

    p.eval(' & '.join([x.split('%')[0] for x in cmd]))

    dfa = p.table('CHEP','CH_E')
    dfa = dfa[np.in1d(dfa.CH, eeg_ch_names)].reset_index(drop=True)
    assert len(dfa)==epoch_num*len(eeg_ch_names)
    dfa = dfa[dfa.CHEP==1].reset_index(drop=True)
    artifact_indicator = np.zeros((epoch_num, len(eeg_ch_names)), dtype=bool)
    for ii, r in dfa.iterrows():
        artifact_indicator[r.E-1, eeg_ch_names.index(r.CH)] = True
    
    # Re-save luna output edf with artifact_indicator as annotation
    edf_path_tmp2 = edf_path_tmp2+'.edf'##lunapi bug
    edf = mne.io.read_raw_edf(edf_path_tmp2, preload=False, verbose=False)
    eeg = edf.get_data(picks=eeg_ch_names)*1e6
    eeg = eeg-np.nanmean(eeg,axis=-1,keepdims=True)
    
    dfa2 = dfa.drop(columns=['ID', 'CHEP']).rename(columns={'E':'start', 'CH':'description'})
    dfa2['start'] = (dfa2.start-1)*epoch_time
    dfa2['description'] = 'artifact_'+dfa2.description
    dfa2['duration'] = epoch_time
    write_edfplus(eeg, sleep_stages_epoch, Fs, save_path, eeg_ch_names, start_time, df_annot=dfa2)

    os.remove(edf_path_tmp)
    os.remove(edf_path_tmp2)
    return eeg, artifact_indicator, Fs

# Iterates through SHHS files to preprocess EEG/ECG data, detect artifacts, compute spectrograms, and save cleaned EDFs and spectral features 
def main():
    visit = 'shhs1'
    """
    df = pd.read_excel('MaYan/Full_subject_list.xlsx')
    df_lights = pd.read_excel('MaYan/Light_out_info_all.xlsx')
    df = df.merge(df_lights, on='nsrrid', how='left', validate='1:1')
    assert df.StLOutP.notna().all()
    df_ch = pd.read_excel('../shhs_mastersheet2.xlsx')
    df_ch = df_ch[df_ch.visitnumber==visit].reset_index(drop=True)
    df = df.merge(df_ch[['nsrrid','EDFPath','AnnotPath']], on='nsrrid', how='left', validate='1:1')
    df2 = pd.read_csv(f'/data/haoqisun/dataset_SHHS/{visit}-dataset-0.20.0.csv')
    df2 = df2.rename(columns={'age_s1':'Age', 'gender':'Sex'})
    df2['Sex'] = df2.Sex.astype(str)
    df2.loc[df2.Sex=='1', 'Sex'] = 'M'
    df2.loc[df2.Sex=='2', 'Sex'] = 'F'
    df = df.merge(df2[['nsrrid','Age','Sex']], on='nsrrid', how='left', validate='1:1')
    N_plot = 100
    
    """
    #df = pd.read_excel('ARIC_Zach/mastersheet_ARIC.xlsx')
    #df = pd.read_excel('FHS_proteomics/mastersheet_FHS.xlsx')
    #print(df.shape)
    df_ch = pd.read_excel('../shhs_mastersheet2.xlsx')
    df_ch = df_ch[df_ch.visitnumber==visit].reset_index(drop=True)
    #df = df.merge(df_ch[['nsrrid','EDFPath','AnnotPath']], on='nsrrid', how='inner', validate='1:1')
    df = df_ch
    print(df.shape)
    #df_lights = pd.read_excel('MaYan/Light_out_info_all.xlsx')
    #df = df.merge(df_lights, on='nsrrid', how='inner', validate='1:1')
    #print(df.shape)
    N_plot = 0

    figure_dir = 'figures_spectrogram'
    os.makedirs(figure_dir, exist_ok=True)
    spec_dir = 'data_spectrogram'
    os.makedirs(spec_dir, exist_ok=True)
    clean_edf_dir = 'clean_edf'
    os.makedirs(clean_edf_dir, exist_ok=True)
    epoch_time = 30
    line_freq = 60
    
    for i in tqdm(range(len(df))):
        sid = str(df.nsrrid.iloc[i])
        #save_path1 = os.path.join(figure_dir, f'{visit}-{sid}-original.png')
        save_path2 = os.path.join(figure_dir, f'{visit}-{sid}.png')
        save_path3 = os.path.join(spec_dir, f'spec_{visit}-{sid}.npz')
        clean_edf_path = os.path.join(clean_edf_dir, f'{visit}-{sid}-clean.edf') 
        if os.path.exists(save_path3) and os.path.exists(clean_edf_path):
            aa = np.load(save_path3)
            if 'Fs' in aa and aa['freq'].max()>20:
                continue

        #age = df.Age.iloc[i]
        #sex = df.Sex.iloc[i]
        edf_path = df.EDFPath.iloc[i]
        annot_path = df.AnnotPath.iloc[i]
        if pd.isna(edf_path) or pd.isna(annot_path):
            continue
        #lights_off_epoch = max(0,df.StLOutP.iloc[i]-1) # inclusive

        # Decide if ECG must be flipped 
        signals, sleep_stages, annot, params = read_shhs_dataset(edf_path, annot_path, ecg=True)
        eeg = signals[:2]
        ecg = signals[2]
        Fs = params['Fs']
        start_time = params['start_time']
        eeg_ch_names = params['ch_names'][:2]
        ecg_ch_name = params['ch_names'][2]
        
        #pad = int(round(lights_off_epoch*30*Fs))
        #eeg = eeg[:,pad:]
        #ecg = ecg[pad:]
        #sleep_stages = sleep_stages[pad:]
        #TODO annot = 
        #start_time += datetime.timedelta(seconds=float(lights_off_epoch*30))
        #TODO align to epochs
        
        # Segment EEG into epochs
        #_, sleep_stages2, epoch_start_ids, spec_db, freq = segment_eeg(eeg, sleep_stages, Fs)
        epoch_size = int(round(30*Fs))
        epoch_start_ids = np.arange(0,eeg.shape[1]-epoch_size+1,epoch_size)
        sleep_stages2 = np.array(sleep_stages)
        sleep_stages2[np.isnan(sleep_stages2)] = -1
        sleep_stages2 = mode(np.array([sleep_stages[x:x+epoch_size] for x in epoch_start_ids]), axis=1, keepdims=False).mode
        sleep_stages2[sleep_stages2==-1] = np.nan
        
        # Plot
        #spec_db = np.array([np.nanmean(spec_db[:,x],axis=1) for x in spec_avg_ids]).transpose(1,0,2)
        #if i<N_plot and not os.path.exists(save_path1):
        #    plot(age, sex, sleep_stages2, spec_db, freq, eeg_ch_names,
        #        start_time, lights_off_time, lights_on_time,
        #        save_path=save_path1)
        eeg, artifact_indicator, Fs = preprocess_luna(sid,
            eeg, ecg, sleep_stages2, annot, Fs, line_freq,
            eeg_ch_names, ecg_ch_name, start_time,
            clean_edf_path)

        # Segment EEG and plot
        #_, _, epoch_start_ids, spec_db, freq = segment_eeg(eeg, None, Fs)
        #spec_db = np.array([np.nanmean(spec_db[:,x],axis=1) for x in spec_avg_ids]).transpose(1,0,2)
        epoch_size = int(round(30*Fs))
        epoch_start_ids = np.arange(0, eeg.shape[1]-epoch_size+1, epoch_size)
        epochs = np.array([eeg[:,x:x+epoch_size] for x in epoch_start_ids])
        spec, freq = mne.time_frequency.psd_array_multitaper(epochs, Fs, fmin=0.1, fmax=30, bandwidth=0.5, normalization='full', n_jobs=None, verbose=False)
        spec_db = 10*np.log10(spec)
        spec_db[np.isinf(spec_db)] = np.nan
        if i<N_plot and not os.path.exists(save_path2):
            plot(age, sex, sleep_stages2, spec_db, freq, eeg_ch_names,
                start_time,# lights_off_time, lights_on_time,
                artifact_indicator, save_path=save_path2)

        np.savez_compressed(save_path3,
            spec_db=spec_db[...,::2].astype('float16'), freq=freq[::2],
            sleep_stages=sleep_stages2, artifact_indicator=artifact_indicator,
            start_time=start_time, epoch_start_seconds=epoch_start_ids/Fs,
            #lights_off_time=lights_off_time, lights_on_time=lights_on_time,
            ch_names=eeg_ch_names, Fs=Fs,
            )

# Run code 
if __name__=='__main__':
    main()

