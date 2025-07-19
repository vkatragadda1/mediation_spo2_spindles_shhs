# Script detects spindles and slow oscillations using Luna package and performs relevant calculations
# Outputs data into CSV

# Load packages 
import os
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import lunapi as lp
proj = lp.proj()


# Adds non-overlapping detections from each EEG channel to ensure symmetric event representation across channels 
def add_symmetric_detection(df1, df2, ch1, ch2, col_ch, col_start, col_stop, col_other):
    df3 = []
    for i in range(len(df1)):
        start1, stop1 = df1.loc[i,col_start], df1.loc[i,col_stop]
        starts2, stops2 = df2[col_start].values, df2[col_stop].values
        if not (((starts2>=start1)&(starts2<=stop1)) | ((stops2>=start1)&(stops2<=stop1)) | ((starts2<=start1)&(stops2>=stop1))).any():
            df3.append({col_ch:ch2, col_start:start1, col_stop:stop1, col_other:df1.loc[i,col_other]})
    for j in range(len(df2)):
        start2, stop2 = df2.loc[j,col_start], df2.loc[j,col_stop]
        starts1, stops1 = df1[col_start].values, df1[col_stop].values
        if not (((starts1>=start2)&(starts1<=stop2)) | ((stops1>=start2)&(stops1<=stop2)) | ((starts1<=start2)&(stops1>=stop2))).any():
            df3.append({col_ch:ch1, col_start:start2, col_stop:stop2, col_other:df2.loc[j,col_other]})
    df_res = []
    if len(df1)>0: df_res.append(df1)
    if len(df2)>0: df_res.append(df2)
    if len(df3)>0: df_res.append(pd.DataFrame(df3))
    if len(df_res)>0:
        df_res = pd.concat(df_res, axis=0, ignore_index=True)
    return df_res
    
# Detects slow/fast spindles and their coupling with slow oscillations using Luna
# Enforces symmetric detection and extracts features for spindles/slow oscillations 
def get_spindle_so(sid, edf_path, ch_names, q, save_path):
    p = proj.inst( str(sid) )
    p.attach_edf(edf_path)
    
    # First detect m-spindles, then save as .annot
    annot_path = f'tmp_{sid}.annot'
    df_msps = []
    for ch in ch_names:
        cmd = [
            'FREEZE tag=original',
            f'SIGNALS keep={ch}',
            'EPOCH',
            f'MASK ifnot=N2,N3',
            f'MASK mask-if=edf_annot[artifact_{ch}]',
            'RE',
            f'SPINDLES sig={ch} fc-lower=10.5 fc-upper=15.5 fc-step=0.5 cycles=7 collate-within-channel per-spindle q={q}',
            'THAW tag=original', ]
        p.eval(' & '.join([x.split('%')[0] for x in cmd]))
        
        df_msp = p.table('SPINDLES', 'CH_MSPINDLE')
        print(ch, df_msp)
        if df_msp is None or len(df_msp)==0:
            df_ = pd.DataFrame(data=np.ones((0,4)), columns=['channel','MSP_START','MSP_STOP','MSP_F'])
            df_msps.append(df_)
        else:
            df_msp['channel'] = ch
            #df_msp = df_msp.sort_values('MSP_START')
            df_msps.append(df_msp[['channel','MSP_START','MSP_STOP','MSP_F']])
        p.refresh()
        
    # Add symmetric channel detections
    ch_pairs = [[0,1]]#, [2,3], [4,5]]
    df_msps_sym = []
    for ch in ch_pairs:
        res = add_symmetric_detection(df_msps[ch[0]], df_msps[ch[1]], ch_names[ch[0]], ch_names[ch[1]], 'channel', 'MSP_START', 'MSP_STOP', 'MSP_F')
        if len(res)>0:
            df_msps_sym.append(res)
            
    df_sp = []
    df_so = []
    if len(df_msps_sym)>0:
        df_msp = pd.concat(df_msps_sym, axis=0, ignore_index=True)
        df_msp['class'] = 'sp_all'
        df_msp['MSP_START'] = df_msp.MSP_START.astype(float)
        df_msp['MSP_STOP'] = df_msp.MSP_STOP.astype(float)
        df_msp2 = df_msp.copy()
        df_msp2.loc[df_msp2.MSP_F<13, 'class'] = 'sp_slow'
        df_msp2.loc[df_msp2.MSP_F>=13, 'class'] = 'sp_fast'
        df_msp = pd.concat([df_msp, df_msp2], axis=0)
        df_msp['instance'] = '.'
        df_msp['meta'] = '.'
        df_msp[['class', 'instance', 'channel', 'MSP_START', 'MSP_STOP', 'meta']].to_csv(annot_path, float_format='%.4f', index=False, header=None, sep='\t')
    
        # Then compute features as .db file
        for ch in ch_names:
            save_path2 = save_path.replace('.db', f'_{ch}.db')
            cmd = [f'SIGNALS keep={ch}',
                'EPOCH',
                f'MASK ifnot=N2,N3',
                f'MASK mask-if=edf_annot[artifact_{ch}]',
                'RE',]
            if (df_msp['class']=='sp_slow').sum()>0:
                cmd.extend([
                'TAG mysp/slow',
                f'SPINDLES precomputed=sp_slow per-spindle if so mag=3 verbose-coupling tl={ch} q={q/2}',])
            if (df_msp['class']=='sp_fast').sum()>0:
                cmd.extend([
                'TAG mysp/fast',
                f'SPINDLES precomputed=sp_fast per-spindle if so mag=3 verbose-coupling tl={ch} q={q/2}',])
            cmd.extend([
                'TAG mysp/all',
                f'SPINDLES precomputed=sp_all per-spindle if so mag=3 verbose verbose-coupling tl={ch} q={q/2}',])
            subprocess.run(['luna', edf_path, 'annot-file='+annot_path, '-o', save_path2,
                '-s', ' & '.join([x.split('%')[0] for x in cmd])])
            proj.import_db(save_path2)
            
            df_sp_ = proj.table('SPINDLES', 'CH_F_SPINDLE_mysp')
            if len(df_sp_)>0:
                df_sp.append( df_sp_[df_sp_.mysp=='all'].drop(columns='mysp') )
            df_so_ = proj.table('SPINDLES', 'CH_N_mysp')
            if len(df_so_)>0:
                df_so.append( df_so_.drop(columns='mysp') )
            
        os.remove(annot_path)
        
    if len(df_sp)>0:
        df_sp = pd.concat(df_sp, axis=0, ignore_index=True)
    else:
        cols = ['ID', 'CH', 'F', 'SPINDLE', 'AMP', 'ANCHOR', 'CHIRP', 'DUR', 'FFT',
       'FRQ', 'FRQ1', 'FRQ2', 'FWHM', 'IF', 'ISA', 'MAXSTAT', 'MEANSTAT',
       'NOSC', 'PASS', 'PEAK', 'PEAK_SP', 'Q', 'SO_NEAREST', 'SO_NEAREST_NUM',
       'SO_PHASE_ANCHOR', 'START', 'START_SP', 'STOP', 'STOP_SP', 'SYMM',
       'SYMM2', 'TROUGH', 'TROUGH_SP']
        df_sp = pd.DataFrame(data=np.ones((0,len(cols))), columns=cols)
    if len(df_so)>0:
        df_so = pd.concat(df_so, axis=0, ignore_index=True)
    else:
        cols = ['ID', 'CH', 'N', 'DOWN_AMP', 'DOWN_IDX', 'DUR', 'DUR1', 'DUR2',
       'DUR_CHK', 'P2P_AMP', 'SLOPE', 'START', 'START_IDX', 'STOP', 'STOP_IDX',
       'TRANS', 'TRANS_FREQ', 'UP_AMP', 'UP_IDX']
        df_so = pd.DataFrame(data=np.ones((0,len(cols))), columns=cols)
    return df_sp, df_so

# Runs spindle and SO detection on cleaned EDFs for all SHHS subjects
# Exports features and CSV annotations per subject.
def main():
    visit = 'shhs1'
    df = pd.read_excel('../shhs_mastersheet2.xlsx')
    df = df[df.visitnumber==visit].reset_index(drop=True)
    
    q = 0.69 
    
    visit = 'shhs1'
    detection_dir = f'detection_luna_q={q}'
    os.makedirs(detection_dir, exist_ok=True)
    edf_dir = 'clean_edf'
    
    ch_names = ['C3M2', 'C4M1']
    for i in tqdm(range(len(df))):
        sid = str(df.nsrrid.iloc[i])
        save_path = os.path.join(detection_dir, f'luna_detection_{sid}.db')
        if all([os.path.exists(save_path.replace('.db',f'_{x}.db')) for x in ch_names]):
            continue
            
        edf_path = os.path.join(edf_dir, f'{visit}-{sid}-clean.edf')
        if not os.path.exists(edf_path):
            continue
        df_sp, df_so = get_spindle_so(sid, edf_path, ch_names, q, save_path)
        
        df_sp['DESC'] = [f'SP q{df_sp.Q.iloc[ii]:.1f} f{df_sp.FFT.iloc[ii]:.1f}@@{df_sp.CH.iloc[ii]}' for ii in range(len(df_sp))]
        df_sp = df_sp[['START', 'STOP', 'DUR', 'CH', 'Q', 'DESC']]
        df_sp.to_csv(os.path.join(detection_dir, f'spindle_{sid}-annotation.csv'), index=False)
        
        df_so['DESC'] = [f'SO@@{df_so.CH.iloc[ii]}' for ii in range(len(df_so))]
        df_so = df_so[['START', 'STOP', 'DUR', 'CH', 'DESC']]
        df_so.to_csv(os.path.join(detection_dir, f'so_{sid}-annotation.csv'), index=False)
        

# Runs code
if __name__=='__main__':
    main()

