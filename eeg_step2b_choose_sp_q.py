# Script evaluates spindle quality and detection performance 

# Load packages 
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import mne
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix, cohen_kappa_score, f1_score
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import seaborn as sns
sns.set_style('ticks')

# Plots raw and filtered spindle data with manual labeling via key press
# Saves plot and label data 
def plot_spindle(eeg, eeg_f, start_idx, stop_idx, Fs, q, pid, sp_idx, save_path, save_path2):
    tt = np.arange(len(eeg))/Fs
    
    plt.close()
    fig = plt.figure(figsize=(8,6))
    
    ax = fig.add_subplot(111)
    ax.plot(tt, eeg, c='k', lw=1)
    ax.plot(tt, eeg_f-80, c='k', lw=1)
    ax.axvline(tt[start_idx], c='r', lw=1, ls='--')
    ax.axvline(tt[stop_idx], c='r', lw=1, ls='--')
    ax.set_xticks(np.arange(int(tt.max())+1))
    ax.xaxis.grid(True)
    #ax.text(0.01, 0.99, txt, ha='left', va='top', transform=ax.transAxes)    
    #ax.axis('off')
    ax.set_ylim(-170,120)
    ax.set_xlim(tt.min(), tt.max())
    sns.despine()
    
    def on_press(event):
        if os.path.exists(save_path2):
            df = pd.read_csv(save_path2)
            df = pd.concat([df, pd.DataFrame(data={'PID':[pid], 'SpindleIdx':[sp_idx], 'Q':[q], 'ManualLabel':[int(event.key)]})], axis=0)
        else:
            df = pd.DataFrame(data={'PID':[pid], 'SpindleIdx':[sp_idx], 'Q':[q], 'ManualLabel':[int(event.key)]})
        
        df.to_csv(save_path2, index=False)
        event.canvas.figure.savefig(save_path, bbox_inches='tight')
        plt.close(event.canvas.figure)

    fig.canvas.mpl_connect('key_press_event', on_press)
    
    plt.tight_layout()
    plt.show()


# Evaluates and visualizes spindle detection performance using AUROC, AUPRC, confusion matrix, and optimal threshold selection 
def plot_perf(y, q, save_path=None):
    auroc = roc_auc_score(y, q)
    fpr, tpr, tt = roc_curve(y, q)
    idx = np.argmin(fpr**2+(1-tpr)**2)
    print(idx)
    #idx = np.argmax(tpr-fpr)
    best_q = tt[idx]
    best_fpr = fpr[idx]
    best_tpr = tpr[idx]
    print(f'best_q = {best_q}')
    
    auprc = average_precision_score(y, q)
    precision, recall, tt = precision_recall_curve(y, q)
    idx = np.argmin(np.abs(tt-best_q))
    best_precision = precision[idx]
    best_recall = recall[idx]

    yp = (q>best_q).astype(int)
    cf = confusion_matrix(y, yp)
    ck = cohen_kappa_score(y, yp)
    f1 = f1_score(y, yp)
    print(cf)
    print(f'F1 = {f1}, Cohen\'s kappa = {ck}')
    bins = np.arange(q.min(), q.max()+0.1, 0.1)
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    plt.close()
    fig = plt.figure(figsize=(9.7*1.1,6*1.15))
    gs = fig.add_gridspec(2,3)
    
    ax = fig.add_subplot(gs[0,:])
    ax.hist(q[y==0], bins=bins, color='b', alpha=0.35, label='human: not spindle')
    ax.hist(q[y==1], bins=bins, color='r', alpha=0.35, label='human: spindle')
    ax.axvline(best_q, color='r', lw=2, ls='--')
    ax.text(best_q+0.02, 45, f'optimal quality index cutoff is {best_q:.2f}', color='r')
    ax.legend(frameon=False, loc='upper right')
    ax.set_ylabel('count')
    ax.set_xlabel('quality index')
    ax.set_xlim(0, q.max())
    ax.yaxis.grid(True)
    ax.text(-0.07, 1, 'a', ha='right', va='top', transform=ax.transAxes, weight='bold')
    sns.despine()
    
    ax = fig.add_subplot(gs[1,0])
    ax.plot([0,1], [0,1], c='k', lw=1, ls='--')
    ax.plot(fpr, tpr, c='k', lw=1.5)
    ax.plot([0, best_fpr, best_fpr], [best_tpr, best_tpr, 0], lw=1, ls='--', c='r')
    ax.scatter([best_fpr], [best_tpr], s=45, c='r')
    ax.text(best_fpr+0.03, 0.01, f'{best_fpr:.2f}', ha='left', va='bottom', color='r')
    ax.text(0.01, best_tpr+0.01, f'{best_tpr:.2f}', ha='left', va='bottom', color='r')
    ax.text(0.01, 0.99, f'AUROC = {auroc:.2f}', ha='left', va='top', color='k')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xticks([0,0.25,0.5,0.75,1], labels=['0','0.25','0.5','0.75','1'])
    ax.set_yticks([0,0.25,0.5,0.75,1], labels=['0','0.25','0.5','0.75','1'])
    ax.grid(True)
    ax.text(-0.28, 1.02, 'b', ha='right', va='top', transform=ax.transAxes, weight='bold')
    sns.despine()
    
    ax = fig.add_subplot(gs[1,1])
    #ax.plot([0,1], [0,1], c='r', lw=1)
    ax.plot(recall, precision, c='k', lw=1.5)
    ax.plot([0, best_recall, best_recall], [best_precision, best_precision, 0], lw=1, ls='--', c='r')
    ax.scatter([best_recall], [best_precision], s=45, c='r')
    ax.text(best_recall+0.03, 0.01, f'{best_recall:.2f}', ha='left', va='bottom', color='r')
    ax.text(0.01, best_precision+0.01, f'{best_precision:.2f}', ha='left', va='bottom', color='r')
    ax.text(0.99, 0.99, f'AUPRC = {auprc:.2f}', ha='right', va='top', color='k')
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xticks([0,0.25,0.5,0.75,1], labels=['0','0.25','0.5','0.75','1'])
    ax.set_yticks([0,0.25,0.5,0.75,1], labels=['0','0.25','0.5','0.75','1'])
    ax.grid(True)
    ax.text(-0.28, 1.02, 'c', ha='right', va='top', transform=ax.transAxes, weight='bold')
    sns.despine()
    
    ax = fig.add_subplot(gs[1,2])
    sns.heatmap(cf, annot=True, cmap="Blues", fmt='.0f', cbar=False)
    ax.set(xlabel='based on quality index', ylabel="human")
    #ax.xaxis.tick_top()
    ax.set_xticks([0.5,1.5], labels=['no', 'yes'])
    ax.set_yticks([0.5,1.5], labels=['no', 'yes'])
    ax.text(0.5, 1.02, rf"F1={f1:.2f}, Cohen's $\kappa$={ck:.2f}", ha='center', va='bottom', transform=ax.transAxes)
    ax.text(-0.2, 1.02, 'd', ha='right', va='top', transform=ax.transAxes, weight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.44, hspace=0.32)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')

# Randomly samples spindles from SHHS subjects for manual labeling, visualizes them for annotation, and evaluates detection quality using performance metrics.
def main():
    visit = 'shhs1'
    df = pd.read_excel('../shhs_mastersheet2.xlsx')
    df = df[df.visitnumber==visit].reset_index(drop=True)
    
    save_path2 = 'sp_q_manual_label.csv'
    ch_names = ['C3M2', 'C4M1']
    spindle_detection_dir = 'detection_luna_q=0'
    edf_dir = 'clean_edf'
    sp_vis_dir = 'figures_spindle_vis'
    os.makedirs(sp_vis_dir, exist_ok=True)
    
    # randomly choose K edfs to annotate
    np.random.seed(2024)
    K = 10
    ids = np.random.choice(len(df), K, replace=False)
    df = df.iloc[ids].reset_index(drop=True)
    Kspindle = 100  # randomly choose `Kspindle` spindles from each EDF to annotate
    
    for i in tqdm(range(len(df))):
        sid = str(df.nsrrid.iloc[i])
        sp_path = os.path.join(spindle_detection_dir, f'spindle_{sid}-annotation.csv')
        if not os.path.exists(sp_path):
            continue
        df_sp = pd.read_csv(sp_path)
        
        edf_path = os.path.join(edf_dir, f'{visit}-{sid}-clean.edf')
        edf = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        assert edf.ch_names==ch_names
        eeg = edf.get_data(picks=ch_names)*1e6
        Fs = edf.info['sfreq']
        pad = int(round(5*Fs))
        eeg_f = mne.filter.filter_data(eeg, Fs, 11, 15, verbose=False)
        
        ids2 = np.random.choice(np.arange(len(df_sp)), min(len(df_sp), Kspindle), replace=False)
        for j in tqdm(ids2):
            q = df_sp.Q.iloc[j]
            save_path = os.path.join(sp_vis_dir, f'sp_vis_q={q:.1f}_{visit}-{sid}_{j}.png')
            if os.path.exists(save_path):
                continue
            center = (df_sp.START.iloc[j]+df_sp.STOP.iloc[j])/2
            center = int(round(center*Fs))
            start = max(0, center-pad)
            stop  = min(eeg.shape[1], center+pad)
            chid  = ch_names.index(df_sp.CH.iloc[j])
            plot_spindle(eeg[chid, start:stop], eeg_f[chid, start:stop],
                int(round(df_sp.START.iloc[j]*Fs))-start,
                int(round(df_sp.STOP.iloc[j]*Fs))-start,
                Fs, q, sid, j,
                save_path, save_path2)
                
    df = pd.read_csv(save_path2)
    y = df.ManualLabel.values
    q = df.Q.values
    plot_perf(y, q, save_path='spindle_q_perf.png')

# Run code
if __name__=='__main__':
    main()

