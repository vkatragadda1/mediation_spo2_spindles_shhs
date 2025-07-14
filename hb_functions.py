# Defining functions for the calculation of hypoxic burden to use in spo2_features_extraction script.

# Load all necessary packages 
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import neurokit2 as nk

# Define threshold for invalid/artifact SPO2 values. Set to 60% here 
BAD_SPO2_THRESHOLD = 60

# Filter SpO2 to ensure that artifacts are replaced by mean of SpO2 values >60%. Downsamples to 1Hz.
def filter_spo2(spo2_arr, spo2_sfreq, verbose=False):
    # Replacing extreme values with mean value of valid SPO2 data
    spo2_mean = np.nanmean(spo2_arr[spo2_arr >= BAD_SPO2_THRESHOLD])
    spo2_arr[spo2_arr < BAD_SPO2_THRESHOLD] = spo2_mean

    # Downsampling to match 1Hz frequency
    if spo2_sfreq != 1:
        spo2_arr = spo2_arr[::int(spo2_sfreq)]
    spo2_filtered = spo2_arr

    if verbose:
        # Makes figures showing changes before/after filtering  but is currently turned off w verbose=False 
        abd = reader.get_single_channel_data('Abdominal', int(end - time_span), int(end + time_span))
        af = reader.get_single_channel_data('Airflow', int(end - time_span), int(end + time_span))
        tt = np.arange(len(spo2_arr))/spo2_sfreq
        tt_abd = np.arange(len(abd))/reader.get_channel_sample_frequency('Abdominal')
        tt_af = np.arange(len(af))/reader.get_channel_sample_frequency('Airflow')

        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(311); ax0 = ax
        ax.plot(tt, spo2_arr)
        ax.plot(tt, spo2_filtered)
        ax.axvline(x=time_span/spo2_sfreq)
        ax = fig.add_subplot(312, sharex=ax0)
        ax.plot(tt_abd, abd)
        ax = fig.add_subplot(313, sharex=ax0)
        ax.plot(tt_af, af)
        plt.show()
        #plt.savefig(f'./img/b/{idx}.png')
    return spo2_filtered


# Calculates desaturation events 
def detect_oxygen_desaturation(spo2, is_plot=False, duration_max=120, return_type='pd', max_duration=None, min_desat=None):
    # Detects all drops in SpO₂ ≥2% for ≥5 seconds, excluding large drops (>50%) as likely artifacts.
    spo2_max = spo2[0]  # initialize max value 
    spo2_max_index = 1  # max index 
    spo2_min = 100  # initialize min 
    des_onset_pred_set = np.array([], dtype=int)  # Store detected start times 
    des_duration_pred_set = np.array([], dtype=int)  # Store durations 
    des_level_set = np.array([])  # Recorded drop levels 
    des_onset_pred_point = 0  # Predicted onset point of event 
    des_flag = 0  # Flagging a possible desat 
    ma_flag = 0  # Motion artifact flag 
    spo2_des_min_thre = 2  # Min drop must be 2% 
    spo2_des_max_thre = 50  # Max drop for motion artifact is 50% 
    duration_min = 5  # Min duration of event is 5 sec
    prob_end = []

    # Calculating drops for each SPO2 value in relation to the set min/max thresholds/surrounding data points 
    for i, current_value in enumerate(spo2):
        
        # Calculates oxygen drop level
        des_percent = spo2_max - current_value   

        # Detecting motion artifacts
        if ma_flag and (des_percent < spo2_des_max_thre):
            if des_flag and len(prob_end) != 0:
                des_onset_pred_set = np.append(des_onset_pred_set, des_onset_pred_point)
                des_duration_pred_set = np.append(des_duration_pred_set, prob_end[-1] - des_onset_pred_point)
                des_level_point = spo2_max - spo2_min
                des_level_set = np.append(des_level_set, des_level_point)
            
            # Reset init values to loop through again 
            spo2_max = current_value
            spo2_max_index = i
            ma_flag = 0
            des_flag = 0
            spo2_min = 100
            prob_end = []
            continue

        # If drop is >2%, Record start of event 
        if des_percent >= spo2_des_min_thre:
            if des_percent > spo2_des_max_thre:
                ma_flag = 1 # If it drops more than max mark as a motion artifact 
            else: # Otherwise continue as a desat event 
                des_onset_pred_point = spo2_max_index
                des_flag = 1
                if current_value < spo2_min:
                    spo2_min = current_value
                    
        # Sets max value based on baseline to compare future drops 
        if current_value >= spo2_max and not des_flag:
            spo2_max = current_value
            spo2_max_index = i
            
        # If desaturation event is marked: 
        elif des_flag:
            if current_value > spo2_min:
                if current_value > spo2[i - 1]:
                    prob_end.append(i)

                # If oxygen continues to drop 
                if current_value <= spo2[i - 1] < spo2[i - 2]:
                    spo2_des_duration = prob_end[-1] - spo2_max_index

                    # If drop time isn't long enough, moves on rather than counting the desat event 
                    if spo2_des_duration < duration_min:
                        spo2_max = spo2[i - 2]
                        spo2_max_index = i - 2
                        spo2_min = 100
                        des_flag = 0
                        prob_end = []
                        continue

                    else:
                        # Records the desat event if all the conditions match up (event time and amount of drop) 
                        if duration_min <= spo2_des_duration <= duration_max:
                            des_onset_pred_set = np.append(des_onset_pred_set, des_onset_pred_point)
                            des_duration_pred_set = np.append(des_duration_pred_set, spo2_des_duration)
                            des_level_point = spo2_max - spo2_min
                            des_level_set = np.append(des_level_set, des_level_point)

                        # If the desat event is an extended time period, it records the events separately 
                        else:
                            # First SPO2 drop 
                            des_onset_pred_set = np.append(des_onset_pred_set, des_onset_pred_point)
                            des_duration_pred_set = np.append(des_duration_pred_set, prob_end[0] - des_onset_pred_point)
                            des_level_point = spo2_max - spo2_min
                            des_level_set = np.append(des_level_set, des_level_point)

                            # Searching again for remaining events 
                            remain_spo2 = spo2[prob_end[0]:i + 1]
                            _onset, _duration, _des_level = detect_oxygen_desaturation(remain_spo2, is_plot=False, return_type='tuple')
                            des_onset_pred_set = np.append(des_onset_pred_set, _onset + prob_end[0])
                            des_duration_pred_set = np.append(des_duration_pred_set, _duration)
                            des_level_set = np.append(des_level_set, _des_level)

                        spo2_max = spo2[i - 2]
                        spo2_max_index = i - 2
                        spo2_min = 100
                        des_flag = 0
                        prob_end = []
                        
    # Visualization plots, currently turned off 
    if is_plot: 
        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
        ax1.plot(spo2, 'b')
        ax1.set_title('ground truth')
        for i in range(0, len(od_start)):
            s = od_start[i]
            e = od_start[i] + od_duration[i]
            se = [item for item in range(s, e + 1)]
            ax1.plot(s, spo2[s], color='r', linestyle='none', marker='o')
            ax1.plot(se, spo2[se], color='r', linestyle='-')
            ax1.plot(e, spo2[e], color='r', linestyle='none', marker='*')

        # Predicted annotation
        ax2.plot(spo2, 'b')
        ax2.set_title('prediction')
        for i in range(0, len(des_onset_pred_set)):
            s = des_onset_pred_set[i]
            e = des_onset_pred_set[i] + des_duration_pred_set[i]
            se = [item for item in np.arange(s, e + 1)]
            ax2.plot(s, spo2[s], color='r', linestyle='none', marker='o')
            ax2.plot(se, spo2[se], color='r', linestyle='-')
            ax2.plot(e, spo2[e], color='r', linestyle='none', marker='*')
        plt.show()

    # Filters desat data if looking for specific lengths/amounts of drops 
    if max_duration is not None:
        ids = des_duration_pred_set<=max_duration
        des_onset_pred_set = des_onset_pred_set[ids]
        des_duration_pred_set = des_duration_pred_set[ids]
        des_level_set = des_level_set[ids]
    if min_desat is not None:
        ids = des_level_set>=min_desat
        des_onset_pred_set = des_onset_pred_set[ids]
        des_duration_pred_set = des_duration_pred_set[ids]
        des_level_set = des_level_set[ids]
    if return_type=='tuple':
        return des_onset_pred_set, des_duration_pred_set, des_level_set
    else:
        return pd.DataFrame(data={'Start':des_onset_pred_set, 'Duration':des_duration_pred_set, 'Desat':des_level_set})



# Calculating hypoxic burden
def calc_hypoxic_burden(spo2, event_times, Fs, tst, verbose=False, time_span=120):
    # Respiratory event duration 10-120 s, the maximum delay of a hypoxic event caused by a respiratory event is 120 seconds.

    # Storing all the related SPO2 values for desats 
    all_ah_related_spo2 = [] 

    # Looking through the times determined by the previous function to find desat events
    for et in event_times:
        start = int(round((et-time_span)*Fs))
        end   = int(round((et+time_span)*Fs))
        if start<0 or end>len(spo2):
            continue
        nearby_spo2 = spo2[start:end]
        if np.isnan(nearby_spo2).mean()>0.5: # Using the average values rather than NaN 
            continue
        filtered_spo2 = nearby_spo2 
        all_ah_related_spo2.append(filtered_spo2)

    if len(all_ah_related_spo2)==0:
        return np.nan, np.nan

    # Calc average values across events 
    all_spo2_dest = np.array(all_ah_related_spo2)
    avg_spo2 = np.nanmean(all_spo2_dest, axis=0)
    avg_spo2 = savgol_filter(avg_spo2, 30, 3)

    # Finding local peaks to figure out when the desat event returns to normal 
    # Finds baseline before event, new baseline after event 
    peaks, _ = find_peaks(avg_spo2)
    start_secs = peaks[np.where(peaks < time_span)[0][-1]]
    end_secs = peaks[np.where(peaks > time_span)[0][0]]

    # Visualizing data 
    if verbose:
        x = np.arange(len(avg_spo2))
        plt.close()
        plt.plot(x, avg_spo2)
        plt.plot(x[start_secs], avg_spo2[start_secs], "o")
        plt.plot(x[end_secs], avg_spo2[end_secs], "*", markersize=10)
        plt.axvline(x=time_span)
        plt.title(f"{name}")
        plt.show()

    # Looping through all the desats identified to calculate the total burden
    # Divide by 60 to get burden per min 
    total_burden = 0
    for spo2_dest_curve in all_spo2_dest:
        # Calc baseline before the desat 
        baseline = np.nanpercentile(spo2_dest_curve[:start_secs], 99)
        interest_spo2 = spo2_dest_curve[start_secs: end_secs]
        total_burden += np.nansum(baseline - interest_spo2)
    total_burden /= 60 

    # Average hypoxic burden 
    per_ah_event_burden = total_burden / len(all_spo2_dest)
    # Adding in average amount for any skipped events. 
    # Useful if there is a time period that within that 120sec window
    total_burden += per_ah_event_burden * (len(event_times) - len(all_spo2_dest))

    # Normalize burden to sleep time 
    return total_burden / tst, avg_spo2



# Linear interpolation of desat events to get features with desats removed 
def get_spo2_no_desat(spo2, od_events):
    # initialize spo2_res to have spo2 data 
    spo2_res = np.array(spo2)

    # Loops through the determined desat events 
    # Defines the points of interests, start time, and duration is the calculated value +t1 
    # V1 and V2 are the spo2 values at the corresponding times 
    for i in range(len(od_events)):
        t1 = od_events.Start.iloc[i]
        v1 = spo2[t1]
        t2 = t1+od_events.Duration.iloc[i]
        v2 = spo2[t2]

        # Makes sure its a numerical value 
        assert not np.isnan(v1) and not np.isnan(v2), od_events.iloc[i]

        # Interpolate w slope and fills in the values accordingly in that time gap that was determined 
        fill_t = np.arange(t1+1,t2)
        spo2_res[fill_t] = (v2-v1)/(t2-t1)*(fill_t-t1)+v1

    # Return the new array 
    return spo2_res


