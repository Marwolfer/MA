#%%
from gzip import BadGzipFile
import os
import mne
import numpy as np
import pandas as pd
from mne_realtime import RtEpochs, MockRtClient
from scipy import signal
import time
import matplotlib.pyplot as plt
#from py_neuromodulation.nm_filter import *
#from py_neuromodulation.nm_resample import *
import py_neuromodulation as nm
from scipy.signal import find_peaks
import re
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from mne.filter import notch_filter
import multiprocessing
import mne
import scipy
import multiprocessing
from mne.preprocessing import ICA
from mne import create_info, EpochsArray
from mne import pick_types
from mne_icalabel import label_components
from queue import Queue
import time
from threading import Thread

from py_neuromodulation.filter import MNEFilter
from py_neuromodulation.processing import Resampler
#Change directories!


# Define epoch parameters
EEG_EPOCH_LIMITS = (-1.005, -0.005)  # in seconds
EMG_EPOCH_LIMITS = (-0.105, 0.100)  # in seconds
EMG_BASELINE_WIN = (-0.105, -0.005)  # in seconds



channel_types = {'APBr': 'emg', 'FDIr': 'emg'}

# Define directories
sound_directory = r'/home/marco/Documents/GitHub/tms_eeg_decoding/preprocessing' 
load_directory = r'/home/marco/Documents/GitHub/tms_eeg_decoding/data'

# Define directories
#load_directory = r'C:\Users\Lisa Haxel\Documents\CEBRA\data_raw'
save_directory = r'/home/marco/Documents/GitHub/tms_eeg_decoding/data/bandpass'
mep_directory = r'/home/marco/Documents/GitHub/tms_eeg_decoding/data/bandpass'

# List all files in the directory
all_files = os.listdir(load_directory)


# Filter for files starting with 'subject_' and ending with '_raw.set'
#subject_files = [f for f in all_files if f.startswith('subject_') and f.endswith('_raw.set')]
#subject_files = [f for f in all_files if f.startswith('subject_') and f.endswith('_pen.set')]
subject_files = [f for f in all_files if f.startswith('subject_') and f.endswith('_py.set')]

# Function to extract the subject number from the filename
def extract_subject_number(filename):
    match = re.search(r'subject_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return float('inf')  # If no number found, place it at the end

# Sort the files based on the extracted subject number
subject_files.sort(key=extract_subject_number)

#%%
# List of indices provided
#indices = []

# Generating the list of subject filenames
#subject_files = [f'subject_{i}_mne_py.set' for i in indices]
#subject_files = [f'subject_{i}_mne_pen.set' for i in indices]
subject_indices = subject_indices = [106, 107, 108, 109, 110, 111]
#subject_indices = [1]
#subject_files = ['subject_2_mne_py.set', 'subject_4_mne_py.set', 'subject_7_mne_py.set', 'subject_9_mne_py.set',
#                  'subject_11_mne_py.set', 'subject_13_mne_py.set', 'subject_14_mne_py.set',
#                  'subject_18_mne_py.set', 'subject_22_mne_py.set', 'subject_24_mne_py.set']
subject_files = [f'subject_{i}_mne_pen.set' for i in subject_indices]

# define frequency band. For every frequency band, later create an MNEFIlter where this frequency is bandstopped/not in frequency range

#frequency_bands = { "delta": (2, 4), "theta": (4, 8),"alpha": (8, 12), "beta": (13, 30), "gamma": (30, 45)}
# Create band pairs
#band_names = list(frequency_bands.keys())
#band_pairs = [(band_names[i], band_names[j]) for i in range(len(band_names)) for j in range(i + 1, len(band_names))]

#frequency_pairs = {}


#for pair in band_pairs:
#    band1, band2 = frequency_bands[pair[0]], frequency_bands[pair[1]]
#    if band1[1] == band2[0]:  # Neighboring bands
        
        #print(f"Neighboring bands: {pair}")
#        outer_lowcut, outer_highcut = band1[0], band2[1]
#        frequency_pairs[f"{pair[0]}_{pair[1]}"] =[(outer_lowcut, outer_highcut)]
        #print(f"Outer band: {outer_lowcut}-{outer_highcut} Hz")
     
#    else:  # Non-neighboring bands
        #print(f"Non-neighboring bands: {pair}")
#        outer_low, outer_high = band1[0], band2[1]
#        inner_low, inner_high = band1[1], band2[0]
#        frequency_pairs[f"{pair[0]}_{pair[1]}"] = [(outer_low, outer_high) ,(inner_high, inner_low)]
        #print(f"Outer band: {outer_low}-{outer_high} Hz")
        #print(f"Inner band: {inner_low}-{inner_high} Hz")

band_combinations = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
    "delta+theta": (0.5, 8),
    "theta+alpha": (4, 13),
    "alpha+beta": (8, 30),
    "beta+gamma": (13, 45),
    "delta+theta+alpha": (0.5, 13),
    "theta+alpha+beta": (4, 30),
    "alpha+beta+gamma": (8, 45),
    "delta+theta+alpha+beta": (0.5, 30),
    "theta+alpha+beta+gamma": (4, 45),
    "delta+theta+alpha+beta+gamma": (0.5, 45)
}


# Loop over all subject files
for freq_band, freq_bands_range in band_combinations.items():
    
    for subject_file in subject_files:
        # Extract subject number
        subject_number = extract_subject_number(subject_file)
        print(f"Processing subject {subject_number}")

        file_path = os.path.join(load_directory, subject_file)

        try: 
            raw_data = mne.io.read_raw_eeglab(file_path, preload=True)
            print(f"Loaded data for {subject_file}")
        except:
            print(f"Could not load data for {subject_file}")
            continue  # Skip to next subject if loading fails


        # Set channel types  
        raw_data.set_channel_types(channel_types)


        # set reference to average
        raw_data.set_eeg_reference(ref_channels='average', projection=False)


        raw_data.drop_channels(['FT9', 'FT10', 'TP9', 'TP10'])

        # Path to the channel names text file
        channel_file_path = os.path.join(sound_directory, 'channel_names.txt')

        # Load channel names from the text file
        with open(channel_file_path, 'r') as file:
            channel_names = file.read().splitlines()

        # Load leadfieldmatrix from the csv file
        leadfield_file_path = os.path.join(sound_directory, 'LFM_Aalto_ReftepPP.csv')
        leadfield_matrix_old = np.loadtxt(leadfield_file_path, delimiter=',')

        leadfield_matrix_mean = np.mean(leadfield_matrix_old, axis=0)
        leadfield_matrix = leadfield_matrix_old - leadfield_matrix_mean
        #leadfield_matrix = leadfield_matrix_old

        # Get the channels present in the raw data
        raw_channels = raw_data.ch_names

        # Identify missing channels
        missing_channels = [ch for ch in channel_names if ch not in raw_channels]
        extra_channels = [ch for ch in raw_channels if ch not in channel_names and ch not in channel_types]


        # Print missing and extra channels
        #if missing_channels:
            #print(f"Missing channels: {missing_channels}")
        #if extra_channels:
           #print(f"Extra channels: {extra_channels}")

         # Remove the missing channels from channel_names and update the lead field matrix
        for ch in missing_channels:
            idx = channel_names.index(ch)
            channel_names.remove(ch)
            leadfield_matrix = np.delete(leadfield_matrix, idx, axis=0)  # Remove corresponding row
            leadfield_matrix = np.delete(leadfield_matrix, idx, axis=1)  # Remove corresponding column

        # Remove extra channels from raw data
        raw_data.drop_channels(extra_channels)

        # Combine EEG and EMG channel names
        all_channel_names = channel_names + [ch for ch in raw_data.ch_names if ch in channel_types]

        # Reorder the raw data channels to match the lead field matrix and include EMG channels
        raw_data.reorder_channels(all_channel_names)

        # Verify the alignment
        #print(f"Reordered channels: {raw_data.ch_names}")
        #print(f"Lead field matrix shape: {leadfield_matrix.shape}")

        # Check if there is a stim channel in raw data
        stim_channel = mne.pick_types(raw_data.info, stim=True)
        if len(stim_channel) == 0:
            #print("No stim channel found. Proceeding without a stim channel.")
            stim_channel = None
        else:
            stim_channel = raw_data.info['ch_names'][stim_channel[0]]


        # Extract events and event_id
        events, event_id = mne.events_from_annotations(raw_data)

        # Get the integer event ID for 'A - Out'
        event_id_out = event_id['Out']

        # Filter events for 'Out' annotations
        filtered_events = events[events[:, 2] == event_id_out]

        # Ensure sufficient pre- and post-stimulus data
        fs = raw_data.info['sfreq']
        filtered_events = filtered_events[
            (filtered_events[:, 0] + EEG_EPOCH_LIMITS[0] * fs > 0) & 
            (filtered_events[:, 0] + EEG_EPOCH_LIMITS[1] * fs < len(raw_data.times))
        ]

        # Limit to 1000 trials if necessary
        if len(filtered_events) > 800:
            filtered_events = filtered_events[:800]
            print('Keeping first 800 trials only')

        # Calculate the total duration for 800 trials
        last_event_sample = filtered_events[-1, 0]  # Sample index of the last event
        epoch_duration_samples = int((EEG_EPOCH_LIMITS[1] - EEG_EPOCH_LIMITS[0]) * fs)
        end_sample = last_event_sample + epoch_duration_samples  # End sample after the last event

        # Ensure end_sample / fs does not exceed max time
        max_time = raw_data.times[-1]
        if end_sample / fs > max_time:
            end_sample = int(max_time * fs)

        # Crop the raw data to include up to the end of the 800th trial
        raw_data.crop(tmax=end_sample / fs)

        # Crop the raw data to include up to the end of the 800th trial
        raw_data.crop(tmax=end_sample / fs)

        eeg_picks = mne.pick_types(raw_data.info, eeg=True)

        # Create a dictionary of channel positions
        ch_pos = {raw_data.ch_names[pick]: raw_data.info['chs'][pick]['loc'][:3] for pick in eeg_picks}

        # Create a custom montage
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')

        # Apply the montage to your raw data
        raw_data.set_montage(montage)

        # Create an Info object for the EEG channels
        raw_info = mne.pick_info(raw_data.info, sel=eeg_picks)

        # Create epochs array without stim channel for later use 
        epochs_eeg = mne.Epochs(raw_data, filtered_events, event_id=event_id_out, tmin=EEG_EPOCH_LIMITS[0], 
                            tmax=EEG_EPOCH_LIMITS[1], baseline=(None,None), preload=True, picks=eeg_picks, detrend=None, verbose=False)

        # save filtered_events as a numpy array
        np.save(os.path.join(save_directory, f"sub-{subject_number}_filtered_events.npy"), filtered_events)


        # save filtered_events as a txt file
        np.savetxt(os.path.join(save_directory, f"sub-{subject_number}_filtered_events.txt"), filtered_events, fmt='%d')	

        # load filtered_events
        filtered_events = np.load(os.path.join(save_directory, f"sub-{subject_number}_filtered_events.npy"))

        # Create the stim channel data
        stim_data = np.zeros((1, len(raw_data.times)))
        event_indices = filtered_events[:, 0].astype(int)
        stim_data[0, event_indices] = event_id_out

        # Create an Info object for the stim channel
        stim_info = mne.create_info(['STI 014'], sfreq=raw_data.info['sfreq'], ch_types=['stim'])

        # Create the RawArray object for the stim channel
        stim_raw = mne.io.RawArray(stim_data, stim_info)

        # Add the stim channel to the existing raw data
        raw_data.add_channels([stim_raw], force_update_info=True)


        eeg_picks = mne.pick_types(raw_data.info, eeg=True)

        picks = mne.pick_types(raw_data.info, eeg=True, stim=True)

        emg_picks = mne.pick_types(raw_data.info, emg=True)

        # Convert EMG data from millivolts to microvolts
        for emg_ch in emg_picks:
            raw_data._data[emg_ch] *= 1e3


        # Create the mock-client object for EEG data
        rt_client = MockRtClient(raw_data)

        # Create real-time epochs objects for EEG and EMG data
        rt_epochs = RtEpochs(rt_client, event_id=event_id_out, tmin=EEG_EPOCH_LIMITS[0], 
                        tmax=EEG_EPOCH_LIMITS[1], baseline=None, picks=picks, detrend=None, verbose=False)

        # Dynamically identify the stim channel index
        stim_channel_index = rt_epochs.ch_names.index('STI 014')

        rt_emg_epochs = RtEpochs(rt_client, event_id=event_id_out, tmin=EMG_EPOCH_LIMITS[0], 
                            tmax=EMG_EPOCH_LIMITS[1], baseline=None, picks=emg_picks, detrend=None, verbose=False)
    
        # edit ICA preprocesser to include bandstop filter
        class ICAPreprocessor:
            def __init__(self, num_channels, n_components=25, tmin=0,  raw_info=None, eeg_picks=None, montage=None, freq_bands=None):
                self.num_channels = num_channels
                self.calibration_epochs = []
                self.calibration_complete = False
                self.exclude_idx = None
                self.tmin = tmin
                self.raw_info = raw_info.copy()
                self.eeg_picks = eeg_picks
                self.ch_names = [raw_info['ch_names'][pick] for pick in eeg_picks]
                self.montage = montage
                self.n_components = n_components
                self.sfreq = 5000
                self.resample_freq_hz = 1000
                self.preprocessed_epochs = []  # New attribute to store preprocessed epochs
                self.real_time_trials_processed = 0
                self.bands_range = freq_bands
            

                self.resampler = Resampler(sfreq=self.sfreq, resample_freq_hz=self.resample_freq_hz)
                self.mne_filter = MNEFilter(f_ranges=[self.bands_range], sfreq=self.resample_freq_hz, filter_length="1000ms", verbose=False)
                #self.mne_filter = MNEFilter(f_ranges=[self.freq_pair[0]], sfreq=self.resample_freq_hz, filter_length="1000ms", verbose=False)
                #if len(self.freq_pair) > 1:
                #    self.mne_filter2 = MNEFilter(f_ranges=[self.freq_pair[1]], sfreq=self.resample_freq_hz, filter_length="1000ms", verbose=False)
                # Initialize the Resampler and MNEFilter
            

                self.calibration_ready = False

                self.ica = ICA(
                    n_components=n_components,
                    method="infomax",
                    fit_params=dict(extended=True),
                    random_state=97,
                    max_iter="auto"
                )
 
                self.update_counter = 0

            def preprocess_epoch(self, epoch):
                # Detrend
                detrended_epoch = signal.detrend(epoch, axis=1, type='linear')
            
                # Resample
                resampled = self.resampler.process(detrended_epoch)

                # Apply bandpass filter
                bpf = self.mne_filter.filter_data(resampled).squeeze()
                #if len(self.freq_pair) > 1:
                #    bpf = self.mne_filter2.filter_data(bpf).squeeze()
            
                return bpf


            def collect_calibration_data(self, epoch):
                if not self.calibration_complete:
                    preprocessed_epoch = self.preprocess_epoch(epoch)
                    self.preprocessed_epochs.append(preprocessed_epoch)
                    if len(self.preprocessed_epochs) >= 100:
                        print(f"Calibration data collected. Number of epochs: {len(self.preprocessed_epochs)}")
                        self.calibrate_ica()

            def equalize_epoch_lengths(self):
                min_length = min(epoch.shape[1] for epoch in self.preprocessed_epochs)
                self.preprocessed_epochs = [epoch[:, :min_length] for epoch in self.preprocessed_epochs]

        
            def calibrate_ica(self):
                print("Calibrating ICA...")

                self.equalize_epoch_lengths()
            
                epochs_array = np.stack(self.preprocessed_epochs, axis=0)

                info = self.raw_info.copy()
                info.set_montage(self.montage)
            
                new_info = mne.create_info(ch_names=info.ch_names, sfreq=self.resample_freq_hz, ch_types=info.get_channel_types())
                new_info.set_montage(self.montage)
            
                epochs = mne.EpochsArray(epochs_array, new_info, tmin=self.tmin)

                print(f"Fitting ICA on {len(epochs)} epochs")
                self.ica.fit(epochs)

                self.ic_labels = label_components(epochs, self.ica, method="iclabel")
                labels = self.ic_labels["labels"]
                probabilities = self.ic_labels["y_pred_proba"]
            
                bad_component_types = ["eye blink"]
                threshold = 0.85
            
                self.exclude_idx = []
                for idx, (label, prob) in enumerate(zip(labels, probabilities)):
                    if label in bad_component_types and prob > threshold:
                        self.exclude_idx.append(idx)
            
                print("Components identified as artifacts by ICLabel:")
                for idx in self.exclude_idx:
                    if hasattr(self, 'ic_labels'):
                        component_type = self.ic_labels["labels"][idx]
                        probability = self.ic_labels["y_pred_proba"][idx]
                        print(f"Component {idx}: {component_type} (probability: {probability:.2f})")
                    else:
                        print(f"Component {idx}: Unknown type")

                self.calibration_complete = True
                self.ica.exclude = self.exclude_idx  # Set the exclude property of the ICA object


            def get_preprocessed_epochs(self):
                info = self.raw_info.copy()
                info.set_montage(self.montage)
            
                # Create new info with updated sampling frequency
                new_info = mne.create_info(ch_names=info.ch_names, sfreq=self.resample_freq_hz, ch_types=info.get_channel_types())
                new_info.set_montage(self.montage)
            
                epochs_array = np.stack(self.preprocessed_epochs, axis=0)
                return mne.EpochsArray(epochs_array, new_info, tmin=self.tmin)

            def apply_ica(self, epoch):
                if not self.calibration_complete:
                    return epoch
            
                self.real_time_trials_processed += 1
            
                if len(self.ica.exclude) == 0:
                    return epoch  # No components excluded, return original data
            
                # Create a temporary Epochs object with the single epoch
                info = self.raw_info.copy()
                info.set_montage(self.montage)
                temp_epochs = mne.EpochsArray(epoch[np.newaxis, :, :], info, tmin=self.tmin)
            
                # Apply ICA using MNE's apply method
                temp_epochs_clean = self.ica.apply(temp_epochs, verbose=False)
            
                # Extract the cleaned epoch
                cleaned_epoch = temp_epochs_clean._data[0]
            
                return cleaned_epoch


        class SOUNDPreprocessor:
            def __init__(self, num_of_eeg_channels, sfreq, resample_freq_hz, leadfield_matrix, raw_info, eeg_picks, montage=None, ica_preprocessor=None, freq_bands=None):
                self.num_of_eeg_channels = num_of_eeg_channels
                self.sfreq = sfreq
                self.resample_freq_hz = resample_freq_hz
                self.lfm = leadfield_matrix
                self.raw_info = raw_info.copy()
                self.eeg_picks = eeg_picks
                self.bands_range = freq_bands
        
            
                if ica_preprocessor is None:
                    self.ica_preprocessor = ICAPreprocessor(
                        num_channels=num_of_eeg_channels, 
                        tmin=0, 
                        raw_info=raw_info,
                        eeg_picks=eeg_picks,
                        montage=montage,
                        freq_bands_range=freq_bands_range
                    )
                else:
                    self.ica_preprocessor = ica_preprocessor


                # SOUND parameters
                self.iterations = 10
                self.lambda0 = 0.01
                self.convergence_boundary = 0.01
                self.update_interval_in_samples = int(0.1 * self.resample_freq_hz)  # 100 ms
                self.sigmas = np.ones((num_of_eeg_channels, 1))


                # Baseline correction
                self.baseline_update_rate = 0.0006
                self.baseline_correction = np.zeros(num_of_eeg_channels)

                # Initialize state
                self.filter = np.eye(self.num_of_eeg_channels)
                self.samples_collected = 0

                # Ensure the bandstop frequencies are within the valid range
                self.resampler = Resampler(sfreq=sfreq, resample_freq_hz=resample_freq_hz)
          
                self.mne_filter = MNEFilter(f_ranges=[self.bands_range], sfreq=self.resample_freq_hz, filter_length="1000ms", verbose=False)
                # Initialize the Resampler and MNEFilter
            

        
            def preprocess_eeg(self, epoch):
                # Detrend
                detrended_epoch = signal.detrend(epoch, axis=1, type='linear')
            
                # Resample
                resampled = self.resampler.process(detrended_epoch)

                # print shape
                print(f"Resampled shape: {resampled.shape}")

                # Apply notch filter
                #notched = self.notch_filter.process(resampled)

                start_time = time.time()
                bpf = self.mne_filter.filter_data(resampled).squeeze()
                end_time = time.time()

                print(f"Bandpass filter took {end_time - start_time:.4f} seconds")


                 # Apply bandpass filter
                # bpf = self.mne_filter.filter_data(resampled).squeeze()
                # # print how long bpf takes
            
            
                # Apply ICA
                ica_processed = self.process_ica(bpf)

                #    Apply SOUND
                sound_processed = self.apply_sound(ica_processed)
            
                return sound_processed
        
            def process_ica(self, epoch):
                return self.ica_preprocessor.apply_ica(epoch)


            def apply_sound(self, epoch):
                self.samples_collected += epoch.shape[1]

                if self.samples_collected >= self.update_interval_in_samples:
                    new_filter, new_sigmas = self.sound(epoch)
                    self.filter = new_filter
                    self.sigmas = new_sigmas
                    self.samples_collected = 0

                # Apply current SOUND filter
                sound_processed = np.matmul(self.filter, epoch)

                # Update baseline correction
                self.baseline_correction = (1 - self.baseline_update_rate) * self.baseline_correction + \
                                    self.baseline_update_rate * np.mean(sound_processed, axis=1)
            
                # Apply baseline correction
                sound_processed -= self.baseline_correction[:, np.newaxis]

                return sound_processed

            def sound(self, data):
                # Actual baseline correction for Sound data buffer
                data = np.subtract(data, self.baseline_correction[:, np.newaxis])

                # Smooth sigmas update coeff:
                sigmas_update_coeff = 0.05

                dataCov = np.matmul(data, data.T) / data.shape[1]

                LL = np.matmul(self.lfm, self.lfm.T)
                regularization_term = self.lambda0 * np.trace(LL) / self.num_of_eeg_channels
                LL_reg = LL / regularization_term

                sigmas_prev_update = np.copy(self.sigmas)

                for k in range(self.iterations):
                    sigmas_old = np.copy(self.sigmas)
                    GAMMA = np.linalg.pinv(LL_reg + np.diagflat(np.square(self.sigmas)))
                
                    GAMMA_diag_inv = 1 / np.diag(GAMMA)
                    GAMMA_scaled = GAMMA * GAMMA_diag_inv[:, np.newaxis]
                    self.sigmas = np.sum(np.matmul(np.matmul(GAMMA_scaled.T, dataCov), GAMMA_scaled), axis=1)
                
                    max_noise_estimate_change = np.max(np.abs(sigmas_old - self.sigmas) / sigmas_old)
                    if max_noise_estimate_change < self.convergence_boundary:
                        #print(f"Output: Convergence reached after {k+1} iterations!")
                        break

                self.sigmas = np.array(self.sigmas, dtype=np.float32)
                self.sigmas = np.expand_dims(self.sigmas, axis=1)
                self.sigmas = sigmas_update_coeff * self.sigmas + (1 - sigmas_update_coeff) * sigmas_prev_update

                W = np.diag(1.0 / np.squeeze(self.sigmas))
                WL = np.matmul(W, self.lfm)
                WLLW = np.matmul(WL, WL.T)
                C = (WLLW + self.lambda0 * np.trace(WLLW) / self.num_of_eeg_channels * np.eye(self.num_of_eeg_channels))
                SOUND_filter = np.matmul(self.lfm, np.matmul(WL.T, np.linalg.solve(C, W)))

                # find the best-quality channel
                best_ch = np.argmin(self.sigmas)
                # Calculate the relative error in the best channel caused by SOUND overcorrection:
                rel_err = np.linalg.norm(np.matmul(SOUND_filter[best_ch,:], data) - data[best_ch,:])/np.linalg.norm(data[best_ch,:])
                #print(f"Output: Relative error in best channel = {rel_err}")

                return SOUND_filter, self.sigmas


        # Function to baseline correct EMG data
        def baseline_correction(data, times, timerange):
            t_idx = timerange
            baselines = np.mean(data[..., t_idx], axis=-1, keepdims=True)
            data_rm = data - baselines
            return data_rm

        # Function to detrend EMG data
        def linear_detrending(data, times, timerange):
            t_idx = timerange
            data_detrended = data.copy()
            for j in range(data.shape[0]):  # for each channel
                data_detrended[j, t_idx] = signal.detrend(data[j, t_idx], type='linear')
            return data_detrended


        # Function to merge EMG data using PCA
        def pca_merge(data):
            n_channels, n_times = data.shape
            reshaped_data = data.T  # Shape: (timepoints, channels)
            pca = PCA(n_components=1)
            pca_data = pca.fit_transform(reshaped_data)  # Shape: (timepoints, 1)
        
            # Enforce consistent direction by checking the sign of the first element
            if np.sum(pca.components_[0]) < 0:
                pca_data = -pca_data
        
            merged_data = pca_data.flatten()  # Shape: (timepoints,)
            return merged_data


        # Function to preprocess an EMG epoch
        def preprocess_emg(epoch, times):
            # Baseline correction using prestimulus time
            mep_amp_prestim_time = times < -0.005
            baseline_corrected = baseline_correction(epoch, times, mep_amp_prestim_time)
        
            # Linear detrending using post-stimulus limits
            mep_amp_limits = (times >= 0.02) & (times <= 0.04)
            detrended = linear_detrending(baseline_corrected, times, mep_amp_limits)
        
            # Merge channels using PCA
            merged = pca_merge(detrended)
            return merged

        # EMG settings
        epochs_emg_times = rt_emg_epochs.times

        # Define mep_amp times
        mep_amp_prestim_time = epochs_emg_times < - 0.005  # Pre-stimulus time window
        mep_amp_limits = (epochs_emg_times >= 0.02) & (epochs_emg_times <= 0.04)  # Post-stimulus time window

        # Define prominence for peak detection
        prominence = 50
        min_required_distance = 5  # Minimum required peak-to-peak distance in samples = 2.5 ms
        min_peak_to_peak_amplitude = 50  # Minimum required peak-to-peak amplitude in microvolts


        # Calibrate ICA using the first 100 trials
        calibration_epochs = epochs_eeg[:100]  # Get first 100 epochs



        # Initialize ICAPreprocessor
        ica_preprocessor = ICAPreprocessor(
            num_channels=len(eeg_picks), 
            tmin=EEG_EPOCH_LIMITS[0], 
            raw_info=raw_info,
            eeg_picks=eeg_picks,
            montage=montage,
            freq_bands=freq_bands_range
        )

        # Calibrate ICA
        for epoch in calibration_epochs:
            ica_preprocessor.collect_calibration_data(epoch)

        # Ensure calibration is complete (in case there were fewer than 100 epochs)
        if not ica_preprocessor.calibration_complete:
            ica_preprocessor.calibrate_ica()

        # # Plot ICA results
        # ica_preprocessor.ica.plot_components()

        # # Get preprocessed epochs
        # preprocessed_epochs = ica_preprocessor.get_preprocessed_epochs()

        # # Create evoked from preprocessed epochs
        # evoked = preprocessed_epochs.average()

        # # Plot overlay
        # ica_preprocessor.ica.plot_overlay(evoked, exclude=[0], picks="eeg")

        # Plot properties
        #ica_preprocessor.ica.plot_properties(preprocessed_epochs, picks=[0])

    
        # Now initialize SOUNDPreprocessor with the calibrated ICA
        sound_preprocessor = SOUNDPreprocessor(
            num_of_eeg_channels=len(eeg_picks),
            sfreq=5000,
            resample_freq_hz=1000,
            leadfield_matrix=leadfield_matrix,
            raw_info=raw_info,
            eeg_picks=eeg_picks,
            montage=montage,
            ica_preprocessor=ica_preprocessor,
            freq_bands=freq_bands_range
                # Pass the calibrated ICA preprocessor
            )


        # Initialize storage for results
        rejections = []
        preprocessing_times = []
        # Initialize DataFrame to store results
        results_df = pd.DataFrame(columns=['mep_raw', 'mep_log', 'loc_max', 'loc_min', 'preinnervation'])
        preinnervation_thresholds = []
        preprocessed_eeg_epochs = []
        mep_data_dict = {}
    

        #   Start the acquisition for both EEG and EMG
        rt_epochs.start()
        rt_emg_epochs.start()

        # Send raw buffers for both EEG and EMG
        rt_client.send_data(rt_epochs, picks, tmin=0, tmax=end_sample / fs, buffer_size=1000)
        rt_client.send_data(rt_emg_epochs, emg_picks, tmin=0, tmax=end_sample / fs, buffer_size=1000)


        try:
            trial_number = 0
            valid_trials = 0
            while True:
                # Fetch the next available EEG epoch
                try:
                    eeg_epoch = next(rt_epochs)
                    # Exclude the stim channel dynamically
                    eeg_epoch = np.delete(eeg_epoch, stim_channel_index, axis=0)
                except StopIteration:
                    print("No more EEG epochs available. Ending acquisition.")
                    break

                # Fetch the next available EMG epoch
                try:
                    emg_epoch = next(rt_emg_epochs)
                except StopIteration:
                    print("No more EMG epochs available. Ending acquisition.")
                    break

                # Start timing the preprocessing (excluding resampling)
                start_time = time.time()

                # Preprocess EEG epoch
                preprocessed_eeg = sound_preprocessor.preprocess_eeg(eeg_epoch)
                preprocessed_eeg_epochs.append(preprocessed_eeg)

                # Preprocess EMG epoch
                preprocessed_emg = preprocess_emg(emg_epoch, rt_emg_epochs.times)

                # Extract MEP and PI data
                mep_data = preprocessed_emg[mep_amp_limits]

                # Store mep_data in the dictionary
                mep_data_dict[trial_number] = mep_data


                # Plot MEP
                #plt.plot(mep_data)
                #plt.show()

                pi_data = preprocessed_emg[mep_amp_prestim_time]

                # Check if extracted data is not empty
                if mep_data.size == 0 or pi_data.size == 0:
                    raise ValueError("MEP or PI data is empty after indexing. Check the logical arrays and preprocessed_emg.")
            
                # initialize variables for MEP calculation
                loc_max = np.nan
                loc_min = np.nan
                mep_raw = np.nan

                time_step = 1

                # Calculate MEPs: find all peaks in data and get range of that
                pks_max, props_max = find_peaks(mep_data, prominence=prominence)
                pks_min, props_min = find_peaks(-mep_data, prominence=prominence)

                # If at least one peak of each type was found
                if len(pks_max) > 0 and len(pks_min) > 0:
                    valid_pairs = []

                    # Test all combinations of max and min peaks
                    for max_idx in pks_max:
                        for min_idx in pks_min:
                            sample_diff = abs(max_idx - min_idx)
                            peak_to_peak_amplitude = abs(mep_data[max_idx] - mep_data[min_idx])
                            if sample_diff >= min_required_distance and peak_to_peak_amplitude >= min_peak_to_peak_amplitude:
                                valid_pairs.append((max_idx, min_idx, peak_to_peak_amplitude))

                    if valid_pairs:
                    # Choose the pair with the largest peak-to-peak amplitude
                        loc_max, loc_min, mep_raw = max(valid_pairs, key=lambda x: x[2])
                    else:
                        loc_max = np.nan
                        loc_min = np.nan
                        mep_raw = np.nan

                else:
                    loc_max = np.nan
                    loc_min = np.nan
                    mep_raw = np.nan

                # Log scale (use np.log1p to avoid log(0) errors)
                mep_log = np.log1p(abs(mep_raw)) if not np.isnan(mep_raw) else np.nan

                # Check if this is a valid trial (non-NaN MEP)
                if not np.isnan(mep_raw):
                    valid_trials += 1
        

                # Calculate preinnervation using MAD
                def calculate_mad(data):
                    median = np.median(data)
                    return np.median(np.abs(data - median))
            
                preinnervation = calculate_mad(pi_data)

                # Store results in DataFrame
                new_row = pd.DataFrame({'mep_raw': [mep_raw], 'mep_log': [mep_log], 'loc_max': [loc_max], 'loc_min': [loc_min], 'preinnervation': [preinnervation]})
                results_df = pd.concat([results_df, new_row], ignore_index=True)

                # Update the threshold for preinnervation rejection based on the last 50 epochs
                if len(results_df) > 50:
                    recent_preinnervation = results_df['preinnervation'].iloc[-50:]
                    median_preinnervation = np.median(recent_preinnervation)
                    mad_preinnervation = calculate_mad(recent_preinnervation)
                
                    # Use a less strict threshold for preinnervation
                    pi_cutoff_upper = median_preinnervation + 5 * mad_preinnervation
                
                    # Determine rejection criteria
                    rejection = {}
                    rejection['preinnervation'] = (preinnervation > pi_cutoff_upper)
                    rejection['no_peak'] = np.isnan(mep_raw) 
                    rejection['all_mep_criteria'] = rejection['preinnervation'] | rejection['no_peak']
                else:
                    # If less than 200 trials, do not reject based on preinnervation
                    rejection = {}
                    rejection['preinnervation'] = False  # Changed to a single boolean
                    rejection['no_peak'] = np.isnan(mep_raw)
                    rejection['all_mep_criteria'] = rejection['preinnervation'] | rejection['no_peak']
                    pi_cutoff_upper = np.nan

                # Store the preinnervation threshold and rejection criteria
                preinnervation_thresholds.append(pi_cutoff_upper)
                rejections.append(rejection)

                # End timing the preprocessing
                end_time = time.time()

                # Calculate and store the preprocessing time
                preprocessing_time = end_time - start_time
                preprocessing_times.append(preprocessing_time)


                # Output the preprocessing time
                print(f"Processed one EEG and one EMG epoch in {preprocessing_time:.4f} seconds.")

                trial_number += 1

                # Add debug print every 100 trials
                if trial_number % 100 == 0:
                    print(f"Processed {trial_number} trials, {valid_trials} valid trials")
                    print(f"ICA calibration complete: {sound_preprocessor.ica_preprocessor.calibration_complete}")

        except KeyboardInterrupt:
            print("Real-time acquisition stopped by user.")

        finally:
            # Ensure to stop the real-time acquisition properly
            rt_epochs.stop(stop_receive_thread=False)
            rt_emg_epochs.stop(stop_receive_thread=False)


        # Extract data from results_df
        rejections_df = pd.DataFrame(rejections)
        results_df['preinnervation_threshold'] = preinnervation_thresholds

        mep_df = results_df[['mep_raw']]
        mep_log_df = results_df[['mep_log']]
        loc_max_df = results_df[['loc_max']]
        loc_min_df = results_df[['loc_min']]
        preinnervation_df = results_df[['preinnervation']]

        # Store results in a dictionary to mimic MATLAB structure
        parameters = {
            'mep_raw': results_df[['mep_raw']],
            'mep_log': results_df[['mep_log']],
            'loc_max': results_df[['loc_max']],
            'loc_min': results_df[['loc_min']],
            'preinnervation': results_df[['preinnervation']],
            'preinnervation_threshold': results_df[['preinnervation_threshold']],
            'rejections': rejections_df
        }

        # Z-score MEP log data
        mean_mep_log = np.nanmean(parameters['mep_log'])
        std_mep_log = np.nanstd(parameters['mep_log'])
        parameters['mep_log'] = (parameters['mep_log'] - mean_mep_log) / std_mep_log

        # Z-score preinnervation data
        mean_preinnervation = np.nanmean(parameters['preinnervation'])
        std_preinnervation = np.nanstd(parameters['preinnervation'])
        parameters['preinnervation'] = (parameters['preinnervation'] - mean_preinnervation) / std_preinnervation

        # Z-score preinnervation threshold data
        mean_preinnervation_threshold = np.nanmean(parameters['preinnervation_threshold'])
        std_preinnervation_threshold = np.nanstd(parameters['preinnervation_threshold'])
        parameters['preinnervation_threshold'] = (parameters['preinnervation_threshold'] - mean_preinnervation_threshold) / std_preinnervation_threshold

        epochs_emg = mne.Epochs(raw_data, filtered_events, event_id=event_id_out, tmin=EMG_EPOCH_LIMITS[0],
                        tmax=EMG_EPOCH_LIMITS[1], baseline=EMG_BASELINE_WIN, preload=True, picks=emg_picks, verbose=False)


        x_emg = epochs_emg_times[mep_amp_limits]

        trials = len(mep_data_dict)
        timepoints = len(next(iter(mep_data_dict.values())))  # Assuming all trials have the same length
        mep_data_array = np.zeros((trials, timepoints))

        for trial in range(trials):
            mep_data_array[trial, :] = mep_data_dict[trial]


        # Convert rejections_df columns to boolean type
        rejections_df['preinnervation'] = rejections_df['preinnervation'].astype(bool)
        rejections_df['no_peak'] = rejections_df['no_peak'].astype(bool)
        rejections_df['all_mep_criteria'] = rejections_df['all_mep_criteria'].astype(bool)

        # Plotting
        # Initialize the figure
        """
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs = axs.flatten()

        # Plot rejection histogram
        axs[0].hist([0, 1, 2], bins=[0, 1, 2, 3], weights=[sum(rejections_df['preinnervation']), sum(rejections_df['no_peak']), sum(rejections_df['all_mep_criteria'])], color='k')
        axs[0].set_xticks([0.5, 1.5, 2.5])
        axs[0].set_xticklabels(['Preinnervation', 'No Peak', 'All Criteria'])
        axs[0].set_ylim([0, 800])
        axs[0].set_ylabel('Number of trials rejected')
        axs[0].set_title(f'Rejection across subject {subject_number}')
        for x, y in zip([0.5, 1.5, 2.5], [sum(rejections_df['preinnervation']), sum(rejections_df['no_peak']), sum(rejections_df['all_mep_criteria'])]):
            axs[0].text(x, y + 50, str(y), ha='center')

        # Plot MEPs
        x = np.arange(len(results_df))
        axs[1].plot(x[~rejections_df['all_mep_criteria'].values], parameters['mep_log'][~rejections_df['all_mep_criteria'].values], 'k.', linewidth=1)
        axs[1].plot(x[rejections_df['preinnervation'].values], parameters['mep_log'][rejections_df['preinnervation'].values], 'c.', linewidth=1)
        axs[1].plot(x[rejections_df['no_peak'].values], parameters['mep_log'][rejections_df['no_peak'].values], 'm.', linewidth=1)
        axs[1].set_ylabel('MEP (z-score)')
        axs[1].set_xlabel('Trial number')
        axs[1].set_title(f'Distribution of MEP across subject {subject_number}')

        # Plot preinnervation
        x = np.arange(len(results_df))
        # Plotting the existing data points
        axs[2].semilogy(x[~rejections_df['all_mep_criteria'].values], results_df['preinnervation'][~rejections_df['all_mep_criteria'].values], 'k.')
        axs[2].semilogy(x[rejections_df['preinnervation'].values], results_df['preinnervation'][rejections_df['preinnervation'].values], 'c.')
        axs[2].semilogy(x[rejections_df['no_peak'].values], results_df['preinnervation'][rejections_df['no_peak'].values], 'm.')
        axs[2].grid(False)

        # Plot pre-innervation threshold for each trial
        for i, cutoff in enumerate(results_df['preinnervation_threshold']):
            if not np.isnan(cutoff):
                axs[2].axhline(y=cutoff, color='c', linestyle='--', linewidth=2, xmin=i/len(results_df), xmax=(i+1)/len(results_df))

        axs[2].set_ylabel('Preinnervation [µV]')
        axs[2].set_xlabel('Trial number')
        axs[2].set_title(f'Distribution of Preinnervation across subject {subject_number}')
        axs[2].legend()

        # Plot EMG response in peak-to-peak window
        x_emg = x_emg
        y_emg = mep_data_array.T
        axs[3].plot(x_emg, y_emg[:, ~rejections_df['all_mep_criteria'].values], color=[0, 0, 0, 0.2])
        axs[3].plot(x_emg, y_emg[:, rejections_df['preinnervation'].values], color=[0, 1, 1, 0.2])
        axs[3].plot(x_emg, y_emg[:, rejections_df['no_peak'].values], color=[1, 0, 1, 0.2])
        axs[3].set_ylabel('EMG response [µV]')
        axs[3].set_xlabel('t w.r.t. TMS pulse [ms]')
        axs[3].set_title(f'EMG response for subject {subject_number}')

        # Show peak locations
        valid_locs_max = results_df['loc_max'][~rejections_df['all_mep_criteria'].values].dropna().astype(int)
        valid_locs_min = results_df['loc_min'][~rejections_df['all_mep_criteria'].values].dropna().astype(int)
        temp1 = np.ravel_multi_index((valid_locs_max, np.where(~rejections_df['all_mep_criteria'].values)[0][:len(valid_locs_max)]), dims=y_emg.shape)
        temp2 = np.ravel_multi_index((valid_locs_min, np.where(~rejections_df['all_mep_criteria'].values)[0][:len(valid_locs_min)]), dims=y_emg.shape)
        axs[3].scatter(x_emg[valid_locs_max], y_emg.ravel()[temp1], marker='^', color='k', s=15, label='Max Peak')
        axs[3].scatter(x_emg[valid_locs_min], y_emg.ravel()[temp2], marker='v', color='k', s=15, label='Min Peak')
        axs[3].legend()

        # Save the plot
        plot_filename = os.path.join(mep_directory, f'subject_{subject_number}_mep_plots_online.png')
        plt.savefig(plot_filename)

        # Show the plot
        plt.tight_layout()
        plt.show()
        """

        # Save mep_log to a CSV file
        meps_final = results_df['mep_log'].to_numpy()
        meps_final = meps_final.reshape(1, -1)
        save_path = os.path.join(save_directory, f'subject_{subject_number}_mep_py_rt_interaction_{freq_band}.csv')
        np.savetxt(save_path, meps_final, delimiter=',', fmt='%.6f')

        # Convert the collected preprocessed EEG epochs to an MNE Epochs object
        epochs_info = epochs_eeg.info

        events = np.array(filtered_events)  # Ensure  events array is in the correct format
        # Truncate each array to have exactly 1000 samples
        truncated_epochs = []
        for epoch in preprocessed_eeg_epochs:
            truncated_epochs.append(epoch[:, :1000])

        # Convert the truncated list to a NumPy array
        preprocessed_eeg_epochs_array = np.array(truncated_epochs)
        epochs_preprocessed = mne.EpochsArray(preprocessed_eeg_epochs_array, epochs_info, events, tmin=EEG_EPOCH_LIMITS[0])


        #save_path = os.path.join(save_directory, f'subject_{subject_number}_eeg_preprocessed_pen.fif')
        save_path = os.path.join(save_directory, f'subject_{subject_number}_bandpass_{freq_band}_eeg_preprocessed_test_py.fif')
        epochs_preprocessed.save(save_path, overwrite=True)

        #save_path = os.path.join(save_directory, f'subject_{subject_number}_freq_interaction_{freq_bands}_eeg_preprocessed_py_interaction.fif')
        #epochs_preprocessed.save(save_path, overwrite=True)

        print(f"Completed processing for subject {subject_number}")

print("Finished processing all subjects")


