#%%
import pandas as pd
import numpy as np
import mne

# Define the paths to the data and MEP files
save_directory = r'/home/marco/Documents/GitHub/tms_eeg_decoding/data/bandpass'
mep_directory = r'/home/marco/Documents/GitHub/tms_eeg_decoding/data/bandpass'
#data_base_path = r'C:\Users\Lisa Haxel\Documents\NMI_Data\data\subject_{:01d}_eeg_preprocessed_pen.fif'
data_base_path = r'/home/marco/Documents/GitHub/tms_eeg_decoding/data/bandpass/subject_{:01d}_bandpass_{}_eeg_preprocessed_test_py.fif'
mep_base_path = r'/home/marco/Documents/GitHub/tms_eeg_decoding/data/bandpass/subject_{:01d}_mep_py_rt_interaction_{}.csv'

band_combinations = {
    
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


#subject_indices = [2]
# Loop over all subjects
#subject_indices = [1,2,3,4,7,9,11,13,14,18,22,24,25,26,27,29,31,33,34,35,39,41,42,43,45,46,47,48,51,52,53,55,56,57,59,60,62,63,65,66,67,69,70,71,72,73,74,76,79,80,83,86,88,92,100,102]
subject_indices =  [76, 83, 94, 85, 100, 103, 104, 106, 107, 108, 109, 110, 111]
for subject_index in subject_indices:
    for freq_bands, freq_bands_range in band_combinations.items():
    #for subject_index in subject_indices:

            # make a try except block to catch errors
        try:
            # Generate file paths for the current subject
            data_path = data_base_path.format(subject_index, freq_bands)
            mep_path = mep_base_path.format(subject_index, freq_bands)
        
            # Load the CSV file into a pandas DataFrame without a header
            mep_data = pd.read_csv(mep_path, header=None)

            # Load data (MNE epochs object)
            epochs = mne.read_epochs(data_path)

            # rereference the data to the average reference
            #äepochs.set_eeg_reference('average', projection=False)

            # Create a boolean mask for bad trials with NaN values in the MEP data
            mep_data_mask = mep_data.isna().any(axis=0)

            # Ensure both masks are numpy arrays for element-wise logical operations
            mep_data_mask = mep_data_mask.values if isinstance(mep_data_mask, pd.Series) else mep_data_mask

            # Get the indices of the bad trials
            bad_trial_indices = np.where(mep_data_mask)[0]
            # Filter out the bad trials from the MNE epochs
            epochs = epochs.drop(bad_trial_indices)
            # Filter out the bad trials from mep_data
            filtered_mep_data = mep_data.loc[:, ~mep_data_mask]

            # Flatten the filtered MEP data to get individual MEP sizes
            flattened_mep_data = filtered_mep_data.values.flatten()

            # Calculate the median MEP size across all trials
            median = np.median(flattened_mep_data)

            # Create binary MEP sizes
            mep_sizes_binary = np.array([0 if mep < median else 1 for mep in flattened_mep_data])

            # Assuming `epochs` is already loaded with your EEG data
            events = epochs.events.copy()

            #    Check that mep_sizes_binary has the same length as the number of trials
            assert len(mep_sizes_binary) == len(events), "Mismatch between mep_sizes_binary and number of trials in epochs."

            # Update the event IDs based on the mep_sizes_binary
            for i, binary_value in enumerate(mep_sizes_binary):
                events[i, 2] = 0 if binary_value == 0 else 1  # Use integer event IDs
        
            # Update the epochs object with the new events
            epochs.events = events

            # Define the event_id mapping
            event_ids = {'low': 0, 'high': 1}
            epochs.event_id = event_ids

            # Remove the existing metadata
            epochs.metadata = None

            # Check if the length of flattened MEP data matches the number of epochs
            assert len(flattened_mep_data) == len(epochs), "Length of MEP data does not match the number of epochs."

            # Create a new metadata DataFrame
            epochs.metadata = pd.DataFrame(index=range(len(epochs)))

            # Add the flattened MEP data as a new column in the metadata
            epochs.metadata['mep_size'] = flattened_mep_data
    
            # Check the number of trials after combining them
            if len(epochs) < 500:
                #save_path = f'{save_directory}/subject_{subject_index:03d}_preprocessed_combined_INSUFFICIENT_pen.fif'
                save_path = f'{save_directory}/subject_{subject_index:03d}_bandpass_freq_interaction_{freq_bands}_preprocessed_combined_INSUFFICIENT_py.fif'
            else:
                #save_path = f'{save_directory}/subject_{subject_index:03d}_preprocessed_combined_pen.fif'
                save_path = f'{save_directory}/subject_{subject_index:03d}_bandpass_freq_interaction_{freq_bands}_preprocessed_combined_py.fif'

            # Save the updated epochs object
            epochs.save(save_path, overwrite=True)

            # Verify the addition
            print(f"Subject {subject_index:03d} metadata:")
            print(epochs.metadata.head())

            # Catch any exceptions
        except Exception as e:
            print(f"Error processing subject {subject_index:03d}: {e}")


# %%
