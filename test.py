import mne

eeg = mne.io.read_raw_edf("D:/seizure_dataset/normal/chb-mit-scalp-eeg-database-1.0.0/chb01/chb01_01.edf")
print(eeg.info)
print(eeg.get_data().shape[1]/256)





