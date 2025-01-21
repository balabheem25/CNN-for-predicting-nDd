import mne
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#1.File Loading
file_path = r"/Users/balabheem/Downloads/ds004504.set"
def load_eeg_data():
    raw =  mne.io.read_raw_eeglab(file_path, preload = True)
    return raw

raw_data = load_eeg_data(file_path)

#2.Preprocess {Band Pass Filtering (if from 1 to 40 Hz = 1., 40.,), Re-referencing, Epoching}

def preprocess_data(raw):
    raw.filter(1., 40., fir_design='firwin') #BPF
    raw.set_eeg_reference('average') #re-ref
    events = mne.make_fixed_length_events(raw, duration= 2.0) #epoch @ fixed-length
    epochs = mne.Epochs(raw, events, tmin=0, tmax=2.0, baseline=None, preload=True)
    return epochs

epochs = preprocess_data(raw_data)

#3.Extracting Features {Power Spectral Density, Band Power in specific frequency bands eg.
#                        delta, theta, alpha, beta, gamma}

def extracted_features(epochs):
    psd, freqs = mne.time_frequency.psd_multitaper(epochs, fmin=1., fmax=40.)
    psd_mean = np.mean(psd,  axis=-1)
    return psd_mean

features = extracted_features(epochs)
print("Features Shape: ", features.shape)

#4.Loading labels & preparing data (#0-Healthy, #1-Alzheimer's, #2-Dementia)

labels = [0]*50 + [1]*50 + [2]*50

#5.Training the classifier [split, train, evaluate]

X_train, X_test, y_train, y_test = train_test_slit(features, labels, test_size=0.25, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy: ", accuracy_score(y_test,y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


