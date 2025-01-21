import mne
import os
import numpy as np
import pandas as pd
#from imbalanced_ensemble.ensemble import BalancedRandomForestClassifier
#(imbalanced_ensemble is having compatibility issues with scikit-learn version)
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from mne.time_frequency import psd_welch

#1. loading eeg data
parent_folder_path = r"/Users/balabheem/Downloads/ds004504"
tsv_file_path = r"/Users/balabheem/Downloads/ds004504/participants.tsv"

labels_df = pd.read_csv(tsv_file_path, sep='\t', header = None) #read file into pandas data frame

label_mapping = {
    "A" : 0,
    "F" : 1,
    "C" : 2
}

subject_label_dict = {row[0]:label_mapping.get(row[3], -1) for row in labels_df.values}

def load_all_set_files(parent_folder_path):
    eeg_data = []
    subject_ids = []
    
    for i in range (1,89):
        folder_name = f"sub-{i:03d}"
        eeg_folder = os.path.join(parent_folder_path, folder_name, "eeg")
        if os.path.exists(eeg_folder):
            for file in os.listdir(eeg_folder):
                if file.endswith(".set"):
                    file_path = os.path.join(eeg_folder, file)
                    print(f"Loading {file_path}...")
                    raw = mne.io.read_raw_eeglab(file_path, preload = True)
                    eeg_data.append(raw)
                    subject_ids.append(folder_name)
    return eeg_data, subject_ids

#2.preprocess data
def preprocess_and_extract_epochs(raw_data_list):
    epochs_list = []
    for raw in raw_data_list:
        #preprocess
        raw.filter(1., 40., fir_design ='firwin') #BP
        raw.set_eeg_reference('average')  #re-reference to average
        #epoch extraction
        events = mne.make_fixed_length_events(raw, duration =2.0)
        epochs = mne.Epochs(raw, events, tmin = 0, tmax = 2.0, baseline = None, preload = True)
        epochs_list.append(epochs)
    return epochs_list

#3.extracting features from data (mean, standard deviation, skewness, kurtosis, median etc.)
def extract_features_from_epochs(epochs_list):
    features = []
    for epochs in epochs_list:
        for epoch in epochs.get_data():
            epoch_features = []
            
            #features of time-domain
            for channel in epoch:
                epoch_features.extend([
                    np.mean((channel)),
                    np.std(channel),
                    skew(channel),
                    kurtosis(channel),
                    np.median(channel),
                    np.sqrt(np.mean(np.square(channel))),
                    np.ptp(channel),
                    np.sum(np.abs(np.diff(channel)))
                ])
            
            
            #features of frequency-domain
            psd, freqs = psd_welch(epochs, fmin = 1., fmax = 40., picks = 'all')
            psd_features = np.mean(psd, axis = 2)
            epoch_features.extend(psd_features.flatten())
            
            #frequency band power
            band_powers = []
            for band in [(0.5, 4), (4, 8), (8, 13), (13, 30)]:
                band_psd = np.mean(psd[(freqs >= band[0]) & (freqs <= band[1])]),
                band_powers.extend(band_psd)
            epoch_features.extend(band_powers)
            
            features.append(epoch_features)
            
    return np.array(features)



#loading all files
all_eeg_data, all_subject_id = load_all_set_files(parent_folder_path)

#preprocess data & extract epochs
all_epochs = preprocess_and_extract_epochs(all_eeg_data)

#extract features
X_features = extract_features_from_epochs(all_epochs)

#4.now assign labels for subject ids (* here specifically as all_epochs are defined)
X = X_features
y = []

#to assign label for each epoch  from subject's label
for subject_id in all_subject_id:
    subject_label = subject_label_dict.get(subject_id, -1)
    #for all epochs corresponding to subjects
    y.extend([subject_label] * len(all_epochs[all_subject_id.index(subject_id)]))

#converting one-hot labels back to integers
#y_brf = np.argmax(y, axis = 1)

#5.test_train_split
X_train_brf, X_test_brf, y_train_brf, y_test_brf, = train_test_split(X_features, y, test_size = 0.25, random_state = 42)

#Initializing the BalancedRandomForest classifier
brf_clf = BalancedRandomForestClassifier(n_estimators = 100, random_state = 42)

#training the model
brf_clf.fit(X_train_brf, y_train_brf)

#predict the model
y_pred_brf = brf_clf.predict(X_test_brf)

#evaluate performance metrics
print("Balanced Random Forest Classification Report: ")
print(classification_report(y_test_brf, y_pred_brf))

#accuracy
accuracy = accuracy_score(y_test_brf, y_pred_brf)
print(f"Accuracy: {accuracy:.4f}")

#confusion matrix
cm_brf = confusion_matrix(y_test_brf, y_pred_brf)
sns.heatmap(cm_brf, annot = True, fmt = 'd', cmap = 'Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Balanced Random Forest Classifier)')
plt.show()

