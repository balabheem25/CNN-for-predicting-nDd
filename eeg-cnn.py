import mne
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from scipy.signal import spectrogram
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



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

#3.convert epochs to spectrograms for giving it as input to CNN
def extract_spectrograms(epochs_list):
    all_spectrograms = []
    for epochs in epochs_list:
        for epoch in epochs.get_data():   #gets epochs as (n_epochs, n_channels, n-times)
            epoch_spectrograms = []
            for channel in epoch:
                f, t, Sxx = spectrogram(channel, fs = epochs.info['sfreq'], nperseg = 128, noverlap = 64) 
                # added nperseg and noverlap after accuracy is 42.01 ti get more detailed representation of signal
                epoch_spectrograms.append(Sxx)
            #now to stack channel spectrogram to form a 3D array (n_channels, n_frequency, n_times)
            all_spectrograms.append(np.array(epoch_spectrograms))
    return np.array(all_spectrograms)

#loading all files
all_eeg_data, all_subject_id = load_all_set_files(parent_folder_path)

#preprocess data & extract epochs
all_epochs = preprocess_and_extract_epochs(all_eeg_data)
#spectrograms
spectrograms = extract_spectrograms(all_epochs)

#4.now assign labels for subject ids (* here specifically as all_epochs are defined)
X = []
y = []

for i, epochs in enumerate(all_epochs):
    epoch_data = epochs.get_data()
    X.extend(epoch_data)
        
    subject_id = all_subject_id[i]
    label = subject_label_dict.get(subject_id, 2)
    y.extend([label] * epoch_data.shape[0])

#5.adjusting learning rate top reduce as the epochs progress

def lr_schedule(epoch):
    if epoch < 5:
        return 0.001 #high learning rate for initial epochs
    else:
        return 0.0001 #low learning rate 

#initializing the learning rate scheduler callback here to avoid confusion and misplacement
lr_scheduler = LearningRateScheduler(lr_schedule)

X = np.array(X)
y = np.array(y)

#Reshaping spectrograms for CNN input
spectrograms = (spectrograms - np.mean(spectrograms)) / np.std(spectrograms) #after accuracy 42.017
X = spectrograms #shape: (n_samples, n_channels, n_frequencies, n_times)
X = X.transpose(0, 2, 3, 1) #reshaped to (n_samples, n_frequencies(height), n_times(width), n_channels)
####n_samples = X.shape[0]
####y = np.array([0] * (n_samples//3) + [1] * (n_samples//3) + [2] + (n_samples//3))
y = to_categorical(y, num_classes = 3)
#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#defining the CNN model
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation = 'relu', input_shape = input_shape, padding = 'same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation = 'relu', padding = 'same'), #added this after accuracy 42.017
        Flatten(),
        Dense(256, activation = 'relu'),
        Dropout(0.5),
        Dense(num_classes, activation = 'softmax')
    ])
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

#build & train model
input_shape = X_train.shape[1:]
num_classes = y_train.shape[1]
cnn_model = build_cnn_model(input_shape, num_classes)

cnn_model.fit(X_train,
              y_train,
              epochs = 20,
              batch_size = 50,
              #needs to check with 32_(accuracy = 81.3003, 80.4833)
              # 64_(accuracy = 79.7685)
              # and 128_(accuracy = 74.9687) and note with accuracy
              validation_data = (X_test, y_test),
              callbacks = [lr_scheduler] # added the learning rate scheduler in later stages for better training
              )

#evaluate the model

loss, accuracy = cnn_model.evaluate(X_test, y_test)
print("Accuracy: ", accuracy)

#confusion matrix
y_pred = cnn_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis = 1)
y_true = np.argmax(y_test, axis = 1)

cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

#######################################
#X = []
#y = []
# ##assigning lables
#label_mapping = {
#    "A": 0, #Azheimer's
#    "F":1, #Frontotemporal Dementia
#    "C":2 #Control (Healthy)
#}

#for i, epochs in enumerate(all_epochs):
#    epoch_data = epochs.get_data() # to get data as numpy array (n_epochs, n_channels, n_times)
#    X.extend(epoch_data)
#    
#    subject_label = label_mapping.get("C", 2) # replaces C with 2
#    y.extend([subject_label] * epoch_data.shape[0])

#X = np.array(X)
#y = np.array(y)

#print("Training set shape: ", X_train.shape)
#print("Test set shape: ", X_test.shape)
#############################################
