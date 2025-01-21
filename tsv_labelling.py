import pandas as pd

#data from the below link is considered for this project
https://openneuro.org/datasets/ds004504/version/1.0.8/download

# Path to the .tsv file that contains subject ids and corresponding labels
tsv_file_path = r"/Users/balabheem/Downloads/ds004504/participants.tsv"

# Read the tsv file into a pandas DataFrame
labels_df = pd.read_csv(tsv_file_path, sep='\t', header = None)

# Example of the .tsv file format:
# sub_id    label
# sub-001   A
# sub-002   F
# sub-003   C
# ...

# Map labels from A, F, C to 0, 1, 2
label_mapping = {
    "A": 0,  # Alzheimer's
    "F": 1,  # Frontotemporal Dementia
    "C": 2   # Control (Healthy)
}

# Create a dictionary from subject IDs to labels
subject_to_label = {row[0]:label_mapping.get(row[3], -1) for row in labels_df.values}

# Initialize the labels array
y = []

# Now assign the labels based on subject IDs
for i, epochs in enumerate(all_epochs):
    epoch_data = epochs.get_data()  # Get the data as (n_epochs, n_channels, n_times)
    subject_id = all_subject_id[i]  # Get the subject ID (e.g., 'sub-001')
    
    # Get the corresponding label for the subject
    subject_label = subject_to_label.get(subject_id, 2)  # Default to 2 (Healthy) if not found
    
    # Assign the same label to all epochs of this subject
    y.extend([subject_label] * epoch_data.shape[0])

# Convert y to a numpy array
y = np.array(y)

# One-hot encode the labels
y = to_categorical(y, num_classes=3)

# Check the distribution of labels
print("Label distribution:", np.unique(y, return_counts=True))
