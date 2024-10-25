import pandas as pd

# Load your dataset
train_data = pd.read_csv('datasets/train.tsv', sep='\t')

# Display the columns
print("Columns in train_data:", train_data.columns.tolist())
