import pandas as pd

# Load train datasets
target_train = pd.read_csv('2023_TAMU_competition_data/target_train.csv')
medclms_train = pd.read_csv('2023_TAMU_competition_data/medclms_train.csv')
rxclms_train = pd.read_csv('2023_TAMU_competition_data/rxclms_train.csv')

# Load holdout datasets
target_holdout = pd.read_csv('2023_TAMU_competition_data/target_holdout.csv')
medclms_holdout = pd.read_csv('2023_TAMU_competition_data/medclms_holdout.csv')
rxclms_holdout = pd.read_csv('2023_TAMU_competition_data/rxclms_holdout.csv')

# Group datasets for exploration
datasets = {
    "target_train": target_train,
    "medclms_train": medclms_train,
    "rxclms_train": rxclms_train,
    "target_holdout": target_holdout,
    "medclms_holdout": medclms_holdout,
    "rxclms_holdout": rxclms_holdout
}

# Explore each dataset
for dataset_name, dataset in datasets.items():
    print(f"Exploring {dataset_name}...\n")
    
    # 1. Column types
    print("Column Types:\n", dataset.dtypes, "\n")
    
    # 2. Identifying columns that have missing values
    missing_values = dataset.isnull().sum()
    columns_with_na = missing_values[missing_values > 0]
    # 3. Print the number of therapy_id
    print(datasets['therapy_id'].value_counts())
    if not columns_with_na.empty:
        print("Columns with missing values:\n", columns_with_na, "\n")
    else:
        print(f"No columns with missing values in {dataset_name}.\n")
    
    print("="*50)
