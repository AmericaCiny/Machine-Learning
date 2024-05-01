import pandas as pd

# Load datasets
target_train = pd.read_csv('2023_TAMU_competition_data/target_train.csv')
medclms_train = pd.read_csv('2023_TAMU_competition_data/medclms_train.csv')
rxclms_train = pd.read_csv('2023_TAMU_competition_data/rxclms_train.csv')

# Merge datasets on therapy_id
merged_data = target_train.merge(medclms_train, on='therapy_id', how='left').merge(rxclms_train, on='therapy_id', how='left')

# Intra-group imputation by therapy_id
grouped = merged_data.groupby('therapy_id')
merged_data = grouped.transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

# For columns that are still missing after intra-group imputation, we can apply general imputation methods:
# Numeric Variables
merged_data['race_cd'].fillna(merged_data['race_cd'].median(), inplace=True)  # Using median
merged_data['est_age'].fillna(merged_data['est_age'].median(), inplace=True)
merged_data['metric_strength'].fillna(merged_data['metric_strength'].median(), inplace=True)
merged_data['rx_cost'].fillna(merged_data['rx_cost'].mean(), inplace=True)  # Using mean

# Categorical Variables
merged_data['sex_cd'].fillna("Unknown", inplace=True)
merged_data['gpi_drug_group_desc'].fillna("Missing", inplace=True)
merged_data['gpi_drug_class_desc'].fillna("Missing", inplace=True)
merged_data['hum_drug_class_desc'].fillna("Missing", inplace=True)

# Binary/Indicator Variables
merged_data['cms_disabled_ind'].fillna(0, inplace=True)  # assuming most are not disabled
merged_data['cms_low_income_ind'].fillna(0, inplace=True)  # assuming most are not receiving low-income subsidies

# Save merged dataset to merged.csv
merged_data.to_csv('merged.csv', index=False)

print("Merging and Imputation completed. 'merged.csv' saved.")


# Assuming your holdout datasets are named 'target_holdout.csv', 'medclms_holdout.csv', and 'rxclms_holdout.csv'

# Load datasets
target_holdout = pd.read_csv("2023_TAMU_competition_data/target_holdout.csv")
medclms_holdout = pd.read_csv("2023_TAMU_competition_data/medclms_holdout.csv")
rxclms_holdout = pd.read_csv("2023_TAMU_competition_data/rxclms_holdout.csv")

# Merge datasets on therapy_id
merged_holdout = target_holdout.merge(medclms_holdout, on='therapy_id', how='left').merge(rxclms_holdout, on='therapy_id', how='left')

# Intra-group imputation by therapy_id
grouped_holdout = merged_holdout.groupby('therapy_id')
merged_holdout = grouped_holdout.transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

# Use imputation values from training data:
merged_holdout['race_cd'].fillna(target_train['race_cd'].median(), inplace=True)
merged_holdout['est_age'].fillna(target_train['est_age'].median(), inplace=True)
merged_holdout['metric_strength'].fillna(rxclms_train['metric_strength'].median(), inplace=True)
merged_holdout['rx_cost'].fillna(rxclms_train['rx_cost'].mean(), inplace=True)

# Categorical Variables (using training data categories or "Unknown"/"Missing")
merged_holdout['sex_cd'].fillna("Unknown", inplace=True)
merged_holdout['gpi_drug_group_desc'].fillna("Missing", inplace=True)
merged_holdout['gpi_drug_class_desc'].fillna("Missing", inplace=True)
merged_holdout['hum_drug_class_desc'].fillna("Missing", inplace=True)

# Binary/Indicator Variables
merged_holdout['cms_disabled_ind'].fillna(0, inplace=True)
merged_holdout['cms_low_income_ind'].fillna(0, inplace=True)

# Save merged dataset to merged_holdout.csv
merged_holdout.to_csv('merged_holdout.csv', index=False)

print("Merging and Imputation for holdout data completed. 'merged_holdout.csv' saved.")
