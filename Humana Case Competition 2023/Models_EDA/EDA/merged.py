import pandas as pd

# Load the CSV files
target_holdout = pd.read_csv("2023_TAMU_competition_data/target_holdout.csv")
medclms_holdout = pd.read_csv("2023_TAMU_competition_data/medclms_holdout.csv")
rxclms_holdout = pd.read_csv("2023_TAMU_competition_data/rxclms_holdout.csv")

# Count therapy_id values in target_holdout
target_therapy_id_counts = target_holdout['therapy_id'].value_counts()

# Check the frequency of each therapy_id in the other datasets
therapy_id_counts_medclms = medclms_holdout['therapy_id'].value_counts()
therapy_id_counts_rxclms = rxclms_holdout['therapy_id'].value_counts()

# Filter to get counts of therapy_id values from target_holdout in the other datasets
common_counts_medclms = therapy_id_counts_medclms[therapy_id_counts_medclms.index.isin(target_therapy_id_counts.index)]
common_counts_rxclms = therapy_id_counts_rxclms[therapy_id_counts_rxclms.index.isin(target_therapy_id_counts.index)]

print("Count of therapy_id values in target_holdout:")
print(target_therapy_id_counts)

print("\nCounts of therapy_id values from target_holdout in medclms_holdout:")
print(common_counts_medclms)

print("\nCounts of therapy_id values from target_holdout in rxclms_holdout:")
print(common_counts_rxclms)
