import pandas as pd
import logging
import numpy as np
from joblib import load

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the holdout data
logger.info('Loading holdout datasets...')
target_holdout = pd.read_csv("2023_TAMU_competition_data/target_holdout.csv")
medclms_holdout = pd.read_csv("2023_TAMU_competition_data/medclms_holdout.csv")
rxclms_holdout = pd.read_csv("2023_TAMU_competition_data/rxclms_holdout.csv")

# Merge holdout datasets
logger.info('Merging holdout datasets...')
data_holdout = pd.merge(target_holdout, medclms_holdout, on="therapy_id", how="left")
data_holdout = pd.merge(data_holdout, rxclms_holdout, on="therapy_id", how="left")

# Preprocess the holdout data
logger.info('Preprocessing holdout data...')
data_holdout = data_holdout.select_dtypes(include=['float64', 'int64'])
data_holdout = data_holdout.dropna()

# Split the holdout data into features and the 'id' column
X_holdout = data_holdout.drop(['tgt_ade_dc_ind', 'therapy_id', 'id'], axis=1, errors='ignore')  # Exclude target and therapy_id for predictions
holdout_ids = data_holdout['id']  # Use id from target_holdout for submission

# Load the trained model
logger.info('Loading trained model...')
model = load('xgboost_model.joblib')

# Predict probabilities with the model for the positive class (1)
logger.info('Predicting probabilities with the trained model...')
y_prob_holdout = model.predict_proba(X_holdout)[:, 1]

# Get ranks for the scores
ranks = (-y_prob_holdout).argsort().argsort() + 1

# Create a submission dataframe
submission = pd.DataFrame({
    'id': holdout_ids,
    'score': y_prob_holdout,
    'rank': ranks
})

# Save the submission file
submission.to_csv("submission.csv", index=False)
logger.info('Submission saved as submission.csv')
