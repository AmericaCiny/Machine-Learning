import pandas as pd
import xgboost as xgb
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info('Loading datasets...')
target_train = pd.read_csv("2023_TAMU_competition_data/target_train.csv")
medclms_train = pd.read_csv("2023_TAMU_competition_data/medclms_train.csv")
rxclms_train = pd.read_csv("2023_TAMU_competition_data/rxclms_train.csv")

logger.info('Merging datasets...')
data = pd.merge(target_train, medclms_train, on="therapy_id", how="left")
data = pd.merge(data, rxclms_train, on="therapy_id", how="left")

# Preprocessing
logger.info('Preprocessing data...')
data = data.select_dtypes(include=['float64', 'int64'])
data = data.dropna()

X = data.drop(['tgt_ade_dc_ind', 'id'], axis=1, errors='ignore')
y = data['tgt_ade_dc_ind']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

logger.info('Setting up XGBoost classifier...')
xgb_classifier = xgb.XGBClassifier(eval_metric="logloss")

# Hyperparameter tuning
logger.info('Performing GridSearchCV for hyperparameter tuning...')
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1],
    'colsample_bytree': [0.8, 0.9, 1]
}
grid_search = GridSearchCV(xgb_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

logger.info(f"Best Parameters found: {best_params}")

# Cross-validation
logger.info('Running cross-validation...')
xgb_classifier = xgb.XGBClassifier(**best_params, eval_metric="logloss")
cross_val_scores = cross_val_score(xgb_classifier, X_train, y_train, cv=5, scoring='accuracy')
logger.info(f"Cross-validation scores: {cross_val_scores}")

# Training the model with the best parameters
logger.info('Training model...')
xgb_classifier.fit(X_train, y_train)

# Predictions
y_pred = xgb_classifier.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
logger.info(f"Accuracy: {accuracy}")
logger.info("Classification Report:")
logger.info(report)
logger.info("Confusion Matrix:")
logger.info("\n" + str(conf_matrix))

# Save the model
dump(xgb_classifier, 'xgboost_model.joblib')
logger.info('Model saved as xgboost_model.joblib')
