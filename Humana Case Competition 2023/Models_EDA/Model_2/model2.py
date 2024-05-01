import pandas as pd
import xgboost as xgb
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV
import optuna
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns


# Function to save the confusion matrix
def save_confusion_matrix_as_image(conf_matrix, labels, filename):
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.tight_layout()
    plt.savefig(filename, format='jpg')
    plt.show()

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

# Feature Engineering
data['ade_claims_count'] = data.groupby('therapy_id')['ade_diagnosis'].transform('sum')

# Data Preprocessing
logger.info('Preprocessing data...')
numeric_data = data.select_dtypes(include=['float64', 'int64'])
categorical_data = data.select_dtypes(exclude=['float64', 'int64', 'object']).astype('str')

# Imputation for numeric columns
imputer_numeric = SimpleImputer(strategy='mean')
numeric_data = pd.DataFrame(imputer_numeric.fit_transform(numeric_data), columns=numeric_data.columns)

# Encoding for categorical columns
encoder = OneHotEncoder(drop='first')
categorical_data_encoded = encoder.fit_transform(categorical_data)
encoded_cols = encoder.get_feature_names_out(categorical_data.columns)
categorical_data_df = pd.DataFrame(categorical_data_encoded.toarray(), columns=encoded_cols)

# Combining processed columns
data = pd.concat([numeric_data, categorical_data_df], axis=1)

X = data.drop(['tgt_ade_dc_ind', 'id'], axis=1, errors='ignore')
y = data['tgt_ade_dc_ind']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

logger.info('Setting up XGBoost classifier...')
xgb_classifier = xgb.XGBClassifier(eval_metric="logloss")

# Use StratifiedKFold for cross-validation
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameter tuning
logger.info('Starting hyperparameter tuning with Optuna...')

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'subsample': trial.suggest_float('subsample', 0.8, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1)
    }
    xgb_classifier_optuna = xgb.XGBClassifier(**param, eval_metric="logloss")
    return cross_val_score(xgb_classifier_optuna, X_train, y_train, cv=cv_strategy, scoring='accuracy').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)

best_params = study.best_params
logger.info(f"Best Parameters found: {best_params}")

# Feature Selection using RFECV
logger.info('Starting feature selection...')
xgb_classifier_for_rfe = xgb.XGBClassifier(**best_params, eval_metric="logloss")
selector = RFECV(estimator=xgb_classifier_for_rfe, step=1, cv=cv_strategy)
selector = selector.fit(X_train, y_train)

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Training the model with the best parameters and selected features
logger.info('Training model...')
xgb_classifier_final = xgb.XGBClassifier(**best_params, eval_metric="logloss")
xgb_classifier_final.fit(X_train_selected, y_train)

# Predictions and Evaluations
y_pred = xgb_classifier_final.predict(X_test_selected)


# Evaluate model

dump(xgb_classifier_final, 'xgboost_best_model.joblib')
logger.info('Best model saved as xgboost_best_model.joblib')

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

logger.info(f"Accuracy: {accuracy}")
logger.info("Classification Report:")
logger.info(str(report))
logger.info("Confusion Matrix:")
logger.info("\n" + str(conf_matrix))

# Save confusion matrix as image
labels = y_test.unique()
save_confusion_matrix_as_image(conf_matrix, sorted(labels), "confusion_matrix2.jpg")

