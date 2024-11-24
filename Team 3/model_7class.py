import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # Used to save the model
import numpy as np

# Load the dataset
df = pd.read_csv('train.csv')

# Preprocess data
# Map categorical features to numeric
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['family_history_with_overweight'] = df['family_history_with_overweight'].map({'no': 0, 'yes': 1})
df['FAVC'] = df['FAVC'].map({'no': 0, 'yes': 1})
df['CAEC'] = df['CAEC'].map({'no': 0, 'some': 1, 'full': 2})
df['SMOKE'] = df['SMOKE'].map({'no': 0, 'yes': 1})
df['SCC'] = df['SCC'].map({'no': 0, 'yes': 1})
df['CALC'] = df['CALC'].map({'no': 0, 'some': 1, 'full': 2})
df['MTRANS'] = df['MTRANS'].map({'motorbike': 0, 'bike': 1, 'public': 2, 'walking': 3})

# Calculate BMI
df['BMI'] = df['Weight'] / (df['Height'] ** 2)

# Define the feature columns
feature_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 
                   'Age', 'Height', 'Weight', 'BMI', 'CH2O', 'FAF', 'FCVC', 'NCP', 'TUE']

# Apply Label Encoding to the target column
label_encoder = LabelEncoder()
df['NObeyesdad'] = label_encoder.fit_transform(df['NObeyesdad'])  # Encode target labels as integers

# Feature and target variables
X = df[feature_columns]  # Use all 16 columns
y = df['NObeyesdad']  # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the XGBoost model
model = xgb.XGBClassifier(objective='multi:softmax', num_class=7, max_depth=6, learning_rate=0.1, n_estimators=100, n_jobs=-1)

# Define the hyperparameter grid for RandomizedSearchCV
param_dist = {
    'learning_rate': np.linspace(0.01, 0.2, 10),
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.1, 0.5, 1]
}

# Use RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, verbose=1, n_jobs=-1, random_state=42)

# Fit the model using RandomizedSearchCV
random_search.fit(X_train_scaled, y_train)

# Get the best model
best_model = random_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Save the best model and scaler
joblib.dump(best_model, 'model_7_category.pkl')
joblib.dump(scaler, 'scaler_7_category.pkl')
print("Model and scaler saved successfully!")