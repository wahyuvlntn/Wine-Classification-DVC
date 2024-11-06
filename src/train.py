import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import yaml

os.makedirs("model", exist_ok=True)

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Load training data
train_df = pd.read_csv('data/wine_train.csv')
X_train = train_df.drop(columns=['target'])
y_train = train_df['target']

test_df = pd.read_csv('data/wine_test.csv')
X_test = test_df.drop(columns=['target'])
y_test = test_df['target']

# Train model
model = LogisticRegression(max_iter=params['train']['model']['max_iter'])
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

# Save model
joblib.dump(model, 'model/model.pkl')
