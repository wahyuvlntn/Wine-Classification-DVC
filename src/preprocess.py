import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Load dataset
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Split and standardize data
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params['train']['test_size'], random_state=params['train']['random_state']
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed data
train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
train_df['target'] = y_train.values
train_df.to_csv('data/wine_train.csv', index=False)

test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
test_df['target'] = y_test.values
test_df.to_csv('data/wine_test.csv', index=False)
