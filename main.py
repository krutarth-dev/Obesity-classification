import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv("/Users/apple/Documents/Projects/obesity/ObesityDataSet_raw_and_data_sinthetic.csv")  # Replace "your_dataset.csv" with the actual file name

# Preprocess the data
# Handle missing values
data.fillna(data.mean(), inplace=True)  

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])  # Encode the 'Gender' column
data['family_history_with_overweight'] = le.fit_transform(data['family_history_with_overweight'])  # Encode 'family_history_with_overweight' column
data['FAVC'] = le.fit_transform(data['FAVC'])  # Encode 'FAVC' column
data['CAEC'] = le.fit_transform(data['CAEC'])  # Encode 'CAEC' column
data['SMOKE'] = le.fit_transform(data['SMOKE'])  # Encode 'SMOKE' column
data['SCC'] = le.fit_transform(data['SCC'])  # Encode 'SCC' column
data['CALC'] = le.fit_transform(data['CALC'])  # Encode 'CALC' column
data['MTRANS'] = le.fit_transform(data['MTRANS'])  # Encode 'MTRANS' column
data['NObeyesdad'] = le.fit_transform(data['NObeyesdad'])  # Encode the target variable 'NObeyesdad' column

# Split the data into features (X) and target variable (y)
X = data[['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC',
          'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']]
y = data['NObeyesdad']

# Scale the feature values using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scale the feature values using StandardScaler

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
