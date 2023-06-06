# Obesity-classification

## In this project, we aimed to classify individuals into different classes based on various features using four different classification algorithms. Here's a summary of the steps we performed:

### 1. Data Loading:
We loaded the dataset containing features such as Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, and MTRANS.

### 2. Data Preprocessing: 
We handled missing values by filling them with the column mean or an appropriate strategy. Categorical variables were encoded using LabelEncoder to convert them into numerical values.

### 3. Feature Scaling:
We applied feature scaling using StandardScaler to standardize the feature values, ensuring that they have zero mean and unit variance. This step helps in improving the performance of many machine learning algorithms.

### 4. Train-Test Split:
We split the dataset into training and testing sets, with 80% of the data used for training and 20% for testing. This allows us to evaluate the performance of the classification models on unseen data.

### 5. Model Training: 
We used the Random Forest classifier from scikit-learn and created an instance of the model with 100 estimators (decision trees) and a random state of 42. The Random Forest algorithm is an ensemble method that combines multiple decision trees to make predictions.

### 6. Model Evaluation: 
We made predictions on the testing set using the trained Random Forest classifier. We evaluated the performance of the model using the classification report, which provides metrics such as precision, recall, F1-score, and support for each class. This report helps us assess the model's accuracy and performance for each class.

By following these steps, we built a classification model that can predict the class labels for individuals based on their features. The Random Forest classifier was chosen as it is a robust and powerful algorithm that can handle both numerical and categorical features effectively. However, feel free to explore and experiment with other classification algorithms to find the one that best suits your data and yields the highest performance.
