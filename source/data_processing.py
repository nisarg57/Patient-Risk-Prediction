

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    """
    Applies basic preprocessing steps to the dataframe.
    - Imputes missing numerical values with the mean.
    - Imputes missing categorical values with the most frequent.
    - One-hot encodes categorical features.
    - Scales numerical features using StandardScaler.
    """
    # Separate target variable if it exists (assuming 'Outcome' or 'Diabetes' as target)
    target_column = 'Outcome' # Adjust based on your dataset's target column name
    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df
        y = None # No target column found

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any
    )

    X_processed = preprocessor.fit_transform(X)

    # To get back a DataFrame with column names (optional, for easier inspection)
    # This part can be tricky with OneHotEncoder.
    # For a beginner project, returning the numpy array X_processed is often sufficient,
    # or you can get feature names after fitting the one-hot encoder.
    # For simplicity, we'll return the processed numpy array and original y.

    return X_processed, y, preprocessor # Return preprocessor for potential inverse transform or feature names

if __name__ == '__main__':
    # Example Usage (replace with your actual data loading)
    # Create a dummy dataframe for demonstration
    data = {
        'Glucose': [100, 120, 80, None, 150],
        'BMI': [25.5, 30.1, None, 22.0, 35.0],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Age': [30, 45, 25, 50, 60],
        'Outcome': [0, 1, 0, 1, 1]
    }
    dummy_df = pd.DataFrame(data)

    print("Original DataFrame:\n", dummy_df)
    X_processed_example, y_example, preprocessor_example = preprocess_data(dummy_df)
    print("\nProcessed X shape:", X_processed_example.shape)
    print("Processed y shape:", y_example.shape if y_example is not None else "No target")

    # You can inspect the feature names created by the preprocessor
    # This requires more advanced handling of ColumnTransformer
    # For this beginner project, focusing on the transformed array is fine.