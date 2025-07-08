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
    # Debug: Print columns and check for 'class'
    print('preprocess_data: columns in df:', df.columns.tolist())
    print("preprocess_data: Does 'class' exist?", 'class' in df.columns)
    # Separate target variable if it exists (using 'class' as target for this dataset)
    target_column = 'class'  # Updated to match your dataset
    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df
        y = None # No target column found
    print('preprocess_data: y is None?', y is None)

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