
import numpy as np
import pandas as pd
from sklearn.compose         import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler, OneHotEncoder


# data validation function
def validate_data(df:pd.DataFrame, outcome, numerical_features:list, categorical_features:list)-> None:
    """
    validate the dataset before processing
    """
    print("=== validation ===")
    print(f"dataset shape: {df.shape}")
    
    # check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("\nmissing values:")
        print(missing_values[missing_values > 0])
    else:
        print("\ndata contains no missing values ✓")
    
    # check target distribution
    if outcome in df.columns:
        target_dist = df[outcome].value_counts()
        conversion_rate = df[outcome].mean()
        print(f"\ntarget distribution:")
        print(f">   false (unconverted): {target_dist[False]:,}")
        print(f">   true  (ordered):      {target_dist[True]:,}")
        print(f">   conversion rate:     {conversion_rate:.3%}")
    
    # cardinalities of categorical 
    
    print(f"\ncategorical variable cardinalities:")
    for col in categorical_features:
        if col in df.columns:
            n_unique = df[col].nunique()
            print(f">   {col}: {n_unique} unique values")
    
    print("=== validation completed ===\n")

def create_preprocessor(numerical_features, categorical_features):
    """
    return a preprocessor of numerical and categorical features
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'  # any column not specified is dropped
    )
    return preprocessor



# preprocessing pipeline
def create_preprocessing_pipeline(numerical_features, categorical_features):
    """
    return a complete preprocessing pipeline
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'  # any column not specified is dropped
    )
    return Pipeline([
        ('preprocessor', preprocessor)
    ])

# helper function to get feature names after preprocessing
def get_feature_names(preprocessor, numerical_features, categorical_features):
    """
    return feature names after preprocessing transformation
    """ 
    # categorical feature names (with one-hot encoding)
    cat_encoder = preprocessor.named_transformers_['cat']
    if hasattr(cat_encoder, 'get_feature_names_out'):
        # modern sklearn
        cat_features = cat_encoder.get_feature_names_out(categorical_features)
    else:
        # fallback for older sklearn versions
        cat_features = cat_encoder.get_feature_names(categorical_features)
    
    # numerical feature names are unchanged:
    return list(numerical_features) + list(cat_features)



def prepare_data(df, preprocessor, outcome, numerical_features, categorical_features, test_size=0.25, random_state=42):
    """
    prepare data with stratified train-test split and preprocessing
    
    parameters:
    -----------
    df : pandas.DataFrame
        with feature engineered dataset
    test_size : float, default=0.25
        proportion of total dedicated to testing 
    random_state : int, default=42
        random seed for reproducibility
    
    returns:
    --------
    X_train_processed, X_test_processed, y_train, y_test, preprocessor
    """
      
    # separate features and target
    y = df[outcome]
    X = df.drop([outcome], axis=1)  # features only
    
    print(f"dataset shape:       {X.shape}")
    print(f"target distribution: {y.value_counts().to_dict()}")
    print(f"conversion rate:     {y.mean():.3f}")
    
    # stratified train-test split for class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size, 
        stratify=y,
        random_state=random_state
    )
    
    print(f"\nafter split:")
    print(f"training set: {X_train.shape[0]} samples ({y_train.mean():.3f} conversion rate)")
    print(f"test set:     {X_test.shape[0]}  samples ({y_test.mean():.3f}  conversion rate)")
    
    # fit preprocessor on training data only
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed  = preprocessor.transform(X_test)
    
    print(f"\nafter preprocessing:")
    print(f"training features shape: {X_train_processed.shape}")
    print(f"test features shape:     {X_test_processed.shape}")
    
    # feature names for later use
    try:
        feature_names = get_feature_names(preprocessor, numerical_features, categorical_features)
        print(f"num features after preprocessing: {len(feature_names)}")
        
        # breakdown
        num_original = len(numerical_features)
        num_categorical = X_train_processed.shape[1] - num_original
        print(f"  - numerical features: {num_original}")
        print(f"  - categorical features (after one-hot): {num_categorical}")
        
    except Exception as e:
        print(f"error extracting feature names ({e})")
        feature_names = None
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names
