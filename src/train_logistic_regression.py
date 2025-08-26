import numpy as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler, OneHotEncoder, LabelEncoder
from utils                   import evaluate_model_performance, find_optimal_threshold


def train_logistic_regression(X_train, X_test, y_train, y_test, feature_names=None):
    """
    logistic regression model and performance evaluation
    """
    print("=== LOGISTIC REGRESSION BASELINE ===\n")
    
    # train model with balanced class weights 
    lr_model = LogisticRegression(
        class_weight='balanced',  # automatically adjusts for class imbalance
        random_state=42,
        max_iter=2000
    )
    
    # fit the model
    lr_model.fit(X_train, y_train)
    
    # predictions:
    y_train_pred = lr_model.predict(X_train)
    y_test_pred  = lr_model.predict(X_test)
    
    # probability scores (propensity to buy)
    y_train_proba = lr_model.predict_proba(X_train)[:, 1]  # probability of class 1 (conversion)
    y_test_proba  = lr_model.predict_proba(X_test )[:, 1]
    
    print("training complete")
    print(f"training samples: {X_train.shape[0]}")
    print(f"test samples:     {X_test.shape[0]}")
    print(f"features:         {X_train.shape[1]}")
    
    return lr_model, y_train_pred, y_test_pred, y_train_proba, y_test_proba


def analyze_feature_importance(model, feature_names=None, top_n=15):
    """
    present feature importance from logistic regression coefficients
    """
    print(f"\n=== FEATURE IMPORTANCE ===\n")
    
    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
        
        if feature_names is not None and len(feature_names) == len(coefficients):
            # prepare feature importance dataframe
            feature_importance = (
                pd.DataFrame(
                {
                    'feature'         : feature_names,
                    'coefficient'     : coefficients,
                    'abs_coefficient' : np.abs(coefficients)
                })
                .sort_values('abs_coefficient', ascending=False)
            )
            
            print(f"TOP {top_n} MOST IMPORTANT FEATURES:")
            print(feature_importance[['feature', 'coefficient']].head(top_n).to_string(index=False))
            
            # separate positive and negative influences
            positive_features = feature_importance[feature_importance['coefficient'] > 0].head(8)
            negative_features = feature_importance[feature_importance['coefficient'] < 0].head(8)
            
            print(f"\nTOP POSITIVE INFLUENCES (boost purchase probability):")
            print(positive_features[['feature', 'coefficient']].to_string(index=False))
            
            print(f"\nTOP NEGATIVE INFLUENCES (reduce purchase probability):")
            print(negative_features[['feature', 'coefficient']].to_string(index=False))
            
            return feature_importance
        else:
            print("check the feature names! either they're not available or they mismatch with coefficients")
            print(f"num coefficients: {coefficients.shape}")
            if feature_names:
                print(f"num feature names: {len(feature_names)}")
            return None
    else:
        print("forgot to train model? model does not have feature coefficients")
        return None


# combined execution function
def run_logistic_regression_analysis(X_train, X_test, y_train, y_test, feature_names=None):
    """
    logistic regression analysis pipeline
    """
    # model training
    lr_model, y_train_pred, y_test_pred, y_train_proba, y_test_proba = train_logistic_regression(
        X_train, X_test, y_train, y_test, feature_names
    )
    
    # performance evalution
    metrics = evaluate_model_performance(
        y_train, y_test, 
        y_train_pred, y_test_pred,
        y_train_proba, y_test_proba, 
        "logistic-regression"
    )
    
    # feature importance
    feature_importance = analyze_feature_importance(lr_model, feature_names)
    
    # optimal thresholds, target high recall
    optimal_threshold = find_optimal_threshold(y_test, y_test_proba, metric='recall', target_value=0.85)
    
    return lr_model, metrics, feature_importance, optimal_threshold, y_test_proba
