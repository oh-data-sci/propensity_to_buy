
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
from sklearn.ensemble   import RandomForestClassifier
from utils              import evaluate_model_performance, find_optimal_threshold, compare_models, analyze_probability_ranges


def train_random_forest(X_train, X_test, y_train, y_test, feature_names=None):
    """
    random forest model optimized for recall and probability output
    """
    print("=== RANDOM FOREST MODEL ===\n")
    
    # random forest with hyperparameters optimized for imbalanced data
    rf_model = RandomForestClassifier(
        n_estimators=300,           # balance of performance vs speed
        max_depth=8,                # prevent overfitting
        min_samples_split=10,       # require meaningful splits
        min_samples_leaf=5,         # prevent tiny leaf nodes
        class_weight='balanced',    # handle class imbalance
        random_state=42,
        n_jobs=-1                   # use all cores
    )
    
    print("training random forest...")
    rf_model.fit(X_train, y_train)
    
    # predict:
    y_train_pred = rf_model.predict(X_train)
    y_test_pred  = rf_model.predict(X_test)
    
    # probability scores (propensity to buy) 
    y_train_proba = rf_model.predict_proba(X_train)[:, 1]
    y_test_proba  = rf_model.predict_proba(X_test)[:, 1]
    
    print("random forest model ready")
    print(f"out-of-bag score: {rf_model.oob_score_:.4f}" if hasattr(rf_model, 'oob_score_') else "out-of-bag score not available")
    
    return rf_model, y_train_pred, y_test_pred, y_train_proba, y_test_proba


def analyze_rf_feature_importance(model, feature_names=None, top_n=15):
    """
    random forest feature importance 
    """
    print(f"\n=== RANDOM FOREST FEATURE IMPORTANCE ===\n")
    
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        # prepare a feature importance dataframe
        feature_importance = (
            pd.DataFrame(
                {
                    'feature'   : feature_names,
                    'importance': model.feature_importances_
                }
            )
            .sort_values('importance', ascending=False)
        )
        
        print(f"TOP {top_n} IMPORTANT FEATURES:")
        print(feature_importance.head(top_n).to_string(index=False))
        
        # cumulative importance
        feature_importance['cumulative_importance'] = feature_importance['importance'].cumsum()
        
        # the top features that make up 80% of importance
        top_80_pct = feature_importance[feature_importance['cumulative_importance'] <= 0.80]
        print(f"\nTOP FEATURES CONTRIBUTING 80% OF IMPORTANCE ({len(top_80_pct)} FEATURES):")
        print(top_80_pct[['feature', 'importance', 'cumulative_importance']].to_string(index=False))
        
        return feature_importance
    else:
        print("feature importance not available")
        return None


def run_random_forest_analysis(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names=None, 
        lr_metrics=None,
        lr_proba=None):
    """
    random-forest analysis pipeline
    """
    # train Random Forest
    rf_model, y_train_pred, y_test_pred, y_train_proba, y_test_proba = train_random_forest(
        X_train, X_test, y_train, y_test, feature_names
    )
    
    # evaluate performance using same function as logistic regression
    rf_metrics = evaluate_model_performance(
        y_train,
        y_test,
        y_train_pred,
        y_test_pred, 
        y_train_proba,
        y_test_proba,
        "random-forest"
    )
    
    #  feature importance (different method than logistic regression)
    rf_feature_importance = analyze_rf_feature_importance(rf_model, feature_names)
    
    # optimal threshold for Random Forest
    rf_threshold = find_optimal_threshold(y_test, y_test_proba, metric='recall', target_value=0.85)
    
    # compare logistic regression, if provided
    if lr_metrics is not None and lr_proba is not None:
        comparison = compare_models(lr_metrics, rf_metrics, lr_proba, y_test_proba, y_test)
    
    # business-focused decile analysis
    decile_analysis = analyze_probability_ranges(y_test, y_test_proba, "Random Forest")
    
    return rf_model, rf_metrics, rf_feature_importance, rf_threshold, y_test_proba, decile_analysis