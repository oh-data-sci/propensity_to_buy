import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
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
from utils import compare_all_models, evaluate_model_performance, find_optimal_threshold, analyze_probability_ranges
import xgboost as xgb


def train_xgboost_model(X_train, X_test, y_train, y_test, feature_names=None):
    """
    xgboost model optimized for propensity scoring and recall
    """
    print("=== XGBOOST MODEL ===\n")
    
    # scale_pos_weight for class imbalance (87% negative / 13% positive ≈ 6.7)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"class imbalance ratio: {scale_pos_weight:.2f}")
    
    # xgboost with optimized parameters for conversion prediction
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,                  # more trees, better predictions, 
        max_depth=6,                       # moderate depth to prevent overfitting
        learning_rate=0.1,                 # standard
        subsample=0.8,                     # row sampling to prevent overfitting
        colsample_bytree=0.8,              # feature sampling
        scale_pos_weight=scale_pos_weight, # class imbalance
        reg_alpha=0.1,                     # L1 regularization
        reg_lambda=0.1,                    # L2 regularization
        random_state=42,
        eval_metric='auc',          
        early_stopping_rounds=10,         # reduces risk of overfitting
        n_jobs=-1,                        # all cores
        verbosity=0                       # reduce noise
    )
    
    print("training xgboost with early stopping ")
    
    # fit with validation set for early stopping
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    print(f"training completed at iteration {xgb_model.get_booster().best_iteration + 1}")
    
    # predict classes
    y_train_pred = xgb_model.predict(X_train)
    y_test_pred  = xgb_model.predict(X_test)
    
    # generate probability scores (propensity to buy)
    y_train_proba = xgb_model.predict_proba(X_train)[:, 1]
    y_test_proba  = xgb_model.predict_proba(X_test)[:, 1]
    
    print(f"model training complete")
    print(f"best iteration: {xgb_model.get_booster().best_iteration + 1}")
    
    return xgb_model, y_train_pred, y_test_pred, y_train_proba, y_test_proba


def analyze_xgb_feature_importance(model, feature_names=None, top_n=20):
    """
    xgboost feature importance analysis
    """
    print(f"\n=== XGBOOST FEATURE IMPORTANCE ANALYSIS ===\n")
    
    if feature_names is None:
        print("feature names missing, using indices")
        feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]
    
    # get different importance types
    importance_types = {
        'weight'     : 'number of times feature is used to split',
        'gain'       : 'average gain when feature is used',
        'cover'      : 'average coverage when feature is used',
        'total_gain' : 'total gain when feature is used',
        'total_cover': 'total coverage when feature is used'
    }
    
    importance_dfs = {}
    
    for imp_type, description in importance_types.items():
        try:
            importances = model.get_booster().get_score(importance_type=imp_type)
            
            # preapre DataFrame
            df = (
                pd.DataFrame(
                    [
                        {
                            'feature': feature_names[int(k.replace('f', ''))], 
                            'importance': v
                        }
                     for k, v in importances.items()
                    ]
                )
                .sort_values('importance', ascending=False)
            )
            
            importance_dfs[imp_type] = df
            
            print(f"\nTOP {min(top_n, len(df))} FEATURES BY {imp_type.upper()} ({description}):")
            print(df.head(top_n)[['feature', 'importance']].to_string(index=False))
            
        except Exception as e:
            print(f"could not get {imp_type} importance: {e}")
    
    # focus on 'gain' as the most interpretable
    if 'gain' in importance_dfs:
        main_importance = importance_dfs['gain']
        
        # cumulative importance
        main_importance['cumulative_importance'] = (
            main_importance['importance'] / main_importance['importance'].sum()
        ).cumsum()
        
        # top features contributing 80% of importance
        top_80_pct = main_importance[main_importance['cumulative_importance'] <= 0.80]
        print(f"\nTOP FEATURES CONTRIBUTING 80% OF TOTAL GAIN ({len(top_80_pct)} features):")
        print(top_80_pct[['feature', 'importance', 'cumulative_importance']].to_string(index=False))
        
        return main_importance, importance_dfs
    
    return None, importance_dfs

def plot_xgb_learning_curves(model):
    """
    plot xgboost learning curves to check for overfitting
    """
    print(f"\n=== XGBOOST LEARNING CURVES ===\n")
    
    try:
        # evaluation results
        eval_results = model.evals_result()
        
        if eval_results:
            train_auc = eval_results['train']['auc']
            valid_auc = eval_results['validation']['auc']
            
            plt.figure(figsize=(18, 6))
            plt.plot(train_auc, label='train AUC', marker='o', markersize=3)
            plt.plot(valid_auc, label='validation AUC', marker='s', markersize=3)
            plt.xlabel('iteration')
            plt.ylabel('AUC')
            plt.title('xgboost learning curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # mark best iteration
            best_iter = model.get_booster().best_iteration
            plt.axvline(x=best_iter, color='red', linestyle='--', 
                       label=f'best iteration: {best_iter + 1}')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
            # Report final scores
            print(f"final training   AUC: {train_auc[-1]:.4f}")
            print(f"final validation AUC: {valid_auc[-1]:.4f}")
            print(f"best  validation AUC: {max(valid_auc):.4f} at iteration {np.argmax(valid_auc) + 1}")
            
            # overfitting?
            final_gap = train_auc[-1] - valid_auc[-1]
            print(f"train-validation AUC gap: {final_gap:.4f}")
            if final_gap > 0.05:
                print("⚠️ overfitting detected (gap > 0.05)")
            else:
                print("✅ no significant overfitting detected")
    
    except Exception as e:
        print(f"could not plot learning curves: {e}")

def run_xgboost_analysis(X_train, X_test, y_train, y_test, feature_names=None,
                        lr_metrics=None, rf_metrics=None, lr_proba=None, rf_proba=None):
    """
    Complete XGBoost analysis pipeline
    """
    # Train XGBoost
    xgb_model, y_train_pred, y_test_pred, y_train_proba, y_test_proba = train_xgboost_model(
        X_train, X_test, y_train, y_test, feature_names
    )
    
    # Evaluate performance
    xgb_metrics = evaluate_model_performance(
        y_train, y_test, y_train_pred, y_test_pred, 
        y_train_proba, y_test_proba, "XGBoost"
    )
    
    # Analyze feature importance
    xgb_feature_importance, all_importances = analyze_xgb_feature_importance(xgb_model, feature_names)
    
    # Plot learning curves
    plot_xgb_learning_curves(xgb_model)
    
    # Find optimal threshold
    xgb_threshold = find_optimal_threshold(y_test, y_test_proba, metric='recall', target_value=0.85)
    
    # Business-focused decile analysis
    xgb_deciles = analyze_probability_ranges(y_test, y_test_proba, "XGBoost")
    
    # Compare with previous models if provided
    if lr_metrics is not None and rf_metrics is not None:
        model_comparison = compare_all_models(
            lr_metrics, rf_metrics, xgb_metrics, 
            lr_proba, rf_proba, y_test_proba, y_test
        )
    else:
        model_comparison = None
    
    return (xgb_model, xgb_metrics, xgb_feature_importance, xgb_threshold, 
            y_test_proba, xgb_deciles, model_comparison)
