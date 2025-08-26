import numpy as np
import pandas as pd
import re
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)


def map_postcodes_to_area_code(postcode:str) -> str:
    """ 
    given a british post code, returns the area code, if allowed. otherwise, replaces with 'other'.
    """
    area_codes_allowed=[
        'N',
        'L',
        'W',
        'B',
        'S',
        'M',
        'G',
        'E',
        'other'
    ]
    scan=re.findall(r'(.*?)\d', postcode)
    if len(scan)>0:
        area_code = scan[0]
    else: return 'other'
    if area_code in area_codes_allowed: return area_code
    else: return 'other'


def evaluate_model_performance(
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
        y_train_proba,
        y_test_proba,
        model_name="logistic-regression"):
    """
    model evaluation with focus on recall
    """
    print(f"\n=== {model_name.upper()} PERFORMANCE EVALUATION ===\n")
    
    # basics
    print("TRAINING SET PERFORMANCE:")
    print(f">   ROC-AUC:   {roc_auc_score(y_train, y_train_proba):.4f}")
    print(f">   PR-AUC:    {average_precision_score(y_train, y_train_proba):.4f}")
    print(f">   precision: {precision_score(y_train, y_train_pred):.4f}")
    print(f">   recall:    {recall_score(y_train, y_train_pred):.4f}")
    print(f">   f1-score:  {f1_score(y_train, y_train_pred):.4f}")
    
    print("\nTEST SET PERFORMANCE:")
    print(f">   ROC-AUC:   {roc_auc_score(y_test, y_test_proba):.4f}")
    print(f">   PR-AUC:    {average_precision_score(y_test, y_test_proba):.4f}")
    print(f">   precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f">   recall:    {recall_score(y_test, y_test_pred):.4f}")
    print(f">   f1-score:  {f1_score(y_test, y_test_pred):.4f}")
    
    # confusion matrix
    print("\nCONFUSION MATRIX (test set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"true  negatives: {tn}, false positives: {fp}")
    print(f"false negatives: {fn}, true  positives: {tp}")
    
    # business-facing metrics
    print("\nBUSINESS METRICS:")
    total_positives = y_test.sum()
    total_predictions = y_test_pred.sum()
    print(f">   total actual conversions:    {total_positives}")
    print(f">   total predicted conversions: {total_predictions}")
    print(f">   conversions captured:        {tp} out of {total_positives} ({tp/total_positives:.1%})")
    print(f">   precision in predictions:    {tp} out of {total_predictions} ({tp/total_predictions:.1%})")
    
    return {
        'roc_auc'  : roc_auc_score(y_test, y_test_proba),
        'pr_auc'   : average_precision_score(y_test, y_test_proba),
        'precision': precision_score(y_test, y_test_pred),
        'recall'   : recall_score(y_test, y_test_pred),
        'f1'       : f1_score(y_test, y_test_pred)
    }


def find_optimal_threshold(y_true, y_proba, metric='recall', target_value=0.85):
    """
    find optimal threshold for classification based on target metric
    """
    print(f"\n=== THRESHOLD OPTIMIZATION ===\n")
    
    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    if metric == 'recall':
        # compute threshold that achieves target recall
        target_idx = np.argmax(recall >= target_value)
        if target_idx > 0:
            optimal_threshold = thresholds[target_idx]
            optimal_precision = precision[target_idx]
            optimal_recall = recall[target_idx]
        else:
            # cannot satisfy recall requirement. switch over to maximizing f1-score
            print(f"failed to achieve target recall of {target_value}, switching to f1-score")
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_precision = precision[optimal_idx]
            optimal_recall = recall[optimal_idx]
    
    print(f"optimal threshold: {optimal_threshold:.4f}")
    print(f">   precision at this threshold: {optimal_precision:.4f}")
    print(f">   recall at this threshold: {optimal_recall:.4f}")
    
    # Show what happens at different recall levels
    print(f"\nPRECISION AT DIFFERENT RECALL LEVELS:")
    for target_recall in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        if len(recall[recall >= target_recall]) > 0:
            idx = np.argmax(recall >= target_recall)
            precision_at_recall = precision[idx]
            threshold_at_recall = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
            print(f">   recall {target_recall:.0%}: precision = {precision_at_recall:.3f}, threshold = {threshold_at_recall:.4f}")
    
    return optimal_threshold


def compare_models(lr_metrics, rf_metrics, lr_proba, rf_proba, y_test):
    """
    compare  performance of logistic-regression and random-forest models
    """
    print("\n=== MODEL COMPARISON ===\n")
    
    comparison_df = pd.DataFrame({
        'metric': ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1-score'],
        'logistic-regression': [
            lr_metrics['roc_auc'], lr_metrics['pr_auc'], 
            lr_metrics['precision'], lr_metrics['recall'], lr_metrics['f1']
        ],
        'random-forest': [
            rf_metrics['roc_auc'], rf_metrics['pr_auc'],
            rf_metrics['precision'], rf_metrics['recall'], rf_metrics['f1']
        ]
    })
    
    # improvements
    comparison_df['random-forest improvement'] = (
        comparison_df['random-forest'] - comparison_df['logistic-regression']
    ).round(4)
    
    print(comparison_df.to_string(index=False))
    
    print(f"\nPROBABILITY DISTRIBUTION COMPARISON:")
    print(f"logistic-regression probabilities:")
    print(f">   min    : {lr_proba.min():.4f}, ")
    print(f">   mean   : {lr_proba.mean():.4f}")
    print(f">   median : {np.median(lr_proba):.4f}")
    print(f">   Max    : {lr_proba.max():.4f}")
    print(f">   std    : {lr_proba.std():.4f}")

    print(f"\nrandom-forest probabilities:")
    print(f">   min    : {rf_proba.min():.4f}, ")
    print(f">   mean   : {rf_proba.mean():.4f}")
    print(f">   median : {np.median(rf_proba):.4f}")
    print(f">   Max    : {rf_proba.max():.4f}")
    print(f">   std    : {rf_proba.std():.4f}")
   
    return comparison_df


def analyze_probability_ranges(y_test, y_proba, model_name="Model", n_deciles=10):
    """
    analyze conversion rates across probability deciles -  analysis
    """
    print(f"\n=== {model_name.upper()} DECILE ANALYSIS ===\n")
    
    # deciles based on predicted probabilities
    df_analysis = pd.DataFrame({
        'actual': y_test,
        'probability': y_proba
    })
    
    # decile bins
    df_analysis['decile'] = pd.qcut(df_analysis['probability'], q=n_deciles, labels=False, duplicates='drop') + 1
    
    # calculate metrics by decile
    decile_stats = (
        df_analysis
        .groupby('decile')
        .agg(
            {
                'actual'     : ['count', 'sum', 'mean'],
                'probability': ['min', 'max', 'mean']
            }
        )
        .round(4)
    )
    
    decile_stats.columns = [
        'num_leads',
        'conversions',
        'conversion_rate',
        'min_prob',
        'max_prob',
        'avg_prob'
    ]
    
    # lift over random
    baseline_rate = y_test.mean()
    decile_stats['lift'] = (decile_stats['conversion_rate'] / baseline_rate).round(2)
    
    # calculate cumulative metrics for targeting
    decile_stats = decile_stats.sort_values('decile', ascending=False)  # start from highest decile
    decile_stats['cumulative_leads']           = decile_stats['num_leads'].cumsum()
    decile_stats['cumulative_conversions']     = decile_stats['conversions'].cumsum()
    decile_stats['cumulative_conversion_rate'] = (
        decile_stats['cumulative_conversions'] / decile_stats['cumulative_leads']
    ).round(4)
    
    print("DECILE PERFORMANCE (highest probability decile first):")
    print(decile_stats.to_string())
    
    # business insights
    top_30_pct_leads = int(0.3 * len(y_test))
    top_30_pct_conversions = decile_stats.head(3)['conversions'].sum()
    total_conversions = y_test.sum()
    
    print(f"\nBUSINESS INSIGHTS:")
    print(f"  top decile lift: {decile_stats.iloc[0]['lift']:.1f}x better than random")
    print(f"  top 30% of leads capture: {top_30_pct_conversions}/{total_conversions} conversions ({top_30_pct_conversions/total_conversions:.1%})")
    print(f"  top decile conversion rate: {decile_stats.iloc[0]['conversion_rate']:.1%}")
    
    return decile_stats


def compare_all_models(lr_metrics, rf_metrics, xgb_metrics, lr_proba, rf_proba, xgb_proba, y_test):
    """
    comprehensive comparison of all three models
    """
    print("\n" + "="*60)
    print("=== COMPREHENSIVE MODEL COMPARISON ===")
    print("="*60 + "\n")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Metric': ['ROC-AUC', 'PR-AUC', 'Precision', 'Recall', 'F1-Score'],
        'Logistic Regression': [
            lr_metrics['roc_auc'], lr_metrics['pr_auc'], 
            lr_metrics['precision'], lr_metrics['recall'], lr_metrics['f1']
        ],
        'Random Forest': [
            rf_metrics['roc_auc'], rf_metrics['pr_auc'],
            rf_metrics['precision'], rf_metrics['recall'], rf_metrics['f1']
        ],
        'XGBoost': [
            xgb_metrics['roc_auc'], xgb_metrics['pr_auc'],
            xgb_metrics['precision'], xgb_metrics['recall'], xgb_metrics['f1']
        ]
    })
    
    # Calculate improvements over baseline (Logistic Regression)
    comparison_df['RF vs LR'] = (comparison_df['Random Forest'] - comparison_df['Logistic Regression']).round(4)
    comparison_df['XGB vs LR'] = (comparison_df['XGBoost'] - comparison_df['Logistic Regression']).round(4)
    comparison_df['XGB vs RF'] = (comparison_df['XGBoost'] - comparison_df['Random Forest']).round(4)
    
    print("PERFORMANCE COMPARISON:")
    print(comparison_df.to_string(index=False))
    
    # Identify best model for each metric
    print(f"\nBEST PERFORMERS:")
    for metric in ['ROC-AUC', 'PR-AUC', 'Precision', 'Recall', 'F1-Score']:
        row = comparison_df[comparison_df['Metric'] == metric].iloc[0]
        best_model = max(['Logistic Regression', 'Random Forest', 'XGBoost'], 
                        key=lambda x: row[x])
        best_score = row[best_model]
        print(f"  {metric}: {best_model} ({best_score:.4f})")
    
    # Probability distribution comparison
    print(f"\nPROBABILITY DISTRIBUTION COMPARISON:")
    models = {'Logistic Regression': lr_proba, 'Random Forest': rf_proba, 'XGBoost': xgb_proba}
    
    for name, proba in models.items():
        print(f"\n{name}:")
        print(f"  Range: [{proba.min():.4f}, {proba.max():.4f}]")
        print(f"  Mean: {proba.mean():.4f}, Std: {proba.std():.4f}")
        print(f"  Quartiles: {np.percentile(proba, [25, 50, 75])}")
    
    # Business impact comparison
    print(f"\nBUSINESS IMPACT (Top 30% of leads):")
    for name, proba in models.items():
        # Get top 30% by probability
        top_30_pct_threshold = np.percentile(proba, 70)
        top_30_pct_mask = proba >= top_30_pct_threshold
        
        total_conversions = y_test.sum()
        captured_conversions = y_test[top_30_pct_mask].sum()
        capture_rate = captured_conversions / total_conversions
        
        print(f"  {name}: {captured_conversions}/{total_conversions} conversions ({capture_rate:.1%})")
    
    return comparison_df

