import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import train

def evaluate_existing_model():
    
    print("\n" + "="*70)
    print("EVALUATING EXISTING TRAINED MODEL")
    print("="*70 + "\n")
    
    print("Checking for trained model files...")
    model_path = 'data/processed/crime_model.pkl'
    scaler_path = 'data/processed/scaler.pkl'
    
    if not os.path.exists(model_path):
        print(f"   Model not found: {model_path}")
        print("   Run the system first to train the model!")
        return None
    
    if not os.path.exists(scaler_path):
        print(f"   Scaler not found: {scaler_path}")
        return None
    
    print(f"   Model found: {model_path}")
    print(f"   Scaler found: {scaler_path}\n")
    
    print("Loading trained model and scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("   Model and scaler loaded\n")
    
    print("Loading crime data...")
    csv_path = 'data/preprocessed.csv'
    
    if not os.path.exists(csv_path):
        print(f"   Crime data not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=['nm_pol', 'lat', 'long'], keep='first')
    print(f"   Loaded {len(df)} crime records\n")
    
    print("Preprocessing data...")
    
    df['crime_severity'] = (
        df['murder'] * 0.20 + 
        df['rape'] * 0.18 + 
        df['gangrape'] * 0.18 + 
        df['robbery'] * 0.04 + 
        df['theft'] * 0.04 + 
        df['assualt murders'] * 0.19 + 
        df['sexual harassement'] * 0.16
    )
    
    df['crime_density'] = df['totalcrime'] / df['totarea']
    
    df['crime_risk_score'] = (
        (df['crime_density'] - df['crime_density'].min()) /
        (df['crime_density'].max() - df['crime_density'].min())
    ) * 100
    
    print("   Features engineered\n")
    
    print("Preparing features...")
    feature_cols = [
        'murder', 'rape', 'gangrape', 'robbery', 'theft', 
        'assualt murders', 'sexual harassement', 'totalcrime', 
        'totarea', 'crime_density', 'crime_severity'
    ]
    
    X = df[feature_cols].values
    y = df['crime_risk_score'].values
    
    print(f"   Features: {X.shape}")
    print(f"   Target: {y.shape}\n")
    
    print("6. Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Train: {len(y_train)} samples")
    print(f"   Test: {len(y_test)} samples\n")
    
    print("Scaling features with loaded scaler...")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   Features scaled\n")
    
    print("Making predictions...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    print("   Predictions complete\n")
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70 + "\n")
    
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mape = np.mean(np.abs((y_train - y_train_pred) / (y_train + 1e-10))) * 100
    train_max_error = np.max(np.abs(y_train - y_train_pred))
    print("TRAINING SET METRICS:")
    print(f"  R² Score:  {train_r2:.4f}")
    print(f"  MAE:       {train_mae:.4f}")
    print(f"  RMSE:      {train_rmse:.4f}")
    print(f"  MAPE:      {train_mape:.2f}%")
    print(f"  Max Error: {train_max_error:.2f}\n")
    
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print("TEST SET METRICS:")
    print(f"  R² Score:  {test_r2:.4f}")
    print(f"  MAE:       {test_mae:.4f}")
    print(f"  RMSE:      {test_rmse:.4f}\n")
    
    test_mape = np.mean(np.abs((y_test - y_test_pred) / (y_test + 1e-10))) * 100
    max_error = np.max(np.abs(y_test - y_test_pred))
    print(f"  MAPE:      {test_mape:.2f}%")
    print(f"  Max Error: {max_error:.2f}\n")

    
    
    print("="*70)
    print("FEATURE IMPORTANCE")
    print("="*70 + "\n")
    
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance.to_string(index=False))
    print()
    
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Crime Risk Model Evaluation', fontsize=16, fontweight='bold')
    
    axes[0, 0].scatter(y_test, y_test_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 
                    'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Risk Score', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Predicted Risk Score', fontsize=12, fontweight='bold')
    axes[0, 0].set_title(f'Actual vs Predicted\nR² = {test_r2:.4f}, MAE = {test_mae:.2f}', 
                         fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    residuals = y_test - y_test_pred
    axes[0, 1].scatter(y_test_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Risk Score', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title(f'Residual Plot\nRMSE = {test_rmse:.2f}', 
                         fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
    axes[1, 0].barh(feature_importance['Feature'], 
                    feature_importance['Importance'],
                    color=colors,
                    edgecolor='black',
                    linewidth=0.5)
    axes[1, 0].set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Feature Importance', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    axes[1, 1].hist(residuals, bins=25, edgecolor='black', alpha=0.7, color='skyblue')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    axes[1, 1].set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1, 1].set_title(f'Error Distribution\nMean Error = {np.mean(residuals):.2f}', 
                         fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
    print("   Saved: model_evaluation_results.png")
    
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70 + "\n")
    
    with open('evaluation_metrics.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("CRIME RISK MODEL - EVALUATION METRICS\n")
        f.write("="*70 + "\n\n")
        
        f.write("USE THESE METRICS IN YOUR RESEARCH PAPER:\n\n")
        
        f.write("TEST SET PERFORMANCE:\n")
        f.write(f"  R² Score (R-squared):           {test_r2:.4f}\n")
        f.write(f"  MAE (Mean Absolute Error):      {test_mae:.2f}\n")
        f.write(f"  RMSE (Root Mean Squared Error): {test_rmse:.2f}\n")
        f.write(f"  MAPE (Mean Abs % Error):        {test_mape:.2f}%\n")
        f.write(f"  Maximum Error:                  {max_error:.2f}\n\n")
        
        f.write("TRAINING SET PERFORMANCE:\n")
        f.write(f"  R² Score:  {train_r2:.4f}\n")
        f.write(f"  MAE:       {train_mae:.2f}\n")
        f.write(f"  RMSE:      {train_rmse:.2f}\n\n")
        f.write(f"  MAPE:      {train_mape:.2f}%\n")
        f.write(f"  Maximum Error: {train_max_error:.2f}\n\n")
        
        f.write("MODEL CONFIGURATION:\n")
        f.write(f"  Algorithm:       Random Forest Regressor\n")
        f.write(f"  n_estimators:    {model.n_estimators}\n")
        f.write(f"  max_depth:       {model.max_depth}\n")
        f.write(f"  Training samples: {len(y_train)}\n")
        f.write(f"  Test samples:     {len(y_test)}\n\n")
        
        f.write("="*70 + "\n")
        f.write("FEATURE IMPORTANCE:\n")
        f.write("="*70 + "\n")
        f.write(feature_importance.to_string(index=False))
        f.write("\n")
    
    print("   Saved: evaluation_metrics.txt")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70 + "\n")
    
    print("FILES GENERATED:")
    print("  1. model_evaluation_results.png  - Visualization charts")
    print("  2. evaluation_metrics.txt        - Complete metrics report")
    print("  3. latex_metrics_table.tex       - LaTeX table for paper\n")
    
    print("INTERPRETATION:")
    print(f"  • R² = {test_r2:.4f} means model explains {test_r2*100:.2f}% of variance")
    print(f"  • MAE = {test_mae:.2f} means average error is {test_mae:.2f} points (on 0-100 scale)")
    print(f"  • RMSE = {test_rmse:.2f} penalizes large errors more than MAE")
    
    if test_r2 > 0.85:
        print(f"  • R² > 0.85: EXCELLENT model performance! ")
    elif test_r2 > 0.70:
        print(f"  • R² > 0.70: GOOD model performance ")
    else:
        print(f"  • R² < 0.70: Model might need improvement")
    
    print()
    
    return {
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_mape': test_mape,
        'train_r2': train_r2,
        'feature_importance': feature_importance
    }


if __name__ == '__main__':
    print("\n" + "="*70)
    print("CRIME RISK MODEL EVALUATION TOOL")
    print("="*70)
    print("Make sure you have:")
    print("  1. Trained model: data/processed/crime_model.pkl")
    print("  2. Scaler: data/processed/scaler.pkl")
    print("  3. Crime data: data/preprocessed.csv")
    
    input("\nPress Enter to continue...")
    
    try:
        metrics = evaluate_existing_model()
        
        if metrics:
            print("\n" + "="*70)
            print("QUICK SUMMARY FOR YOUR PAPER:")
            print("="*70)
            print(f"\nR² Score:  {metrics['test_r2']:.4f}")
            print(f"MAE:       {metrics['test_mae']:.2f}")
            print(f"RMSE:      {metrics['test_rmse']:.2f}")
            print(f"MAPE:      {metrics['test_mape']:.2f}%")
        
    except FileNotFoundError as e:
        print(f"\nFile not found: {e}")
        print("\nMake sure you're running this from the 'backend' directory")
        print("and that you've trained the model first.\n")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()