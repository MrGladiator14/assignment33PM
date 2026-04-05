# Q1: Dataset with 100 features and 50 samples - Algorithm Selection

**Best Algorithms:**

- **SVM with RBF kernel**: Effective in high-dimensional spaces, handles feature interactions well
- **KNN with feature selection/dimensionality reduction**: Works well when distance metrics become meaningful after reducing dimensionality
- **Regularized models (Lasso/Ridge)**: Prevent overfitting by penalizing complex models in high-dimensional space

**Why these work**: The high feature-to-sample ratio (100:50) creates a "curse of dimensionality" problem. These algorithms either handle high dimensions well (SVM) or include regularization mechanisms to prevent overfitting.

**Algorithms that would fail:**

- **Standard KNN without dimensionality reduction**: Distance metrics become meaningless in high dimensions
- **Decision trees without pruning**: Will severely overfit due to unlimited feature splitting
- **Naive Bayes**: Independence assumption violated with many correlated features

## Q2: Model Selection Report Function

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel
from sklearn.base import clone

def model_selection_report(X, y, models_dict, cv=5, test_size=0.2, random_state=42):
    """
    Generate comprehensive model selection report with statistical comparison.
  
    Parameters:
    X: Feature matrix
    y: Target vector
    models_dict: Dictionary of model names to model objects
    cv: Number of cross-validation folds
    test_size: Test set proportion
    random_state: Random seed
  
    Returns:
    DataFrame with model performance metrics and statistical comparison
    """
  
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
  
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
  
    results = []
    cv_scores_dict = {}
  
    for name, model in models_dict.items():
        model_clone = clone(model)
  
        cv_scores = cross_val_score(model_clone, X_train_scaled, y_train, cv=cv)
        cv_scores_dict[name] = cv_scores
  
        model_clone.fit(X_train_scaled, y_train)
        test_score = model_clone.score(X_test_scaled, y_test)
  
        results.append({
            'Model': name,
            'CV_Mean': np.mean(cv_scores),
            'CV_Std': np.std(cv_scores),
            'CV_Min': np.min(cv_scores),
            'CV_Max': np.max(cv_scores),
            'Test_Score': test_score
        })
  
    df = pd.DataFrame(results)
  
    model_names = list(models_dict.keys())
    best_model = None
    best_p_value = 1.0
  
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i != j:
                scores1 = cv_scores_dict[model1]
                scores2 = cv_scores_dict[model2]
  
                # Paired t-test
                t_stat, p_value = ttest_rel(scores1, scores2)
  
                # Check if model1 is significantly better than model2
                if p_value < 0.05 and np.mean(scores1) > np.mean(scores2):
                    if p_value < best_p_value:
                        best_p_value = p_value
                        best_model = model1
  
    df['Statistically_Best'] = df['Model'] == best_model if best_model else False
  
    df = df.sort_values('CV_Mean', ascending=False).reset_index(drop=True)
  
    return df, best_model, best_p_value if best_model else None
```

## Q3: SVM Analysis - Train Accuracy 1.0, Test Accuracy 0.52

**Problem Identification**: This is a classic case of severe overfitting. The model has memorized the training data perfectly but fails to generalize.

**Root Causes:**

1. **RBF kernel complexity**: The RBF kernel with default parameters can create highly complex decision boundaries
2. **Insufficient regularization**: C parameter likely too high, allowing the model to fit noise
3. **Gamma parameter**: May be too large, creating overly narrow influence regions

**Three Specific Fixes:**

1. **Adjust Regularization Parameters**:

   - Reduce C parameter (e.g., from default 1.0 to 0.1 or 0.01)
   - Increase gamma parameter to reduce model complexity
   - Use grid search: `C=[0.01, 0.1, 1, 10]`, `gamma=['scale', 'auto', 0.001, 0.01]`
2. **Feature Engineering/Selection**:

   - Apply PCA to reduce dimensionality and noise
   - Use feature selection (e.g., SelectKBest with f_classif)
   - Remove correlated features that may cause multicollinearity
3. **Cross-Validation and Ensemble Methods**:

   - Implement stratified k-fold cross-validation for robust evaluation
   - Use ensemble methods like Bagging with SVM as base estimator
   - Consider simpler models as baseline (Logistic Regression, Linear SVM)

**Additional Recommendations**:

- Check for data leakage between train/test sets
- Examine class distribution - consider balanced class weights
- Validate data preprocessing pipeline for consistency
