import numpy as np
from scipy import stats

def select_features(X, y=None, correlation_threshold=0.98, fdr_level=0.05):
    """
    Two-stage feature selection (FRESH filter).
    Stage 1: Unsupervised (Constant and highly correlated features).
    Stage 2: Supervised (P-value calculation + FDR control).
    """
    # Stage 1: Unsupervised
    # 1. Remove constant features
    std = np.nanstd(X, axis=0)
    constant_mask = std > 0
    X = X[:, constant_mask]
    
    if X.shape[1] == 0:
        return X, constant_mask

    # 2. Remove highly correlated features
    if X.shape[1] > 1:
        corr_matrix = np.abs(np.corrcoef(X, rowvar=False))
        to_drop = set()
        for i in range(corr_matrix.shape[1]):
            if i in to_drop:
                continue
            for j in range(i + 1, corr_matrix.shape[1]):
                if corr_matrix[i, j] > correlation_threshold:
                    to_drop.add(j)
        
        keep_indices = [i for i in range(X.shape[1]) if i not in to_drop]
        X = X[:, keep_indices]

    if y is None or X.shape[1] == 0:
        return X, constant_mask

    # Stage 2: Supervised (FRESH)
    p_values = []
    for i in range(X.shape[1]):
        feature = X[:, i]
        if len(np.unique(y)) == 2: # Classification
            group0 = feature[y == 0]
            group1 = feature[y == 1]
            if len(group0) > 1 and len(group1) > 1:
                _, p = stats.ttest_ind(group0, group1)
            else:
                p = 1.0
        else: # Regression
            _, p = stats.kendalltau(feature, y)
        
        if np.isnan(p):
            p = 1.0
        p_values.append(p)
    
    p_values = np.array(p_values)
    m = len(p_values)
    if m == 0:
        return X, constant_mask

    # Benjamini-Hochberg (FDR) control
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    significant = sorted_p <= (np.arange(1, m + 1) / m) * fdr_level
    if not any(significant):
        # If nothing is significant under FDR, we might want to return some top features 
        # but strictly following FRESH/FDR we should return nothing.
        # For a demo/test, let's at least return something if it's better than random?
        # No, let's be strict.
        return X[:, []], constant_mask
    
    max_idx = np.max(np.where(significant))
    selected_indices = sorted_indices[:max_idx + 1]
    
    return X[:, selected_indices], constant_mask
