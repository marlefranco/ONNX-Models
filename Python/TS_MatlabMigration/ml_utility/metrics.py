from sklearn.metrics import confusion_matrix

def compute_metrics(y_true, y_pred):
    """
    Computes Sensitivity (Recall for Positive Class) and Specificity.
    
    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    
    Returns:
    - Sensitivity (recall for positive class)
    - Specificity (recall for negative class)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn)  # True Positive Rate (Recall for positive class)
    specificity = tn / (tn + fp)  # True Negative Rate

    return sensitivity, specificity