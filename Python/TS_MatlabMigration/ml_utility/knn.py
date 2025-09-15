from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from metrics import compute_metrics

def train_knn(X_train, y_train):
    """
    Trains a KNN model with hyperparameter tuning.
    
    Returns:
    - Best trained KNN model.
    """
    param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

def evaluate_knn(knn_model, X_test, y_test):
    """
    Evaluates the trained KNN model using Sensitivity & Specificity.
    
    Parameters:
    - knn_model: Trained KNN model.
    - X_test: Test features.
    - y_test: True test labels.
    """
    y_pred = knn_model.predict(X_test)
    sensitivity, specificity = compute_metrics(y_test, y_pred)
    
    print(f"Sensitivity (Recall for Positive Class): {sensitivity:.2f}")
    print(f"Specificity (Recall for Negative Class): {specificity:.2f}")