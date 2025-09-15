from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from ml_utility.metrics import compute_metrics
from sklearn.preprocessing import LabelEncoder

def train_models(X_train, y_train):
    """
    Trains multiple ML models (KNN, Decision Tree, SVM, XGBoost, Random Forest) 
    with hyperparameter tuning and selects the best model while checking for overfitting.

    Returns:
    - Best trained model (avoiding overfitting).
    - Cross-validation results.
    - Best model name.
    - Best cross-validation accuracy.
    """
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    # Print to verify label encoder is fitted
    print("Classes in LabelEncoder (Train):", label_encoder.classes_)  #    


    models = {
        "KNN": (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }),
        "DecisionTree": (DecisionTreeClassifier(), {
            'criterion': ['gini', 'entropy'],
            'max_depth': [5, 10, 20, None]
        }),
        "RandomForest": (RandomForestClassifier(), {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'criterion': ['gini', 'entropy']
        }),
        
        "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric="logloss"), {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        })
        # "SVM": (SVC(probability=True), {
        #     'C': [0.1, 1, 10, 100],
        #     'kernel': ['linear', 'rbf', 'poly']
        # })
        
    }

    best_model = None
    best_model_name = None
    best_cv_score = 0
    best_cv_results = None
    best_overfit_penalty = float('inf')

    for model_name, (model, param_grid) in models.items():
        print(f"\nTraining {model_name} with hyperparameter tuning...")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_train_acc = grid_search.best_estimator_.score(X_train, y_train)  # Training accuracy
        best_cv_acc = grid_search.best_score_  # Cross-validation accuracy
        overfit_penalty = abs(best_train_acc - best_cv_acc)  # Overfitting measure

        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Train Accuracy: {best_train_acc:.4f} | CV Accuracy: {best_cv_acc:.4f} | Overfit Penalty: {overfit_penalty:.4f}")

        # Select the best model based on CV accuracy while considering overfitting
        if best_cv_acc > best_cv_score and overfit_penalty < best_overfit_penalty:
            best_cv_score = best_cv_acc
            best_model = grid_search.best_estimator_
            best_model_name = model_name
            best_cv_results = grid_search.cv_results_
            best_overfit_penalty = overfit_penalty

    print(f"\nBest Model (Balanced for Overfitting): {best_model_name} with CV Accuracy: {best_cv_score:.4f}")
    return best_model, best_cv_results, best_model_name, best_cv_score, label_encoder  

def evaluate_model(best_model, X_train, y_train, X_test, y_test, best_cv_score, model_name, label_encoder):
    """
    Evaluates the trained model using:
    - Sensitivity & Specificity for Tissue & Non-Tissue
    - Training, CV, and Test Errors
    """

    # Ensure the label encoder is correctly passed
    assert label_encoder is not None, "Error: label_encoder is None! Ensure train_models() returns it."

    # Convert both `y_train` and `y_test` to numeric labels before passing to compute_metrics()
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # `y_train_encoded` and `y_train_pred` are both numeric!
    sensitivity_train, specificity_train = compute_metrics(y_train_encoded, y_train_pred)
    sensitivity_test, specificity_test = compute_metrics(y_test_encoded, y_test_pred)

    # Compute Errors
    train_error = 1 - accuracy_score(y_train_encoded, y_train_pred)
    test_error = 1 - accuracy_score(y_test_encoded, y_test_pred)
    cv_error = 1 - best_cv_score

    # Print Results
    print("\n--- Model Evaluation ---")
    print(f"Best Model: {model_name}")
    print(f"Sensitivity (Tissue - Train): {sensitivity_train:.2f}")
    print(f"Specificity (Non-Tissue - Train): {specificity_train:.2f}")
    print(f"Sensitivity (Tissue - Test): {sensitivity_test:.2f}")
    print(f"Specificity (Non-Tissue - Test): {specificity_test:.2f}")
    print(f"\nTrain Error: {train_error:.4f}")
    print(f"Cross-Validation Error: {cv_error:.4f}")
    print(f"Test Error: {test_error:.4f}")


# def evaluate_model(best_model, X_train, y_train, X_test, y_test, best_cv_score, model_name, label_encoder):
#     """
#     Evaluates the trained model using:
#     - Sensitivity & Specificity for Tissue & Non-Tissue
#     - Training, CV, and Test Errors

#     Parameters:
#     - best_model: Trained ML model.
#     - X_train: Training features.
#     - y_train: True training labels.
#     - X_test: Test features.
#     - y_test: True test labels.
#     - best_cv_score: Best cross-validation accuracy.
#     - model_name: Name of the best-performing model.

#     Prints:
#     - Sensitivity & Specificity for Tissue and Non-Tissue
#     - Training, Cross-Validation, and Test Errors
#     """
#     #label_encoder = LabelEncoder()
#    # y_train = LabelEncoder.transform(y_train)
#     print("Classes in LabelEncoder (Before Transform):", label_encoder.classes_)
#     assert hasattr(label_encoder, "classes_"), "Error: label_encoder is not fitted!"

#     y_test = label_encoder.transform(y_test)

#     # Predictions on training and test sets
#     y_train_pred = best_model.predict(X_train)
#     y_test_pred = best_model.predict(X_test)

#     # Compute Sensitivity & Specificity for Training and Test Sets
#     sensitivity_train, specificity_train = compute_metrics(y_train, y_train_pred)
#     sensitivity_test, specificity_test = compute_metrics(y_test, y_test_pred)

#     # Compute Errors
#     train_error = 1 - accuracy_score(y_train, y_train_pred)
#     test_error = 1 - accuracy_score(y_test, y_test_pred)
#     cv_error = 1 - best_cv_score

#     # Print Results
#     print("\n--- Model Evaluation ---")
#     print(f"Best Model: {model_name}")
#     print(f"Sensitivity (Tissue - Train): {sensitivity_train:.2f}")
#     print(f"Specificity (Non-Tissue - Train): {specificity_train:.2f}")
#     print(f"Sensitivity (Tissue - Test): {sensitivity_test:.2f}")
#     print(f"Specificity (Non-Tissue - Test): {specificity_test:.2f}")

#     print(f"\nTrain Error: {train_error:.4f}")
#     print(f"Cross-Validation Error: {cv_error:.4f}")
#     print(f"Test Error: {test_error:.4f}")
