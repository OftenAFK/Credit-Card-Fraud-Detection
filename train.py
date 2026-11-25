
import joblib
import numpy as np
import pandas as pd

import sklearn.linear_model
import sklearn.neighbors
import sklearn.tree
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.decomposition
import sklearn.metrics
import sklearn.model_selection

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from data_preprocessing import load_and_preprocess_data


def train_models():

    print("Loading data...")

    data = load_and_preprocess_data('creditcard.csv')

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    # ==========================================
    # 1. ORIGINAL CLASS DISTRIBUTION
    # ==========================================

    print("\nOriginal training distribution:")

    original_counts = pd.Series(y_train).value_counts().sort_index()

    print(f"Normal transactions: {original_counts[0]:,}")
    print(f"Fraud transactions:  {original_counts[1]:,}")

    # ==========================================
    # 2. PCA FOR VISUALIZATION
    # ==========================================

    print("\nFitting PCA...")

    pca = sklearn.decomposition.PCA(
        n_components=2,
        random_state=42
    )

    # Fit PCA on original training data
    X_pca_train = pca.fit_transform(X_train)

    # ==========================================
    # 3. APPLY SMOTE
    # ==========================================

    print("\nApplying SMOTE...")

    smote = SMOTE(random_state=42)

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    # Keep DataFrame structure and feature names
    X_train_smote = pd.DataFrame(
        X_train_smote,
        columns=X_train.columns
    )

    y_train_smote = pd.Series(y_train_smote)

    # Extract only the newly generated fraud records
    synthetic_fraud = X_train_smote.iloc[len(X_train):].copy()

    synthetic_fraud = synthetic_fraud[
        np.asarray(y_train_smote.iloc[len(X_train):]) == 1
    ]

    # ==========================================
    # 4. DISPLAY NEW CLASS DISTRIBUTION
    # ==========================================

    print("\nTraining distribution after SMOTE:")

    smote_counts = y_train_smote.value_counts().sort_index()

    print(f"Normal transactions: {smote_counts[0]:,}")
    print(f"Fraud transactions:  {smote_counts[1]:,}")

    print(
        f"\nSynthetic fraud transactions generated: "
        f"{len(synthetic_fraud):,}"
    )

    # ==========================================
    # 5. INITIALIZE MODELS
    # ==========================================

    # class_weight is intentionally omitted because
    # SMOTE is now handling class imbalance.

    models = {

        'Logistic Regression':
            sklearn.linear_model.LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

        'Decision Tree':
            sklearn.tree.DecisionTreeClassifier(
                random_state=42,
                max_depth=10
            ),

        'Random Forest':
            sklearn.ensemble.RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),

        'Naive Bayes':
            sklearn.naive_bayes.GaussianNB(),

        'KNN':
            sklearn.neighbors.KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            )
    }

    # ==========================================
    # 6. CROSS VALIDATION
    # ==========================================

    results = {}
    cv_scores = {}

    cv = sklearn.model_selection.StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # ==========================================
    # 7. TRAIN MODELS AND EVALUATE
    # ==========================================

    print("\nTraining models and calculating metrics...")

    for name, model in models.items():

        print(f"\n-> Fitting {name}")

        # Train final model on SMOTE-balanced data
        model.fit(X_train_smote, y_train_smote)

        # Apply SMOTE independently inside each CV fold
        cv_pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', model.__class__(**model.get_params()))
        ])

        scores = sklearn.model_selection.cross_val_score(
            cv_pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring='f1',
            n_jobs=-1
        )

        cv_scores[name] = (
            f"{scores.mean().round(4)} "
            f"+/- {scores.std().round(4)}"
        )

        # Evaluate on original, untouched test data
        preds = model.predict(X_test)

        probabilities = model.predict_proba(X_test)

        # Identify the fraud probability column
        fraud_index = list(model.classes_).index(1)

        probas = probabilities[:, fraud_index]

        results[name] = {

            'accuracy': sklearn.metrics.accuracy_score(
                y_test, preds
            ),

            'precision': sklearn.metrics.precision_score(
                y_test, preds, zero_division=0
            ),

            'recall': sklearn.metrics.recall_score(
                y_test, preds, zero_division=0
            ),

            'f1': sklearn.metrics.f1_score(
                y_test, preds, zero_division=0
            ),

            'roc_auc': sklearn.metrics.roc_auc_score(
                y_test, probas
            ),

            'preds': preds,
            'probas': probas
        }

    # ==========================================
    # 8. PACKAGE ARTIFACTS FOR STREAMLIT
    # ==========================================

    artifacts = {

        'models': models,

        'results': results,

        'cv_scores': cv_scores,

        'pca': pca,

        'X_pca_train': X_pca_train,

        # Original training data
        'X_train': X_train,
        'y_train': y_train,

        # SMOTE-balanced training data
        'X_train_smote': X_train_smote,
        'y_train_smote': y_train_smote,

        # Only newly generated fraud records
        'synthetic_fraud': synthetic_fraud,

        # Original test data (never oversampled)
        'X_test': X_test,
        'y_test': y_test,

        'amount_scaler': data['amount_scaler'],
        'time_scaler': data['time_scaler']
    }

    joblib.dump(artifacts, 'saved_models.pkl')

    print("\nAll models trained successfully!")

    print("Saved trained models and evaluation data to saved_models.pkl")


if __name__ == "__main__":
    train_models()
