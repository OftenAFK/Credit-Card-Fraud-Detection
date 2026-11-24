import joblib
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.metrics
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Dashboard", layout="wide")

st.title("Credit Card Fraud Detection Dashboard")
st.write("Course project for 24CSE0316 (AI and Machine Learning)")

# Load trained objects
@st.cache_resource
def load_saved_stuff():
    return joblib.load('saved_models.pkl')

@st.cache_data
def load_raw_csv():
    return pd.read_csv('creditcard.csv')

try:
    artifacts = load_saved_stuff()
    df_raw = load_raw_csv()
except FileNotFoundError:
    st.error("Missing saved files! Run `python train.py` first to generate saved_models.pkl.")
    st.stop()

models = artifacts['models']
results = artifacts['results']
cv_scores = artifacts['cv_scores']
X_test = artifacts['X_test']
y_test = artifacts['y_test']
X_train = artifacts['X_train']

# Navigation sidebar
menu = st.sidebar.radio(
    "Navigation", 
    ["Exploratory Data Analysis", "PCA Projection", "Model Results & Metrics", "Feature Importance", "Test Custom Transaction"]
)

# ----------------------------
# Page 1: EDA
# ----------------------------
if menu == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis (EDA)")
    
    st.subheader("Class Distribution")
    counts = df_raw['Class'].value_counts()
    st.write(counts)
    
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        df_raw['Class'].value_counts().plot(kind='bar', ax=ax1)
        plt.title('Class counts (0 = normal, 1 = fraud)')
        plt.yscale('log')
        st.pyplot(fig1)
        
    with col2:
        st.write("Percentage breakdown:")
        st.write(df_raw['Class'].value_counts(normalize=True) * 100)
        st.info("Note: Fraud accounts for only 0.17% of total records. Accuracy alone is misleading here.")

    st.subheader("Amount & Time Distributions")
    fig2, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(df_raw[df_raw.Class == 0]['Amount'], bins=50, alpha=0.6, label='normal')
    axs[0].hist(df_raw[df_raw.Class == 1]['Amount'], bins=50, alpha=0.6, label='fraud')
    axs[0].set_title('Amount distribution')
    axs[0].legend()

    axs[1].hist(df_raw['Time'], bins=50)
    axs[1].set_title('Time distribution')
    st.pyplot(fig2)

    st.subheader("Correlations with Target Class")
    corr = df_raw.corr()['Class'].sort_values()
    fig3, ax3 = plt.subplots(figsize=(6, 8))
    corr.drop('Class').plot(kind='barh', ax=ax3)
    plt.title('Correlation of each feature with Class')
    st.pyplot(fig3)

# ----------------------------
# Page 2: PCA
# ----------------------------
elif menu == "PCA Projection":
    st.header("PCA 2D Visualization")
    
    X_pca_train = artifacts['X_pca_train']
    y_train = artifacts['y_train']
    pca = artifacts['pca']
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X_pca_train[y_train == 0, 0], X_pca_train[y_train == 0, 1], s=5, alpha=0.3, label='normal')
    ax.scatter(X_pca_train[y_train == 1, 0], X_pca_train[y_train == 1, 1], s=8, alpha=0.7, label='fraud', color='red')
    plt.legend()
    plt.title('2D PCA projection of training data')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    st.pyplot(fig)
    
    st.write("Explained variance ratio:", pca.explained_variance_ratio_)

# ----------------------------
# Page 3: Model Metrics
# ----------------------------
elif menu == "Model Results & Metrics":
    st.header("Model Evaluation")

    # Metrics summary dataframe
    table_data = {}
    for name, res in results.items():
        table_data[name] = {k: v for k, v in res.items() if k not in ['preds', 'probas']}
    
    res_df = pd.DataFrame(table_data).T
    st.subheader("Test Set Performance (Sorted by F1-Score)")
    st.dataframe(res_df.sort_values('f1', ascending=False))

    st.subheader("5-Fold Cross-Validation Scores (F1)")
    cv_df = pd.DataFrame(list(cv_scores.items()), columns=['Model', 'CV F1 Mean +/- Std'])
    st.table(cv_df)

    st.subheader("Confusion Matrix Display")
    selected_model_name = st.selectbox("Select Model to Display CM", list(models.keys()))
    
    cm = sklearn.metrics.confusion_matrix(y_test, results[selected_model_name]['preds'])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=['normal', 'fraud'])
    disp.plot(cmap='Blues', ax=ax)
    plt.title(selected_model_name)
    st.pyplot(fig)

    st.subheader("ROC Curves")
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    for name, res in results.items():
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, res['probas'])
        plt.plot(fpr, tpr, label=f"{name} (auc={res['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(fontsize=8)
    plt.title('ROC curves')
    st.pyplot(fig2)

# ----------------------------
# Page 4: Feature Importance
# ----------------------------
elif menu == "Feature Importance":
    st.header("Feature Importance")
    
    rf = models['Random Forest']
    importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    st.subheader("Random Forest - Top 15 Features")
    fig, ax = plt.subplots(figsize=(7, 5))
    importances.head(15).plot(kind='barh', ax=ax)
    plt.gca().invert_yaxis()
    plt.title('Random Forest feature importance (top 15)')
    st.pyplot(fig)

# ----------------------------
# Page 5: Prediction Interface
# ----------------------------
elif menu == "Test Custom Transaction":
    st.header("Predict Single Transaction Fraud")
    st.caption("V1-V28 are anonymized PCA components from the original dataset - they don't map to "
               "real-world fields, so leave them at 0 unless you're testing specific values from the data.")

    chosen_model_name = st.selectbox("Choose Classification Model", list(models.keys()))
    selected_model = models[chosen_model_name]

    c1, c2 = st.columns(2)
    with c1:
        user_amount = st.number_input("Transaction Amount ($)", value=150.0)
    with c2:
        user_time = st.number_input("Time Elapsed (seconds)", value=25000.0)

    st.write("##### PCA Features (V1 - V28):")
    v_dict = {}
    cols = st.columns(4)
    for i in range(1, 29):
        v_dict[f'V{i}'] = cols[(i - 1) % 4].number_input(f'V{i}', value=0.0, step=0.1)

    if st.button("Predict Fraud Status"):
        # Scale input values
        scaled_amount_val = artifacts['amount_scaler'].transform([[user_amount]])[0][0]
        scaled_time_val = artifacts['time_scaler'].transform([[user_time]])[0][0]

        # Structure dictionary matching feature column order
        input_data = v_dict.copy()
        input_data['scaled_amount'] = scaled_amount_val
        input_data['scaled_time'] = scaled_time_val

        input_df = pd.DataFrame([input_data])[X_train.columns]

        pred = selected_model.predict(input_df)[0]
        prob = selected_model.predict_proba(input_df)[0][1]

        st.write("---")
        if pred == 1:
            st.error(f"Prediction: **FRAUD DETECTED** | Fraud Probability: {prob:.2%}")
        else:
            st.success(f"Prediction: **NORMAL TRANSACTION** | Fraud Probability: {prob:.2%}")