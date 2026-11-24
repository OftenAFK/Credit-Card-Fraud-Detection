import joblib
import numpy as np
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
    st.caption("Understanding transaction patterns and identifying fraudulent behavior.")

    # Consistent colors
    NORMAL = "#3498DB"
    FRAUD = "#E74C3C"
    BG = "#FFFFFF"
    TEXT = "#2C3E50"

    # Common plot styling
    def style_plot(ax):
        ax.set_facecolor(BG)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#DDDDDD')
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.tick_params(colors=TEXT)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    # 1. CLASS DISTRIBUTION
    st.subheader("Class Distribution")

    counts = df_raw['Class'].value_counts()
    normal_count = counts.get(0, 0)
    fraud_count = counts.get(1, 0)
    total = len(df_raw)

    normal_pct = (normal_count / total) * 100
    fraud_pct = (fraud_count / total) * 100

    m1, m2, m3 = st.columns(3)

    m1.metric("Total Transactions", f"{total:,}")
    m2.metric("Normal Transactions", f"{normal_count:,}", f"{normal_pct:.2f}%")
    m3.metric("Fraudulent Transactions", f"{fraud_count:,}", f"{fraud_pct:.2f}%")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(7, 5))

        bars = ax1.bar(
            ['Normal', 'Fraud'],
            [normal_count, fraud_count],
            color=[NORMAL, FRAUD],
            width=0.55
        )

        ax1.set_yscale('log')
        ax1.set_title("Transaction Class Counts", fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylabel("Number of Transactions (Log Scale)")

        for bar, count in zip(bars, [normal_count, fraud_count]):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.15,
                f'{count:,}',
                ha='center',
                fontsize=10,
                fontweight='bold'
            )

        style_plot(ax1)
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(7, 5))

        wedges, texts, autotexts = ax2.pie(
            [normal_count, fraud_count],
            labels=['Normal', 'Fraud'],
            colors=[NORMAL, FRAUD],
            autopct='%1.2f%%',
            startangle=90,
            pctdistance=0.7,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2},
            textprops={'fontsize': 11}
        )

        ax2.set_title("Transaction Class Proportions", fontsize=14, fontweight='bold', pad=15)

        for text in autotexts:
            text.set_color('white')
            text.set_fontweight('bold')

        st.pyplot(fig2)
        plt.close(fig2)

    st.info(
        f"Fraudulent transactions account for only {fraud_pct:.2f}% "
        "of the dataset. This severe class imbalance makes accuracy "
        "alone an unreliable evaluation metric."
    )

    # 2. AMOUNT DISTRIBUTION
    st.subheader("Transaction Amount Distribution")

    normal = df_raw[df_raw['Class'] == 0]['Amount']
    fraud = df_raw[df_raw['Class'] == 1]['Amount']

    col1, col2 = st.columns(2)

    with col1:
        fig3, ax3 = plt.subplots(figsize=(7, 5))

        ax3.hist(
            normal,
            bins=50,
            density=True,
            alpha=0.65,
            color=NORMAL,
            label='Normal'
        )

        ax3.hist(
            fraud,
            bins=50,
            density=True,
            alpha=0.65,
            color=FRAUD,
            label='Fraud'
        )

        ax3.set_title("Amount Distribution by Class", fontsize=14, fontweight='bold', pad=15)
        ax3.set_xlabel("Transaction Amount")
        ax3.set_ylabel("Density")
        ax3.legend(frameon=False)

        style_plot(ax3)
        st.pyplot(fig3)
        plt.close(fig3)

    with col2:
        fig4, ax4 = plt.subplots(figsize=(7, 5))

        box = ax4.boxplot(
            [normal, fraud],
            tick_labels=['Normal', 'Fraud'],
            showfliers=False,
            patch_artist=True,
            medianprops={'color': 'black', 'linewidth': 2}
        )

        box['boxes'][0].set_facecolor(NORMAL)
        box['boxes'][1].set_facecolor(FRAUD)

        for item in box['boxes']:
            item.set_alpha(0.7)

        ax4.set_title("Amount Spread by Class", fontsize=14, fontweight='bold', pad=15)
        ax4.set_ylabel("Transaction Amount")

        style_plot(ax4)
        st.pyplot(fig4)
        plt.close(fig4)

    # 3. TIME DISTRIBUTION
    st.subheader("Transaction Time Distribution")

    fig5, ax5 = plt.subplots(figsize=(11, 5))

    ax5.hist(
        df_raw[df_raw['Class'] == 0]['Time'],
        bins=60,
        density=True,
        alpha=0.55,
        color=NORMAL,
        label='Normal'
    )

    ax5.hist(
        df_raw[df_raw['Class'] == 1]['Time'],
        bins=60,
        density=True,
        alpha=0.55,
        color=FRAUD,
        label='Fraud'
    )

    ax5.set_title("Transaction Activity Over Time", fontsize=15, fontweight='bold', pad=15)
    ax5.set_xlabel("Time (seconds)")
    ax5.set_ylabel("Density")
    ax5.legend(frameon=False)

    style_plot(ax5)
    st.pyplot(fig5)
    plt.close(fig5)

    # 4. CORRELATION WITH TARGET
    st.subheader("Correlations with Target Class")

    corr = df_raw.corr()['Class'].drop('Class').sort_values()

    fig6, ax6 = plt.subplots(figsize=(9, 10))

    colors = [
        FRAUD if value > 0 else NORMAL
        for value in corr
    ]

    bars = ax6.barh(
        corr.index,
        corr.values,
        color=colors,
        height=0.7
    )

    ax6.axvline(0, color='#555555', linewidth=0.8)

    ax6.set_title(
        "Feature Correlation with Fraud",
        fontsize=15,
        fontweight='bold',
        pad=15
    )

    ax6.set_xlabel("Pearson Correlation Coefficient")

    for bar, value in zip(bars, corr.values):
        offset = 0.003 if value >= 0 else -0.003

        ax6.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f'{value:.3f}',
            va='center',
            ha='left' if value >= 0 else 'right',
            fontsize=8
        )

    ax6.grid(axis='x', linestyle='--', alpha=0.3)
    ax6.set_axisbelow(True)

    style_plot(ax6)
    st.pyplot(fig6)
    plt.close(fig6)

# ----------------------------
# Page 2: PCA
# ----------------------------

elif menu == "PCA Projection":
    st.header("PCA 2D Visualization")
    st.write(
        "Explore how the transaction data is represented in two "
        "dimensions using Principal Component Analysis (PCA)."
    )

    # Retrieve PCA artifacts
    X_pca_train = artifacts['X_pca_train']
    y_train = artifacts['y_train']
    pca = artifacts['pca']

    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    pc1_variance = explained_variance[0] * 100
    pc2_variance = explained_variance[1] * 100
    total_variance = pc1_variance + pc2_variance

    # Class counts
    normal_count = (y_train == 0).sum()
    fraud_count = (y_train == 1).sum()

    # --------------------------------------------------
    # 1. PCA METRICS
    # --------------------------------------------------
    st.subheader("Explained Variance")

    col1, col2, col3 = st.columns(3)

    col1.metric("PC1 Variance", f"{pc1_variance:.2f}%")
    col2.metric("PC2 Variance", f"{pc2_variance:.2f}%")
    col3.metric("Combined Variance", f"{total_variance:.2f}%")

    st.caption(
        "Explained variance represents how much of the dataset's "
        "total variation is captured by each principal component."
    )

    st.divider()

    # --------------------------------------------------
    # 2. 2D PCA SCATTER PLOT
    # --------------------------------------------------
    st.subheader("Transaction Distribution in 2D")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        X_pca_train[y_train == 0, 0],
        X_pca_train[y_train == 0, 1],
        s=7,
        alpha=0.25,
        color="#3498DB",
        label=f"Normal ({normal_count:,})"
    )

    ax.scatter(
        X_pca_train[y_train == 1, 0],
        X_pca_train[y_train == 1, 1],
        s=18,
        alpha=0.8,
        color="#E74C3C",
        label=f"Fraud ({fraud_count:,})",
        zorder=3
    )

    ax.set_title("2D PCA Projection of Training Data", fontsize=14, pad=12)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    ax.legend(frameon=True)
    ax.grid(alpha=0.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Each point represents one transaction. Blue indicates normal "
        "transactions, while red indicates fraudulent transactions."
    )

    st.divider()

    # --------------------------------------------------
    # 3. INDIVIDUAL COMPONENT VARIANCE
    # --------------------------------------------------
    st.subheader("Variance Captured by Each Component")

    fig2, ax2 = plt.subplots(figsize=(9, 4))

    components = ["PC1", "PC2"]
    variances = [pc1_variance, pc2_variance]

    bars = ax2.bar(
        components,
        variances,
        color=["#3498DB", "#E74C3C"],
        width=0.5
    )

    for bar, value in zip(bars, variances):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11
        )

    ax2.set_ylabel("Explained Variance (%)")
    ax2.set_title("Explained Variance per Principal Component")
    ax2.grid(axis="y", alpha=0.2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.divider()

    # --------------------------------------------------
    # 4. PCA INTERPRETATION
    # --------------------------------------------------
    st.subheader("PCA Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Dataset composition**")
        st.metric("Normal Transactions", f"{normal_count:,}")
        st.metric("Fraudulent Transactions", f"{fraud_count:,}")

    with col2:
        st.write("**Dimensionality reduction**")
        st.write(
            f"The first two principal components retain "
            f"**{total_variance:.2f}%** of the total variance."
        )

        if total_variance >= 70:
            st.success("The 2D projection retains a substantial amount of variance.")
        elif total_variance >= 40:
            st.info("The 2D projection retains a moderate amount of variance.")
        else:
            st.warning(
                "The 2D projection retains a relatively small amount "
                "of the total variance."
            )

    st.info(
        "PCA is an unsupervised dimensionality reduction technique. "
        "It does not specifically try to separate fraudulent and normal "
        "transactions. Overlapping points in this visualization do not "
        "necessarily mean that a machine learning model cannot distinguish them."
    )


# ----------------------------
# Page 3: Model Metrics
# ----------------------------

elif menu == "Model Results & Metrics":
    st.header("Model Evaluation & Performance")
    st.write(
        "Compare the performance of different machine learning models "
        "and evaluate their ability to identify fraudulent transactions."
    )

    # --------------------------------------------------
    # 1. PREPARE RESULTS DATAFRAME
    # --------------------------------------------------
    table_data = {}

    for name, res in results.items():
        table_data[name] = {
            k: v for k, v in res.items()
            if k not in ['preds', 'probas']
        }

    res_df = pd.DataFrame(table_data).T

    # --------------------------------------------------
    # 2. MODEL PERFORMANCE COMPARISON
    # --------------------------------------------------
    st.subheader("Overall Model Performance")

    display_columns = [
        col for col in [
            'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
        ] if col in res_df.columns
    ]

    if display_columns:
        st.dataframe(
            res_df[display_columns].sort_values(
                'f1', ascending=False
            ).style.format("{:.4f}").highlight_max(
                axis=0, color="#d5f5e3"
            ),
            use_container_width=True
        )

    st.caption(
        "Models are ranked by F1-score. Higher precision means fewer "
        "false fraud alerts, while higher recall means fewer missed fraud cases."
    )

    st.divider()

    # --------------------------------------------------
    # 3. SELECT MODEL AND DISPLAY METRIC CARDS
    # --------------------------------------------------
    st.subheader("Individual Model Evaluation")

    selected_model_name = st.selectbox(
        "Select a model",
        list(results.keys())
    )

    selected_result = results[selected_model_name]

    col1, col2, col3, col4 = st.columns(4)

    if 'accuracy' in selected_result:
        col1.metric(
            "Accuracy",
            f"{selected_result['accuracy']:.4f}"
        )

    if 'precision' in selected_result:
        col2.metric(
            "Precision",
            f"{selected_result['precision']:.4f}"
        )

    if 'recall' in selected_result:
        col3.metric(
            "Recall",
            f"{selected_result['recall']:.4f}"
        )

    if 'f1' in selected_result:
        col4.metric(
            "F1-Score",
            f"{selected_result['f1']:.4f}"
        )

    if 'roc_auc' in selected_result:
        st.metric(
            "ROC-AUC Score",
            f"{selected_result['roc_auc']:.4f}"
        )

    st.divider()

    # --------------------------------------------------
    # 4. METRIC COMPARISON GRAPH
    # --------------------------------------------------
    st.subheader("Performance Comparison")

    comparison_metrics = [
        metric for metric in ['precision', 'recall', 'f1']
        if metric in res_df.columns
    ]

    if comparison_metrics:
        fig, ax = plt.subplots(figsize=(10, 5))

        res_df[comparison_metrics].plot(
            kind='bar',
            ax=ax,
            width=0.75,
            color=['#3498DB', '#E74C3C', '#2ECC71'][
                :len(comparison_metrics)
            ]
        )

        ax.set_title("Precision, Recall and F1-Score by Model")
        ax.set_ylabel("Score")
        ax.set_xlabel("Machine Learning Model")
        ax.set_ylim(0, 1.05)
        ax.legend(title="Metrics")
        ax.grid(axis='y', alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(rotation=20, ha='right')

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # --------------------------------------------------
    # 5. CONFUSION MATRIX
    # --------------------------------------------------
    st.subheader("Confusion Matrix")

    cm = sklearn.metrics.confusion_matrix(
        y_test,
        selected_result['preds'],
        labels=[0, 1]
    )

    fig2, ax2 = plt.subplots(figsize=(6, 5))

    disp = sklearn.metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Normal', 'Fraud']
    )

    disp.plot(
        cmap='Blues',
        ax=ax2,
        values_format=',d',
        colorbar=False
    )

    ax2.set_title(f"{selected_model_name} - Confusion Matrix")
    ax2.set_xlabel("Predicted Class")
    ax2.set_ylabel("Actual Class")

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # Explain confusion matrix values
    tn, fp, fn, tp = cm.ravel()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("True Negatives", f"{tn:,}")
        st.caption("Normal transactions correctly identified.")

        st.metric("False Positives", f"{fp:,}")
        st.caption("Normal transactions incorrectly flagged as fraud.")

    with col2:
        st.metric("False Negatives", f"{fn:,}")
        st.caption("Fraudulent transactions incorrectly classified as normal.")

        st.metric("True Positives", f"{tp:,}")
        st.caption("Fraudulent transactions correctly identified.")

    st.divider()

    # --------------------------------------------------
    # 6. ROC CURVES
    # --------------------------------------------------
    st.subheader("ROC Curve Comparison")

    fig3, ax3 = plt.subplots(figsize=(8, 6))

    for name, res in results.items():
        if res.get('probas') is None:
            continue

        scores = np.asarray(res['probas'])

        if scores.ndim == 2:
            scores = scores[:, 1]

        fpr, tpr, _ = sklearn.metrics.roc_curve(
            y_test, scores
        )

        auc_value = sklearn.metrics.roc_auc_score(
            y_test, scores
        )

        ax3.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{name} (AUC = {auc_value:.3f})"
        )

    ax3.plot(
        [0, 1], [0, 1],
        linestyle='--',
        color='gray',
        label='Random Classifier'
    )

    ax3.set_title("Receiver Operating Characteristic (ROC)")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.2)

    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.caption(
        "A higher ROC-AUC generally indicates better discrimination "
        "between normal and fraudulent transactions."
    )

    st.divider()

    # --------------------------------------------------
    # 7. PRECISION-RECALL CURVES
    # --------------------------------------------------
    st.subheader("Precision-Recall Curve Comparison")

    fig4, ax4 = plt.subplots(figsize=(8, 6))

    for name, res in results.items():
        if res.get('probas') is None:
            continue

        scores = np.asarray(res['probas'])

        if scores.ndim == 2:
            scores = scores[:, 1]

        precision_values, recall_values, _ = (
            sklearn.metrics.precision_recall_curve(
                y_test, scores
            )
        )

        pr_auc = sklearn.metrics.average_precision_score(
            y_test, scores
        )

        ax4.plot(
            recall_values,
            precision_values,
            linewidth=2,
            label=f"{name} (AP = {pr_auc:.3f})"
        )

    # Baseline: positive class prevalence
    baseline = np.mean(y_test)

    ax4.axhline(
        baseline,
        linestyle='--',
        color='gray',
        label=f"Random Baseline ({baseline:.3f})"
    )

    ax4.set_title("Precision-Recall Curve")
    ax4.set_xlabel("Recall")
    ax4.set_ylabel("Precision")
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1.05)
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.2)

    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    st.caption(
        "Precision-Recall curves are particularly useful for imbalanced "
        "datasets because they focus on the identification of the minority class."
    )

    st.divider()

    # --------------------------------------------------
    # 8. CROSS-VALIDATION RESULTS
    # --------------------------------------------------
    st.subheader("5-Fold Cross-Validation")

    cv_df = pd.DataFrame(
        list(cv_scores.items()),
        columns=['Model', 'CV F1 Mean +/- Std']
    )

    st.dataframe(
        cv_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Cross-validation evaluates model performance across five "
        "different training and validation splits. It helps assess "
        "how consistent the model's performance is."
    )

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