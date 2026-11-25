import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
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

# models = artifacts['models']
# results = artifacts['results']
# cv_scores = artifacts['cv_scores']
# X_test = artifacts['X_test']
# y_test = artifacts['y_test']
# X_train = artifacts['X_train']

models = artifacts['models']
results = artifacts['results']
cv_scores = artifacts['cv_scores']

X_train = artifacts['X_train']
y_train = artifacts['y_train']

X_test = artifacts['X_test']
y_test = artifacts['y_test']

# Navigation sidebar
menu = st.sidebar.radio(
    "Navigation", 
    [
        "Exploratory Data Analysis", 
        "PCA Projection", 
        "Model Results & Metrics", 
        "Feature Importance", 
        "Test Custom Transaction",
        "Synthetic Transaction Generator",
        "SMOTE - Class Balancing"
    ]
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
                axis=0, color="#2e5c42"
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
    st.header("Feature Importance & Analysis")
    st.write(
        "Explore which features contribute most to the Random Forest "
        "model's predictions and how individual features behave across "
        "normal and fraudulent transactions."
    )

    # --------------------------------------------------
    # 1. CALCULATE FEATURE IMPORTANCE
    # --------------------------------------------------
    rf = models['Random Forest']

    importances = pd.Series(
        rf.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    st.divider()

    # --------------------------------------------------
    # 2. TOP 5 MOST IMPORTANT FEATURES
    # --------------------------------------------------
    st.subheader("Most Influential Features")

    top_features = importances.head(5)

    cols = st.columns(5)

    for i, (feature, importance) in enumerate(top_features.items()):
        cols[i].metric(
            feature,
            f"{importance * 100:.2f}%"
        )

    st.caption(
        "Importance values represent each feature's relative "
        "contribution to the Random Forest's total feature importance."
    )

    st.divider()

    # --------------------------------------------------
    # 3. TOP 15 FEATURE IMPORTANCE GRAPH
    # --------------------------------------------------
    st.subheader("Random Forest - Top 15 Features")

    top_15 = importances.head(15).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.barh(
        top_15.index,
        top_15.values,
        color="#3498DB"
    )

    for bar, value in zip(bars, top_15.values):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {value:.4f}",
            va='center',
            fontsize=9
        )

    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")
    ax.grid(axis='x', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # --------------------------------------------------
    # 4. CUMULATIVE FEATURE IMPORTANCE
    # --------------------------------------------------
    st.subheader("Cumulative Feature Importance")

    cumulative_importance = importances.cumsum()

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.plot(
        range(1, len(cumulative_importance) + 1),
        cumulative_importance.values * 100,
        marker='o',
        markersize=4,
        linewidth=2,
        color="#2ECC71"
    )

    ax2.axhline(
        80,
        color="#E74C3C",
        linestyle='--',
        label="80% importance"
    )

    ax2.set_title("Cumulative Importance Across Features")
    ax2.set_xlabel("Number of Top Features")
    ax2.set_ylabel("Cumulative Importance (%)")
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(alpha=0.2)

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    features_for_80 = (
        cumulative_importance.searchsorted(0.80) + 1
    )

    st.info(
        f"The top {features_for_80} features account for "
        f"at least 80% of the model's total feature importance."
    )

    st.divider()

    # --------------------------------------------------
    # 5. FEATURE IMPORTANCE TABLE
    # --------------------------------------------------
    st.subheader("Complete Feature Importance Ranking")

    importance_df = pd.DataFrame({
        'Feature': importances.index,
        'Importance': importances.values,
        'Importance (%)': importances.values * 100
    })

    importance_df.insert(
        0,
        'Rank',
        range(1, len(importance_df) + 1)
    )

    st.dataframe(
        importance_df.style.format({
            'Importance': '{:.5f}',
            'Importance (%)': '{:.2f}%'
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------
    # 6. INDIVIDUAL FEATURE ANALYSIS
    # --------------------------------------------------

    st.subheader("Individual Feature Analysis")

    selected_feature = st.selectbox(
        "Select a feature to examine",
        X_train.columns.tolist()
    )

    y_train = artifacts['y_train']
    # Use processed training data instead of df_raw
    feature_data = X_train[[selected_feature]].copy()

    # Attach corresponding training labels
    feature_data['Class'] = np.asarray(y_train)

    fig3, ax3 = plt.subplots(figsize=(10, 5))

    normal_values = feature_data[
        feature_data['Class'] == 0
    ][selected_feature]

    fraud_values = feature_data[
        feature_data['Class'] == 1
    ][selected_feature]

    ax3.hist(
        normal_values,
        bins=50,
        density=True,
        alpha=0.5,
        color="#3498DB",
        label="Normal"
    )

    ax3.hist(
        fraud_values,
        bins=50,
        density=True,
        alpha=0.6,
        color="#E74C3C",
        label="Fraud"
    )

    ax3.set_title(
        f"{selected_feature}: Normal vs Fraud Distribution"
    )
    ax3.set_xlabel(selected_feature)
    ax3.set_ylabel("Density")
    ax3.legend()
    ax3.grid(alpha=0.2)

    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.caption(
        "Distributions are normalized separately to make their shapes "
        "easier to compare despite the class imbalance. Features are shown "
        "after preprocessing."
    )


    # --------------------------------------------------
    # 7. INTERPRETATION
    # --------------------------------------------------
    st.subheader("Understanding Feature Importance")

    st.markdown(
        """
        - **Higher importance:** The feature contributes more to the
          Random Forest's decision-making process.
        - **Lower importance:** The feature contributes relatively
          less to the model's predictions.
        - **Cumulative importance:** Shows how much of the total
          importance is accounted for by the top-ranked features.

        **Important:** Feature importance does not establish causation.
        A highly ranked feature is not necessarily a direct indicator
        of fraudulent activity. Correlated features can also share or
        influence importance scores.
        """
    )


# ----------------------------
# Page 5: Prediction Interface
# ----------------------------


elif menu == "Test Custom Transaction":
    st.header("Custom Transaction Fraud Detection")

    st.write(
        "Enter transaction details and let a trained machine learning "
        "model evaluate whether the transaction is potentially fraudulent."
    )

    st.warning(
        "V1–V28 are anonymized PCA components, not real-world transaction "
        "fields. Their default value is 0. Arbitrary values may not represent "
        "realistic transactions."
    )

    # --------------------------------------------------
    # 1. MODEL SELECTION
    # --------------------------------------------------
    chosen_model_name = st.selectbox(
        "Choose Classification Model",
        list(models.keys())
    )

    selected_model = models[chosen_model_name]

    st.divider()

    # --------------------------------------------------
    # 2. TRANSACTION INPUT
    # --------------------------------------------------
    st.subheader("Transaction Details")

    c1, c2 = st.columns(2)

    with c1:
        user_amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=st.session_state.get('custom_amount', 150.0),
            step=10.0,
            key="custom_amount"
        )

    with c2:
        user_time = st.number_input(
            "Time Elapsed (seconds)",
            min_value=0.0,
            value=st.session_state.get('custom_time', 25000.0),
            step=100.0,
            key="custom_time"
        )

    st.subheader("PCA Features (V1–V28)")

    st.caption(
        "Enter values manually or leave them at 0 for experimentation."
    )

    v_dict = {}
    cols = st.columns(4)

    for i in range(1, 29):
        key = f"custom_V{i}"

        with cols[(i - 1) % 4]:
            v_dict[f'V{i}'] = st.number_input(
                f'V{i}',
                value=st.session_state.get(key, 0.0),
                step=0.1,
                format="%.4f",
                key=key
            )

    st.divider()

    # --------------------------------------------------
    # 3. PREDICTION
    # --------------------------------------------------
    if st.button("Predict Fraud Status", type="primary"):

        # Scale Amount and Time
        scaled_amount_val = artifacts['amount_scaler'].transform(
            [[user_amount]]
        )[0][0]

        scaled_time_val = artifacts['time_scaler'].transform(
            [[user_time]]
        )[0][0]

        # Prepare input
        input_data = v_dict.copy()

        input_data['scaled_amount'] = scaled_amount_val
        input_data['scaled_time'] = scaled_time_val

        input_df = pd.DataFrame(
            [input_data],
            columns=X_train.columns
        )

        # Predict
        pred = selected_model.predict(input_df)[0]

        if not hasattr(selected_model, 'predict_proba'):
            st.error("This model does not support probability estimation.")
            st.stop()

        probabilities = selected_model.predict_proba(input_df)[0]

        fraud_index = list(selected_model.classes_).index(1)
        normal_index = list(selected_model.classes_).index(0)

        fraud_prob = probabilities[fraud_index]
        normal_prob = probabilities[normal_index]

        # Store prediction so it survives Streamlit reruns
        st.session_state['custom_prediction_result'] = {
            'model_name': chosen_model_name,
            'prediction': pred,
            'fraud_prob': fraud_prob,
            'normal_prob': normal_prob,
            'input_df': input_df
        }

    # --------------------------------------------------
    # 4. DISPLAY STORED PREDICTION
    # --------------------------------------------------
    if 'custom_prediction_result' in st.session_state:

        result = st.session_state['custom_prediction_result']

        pred = result['prediction']
        fraud_prob = result['fraud_prob']
        normal_prob = result['normal_prob']
        input_df = result['input_df']
        model_name = result['model_name']

        st.divider()

        # --------------------------------------------------
        # 5. PREDICTION RESULT
        # --------------------------------------------------
        st.subheader("Prediction Result")

        if pred == 1:
            st.error("FRAUD DETECTED")
        else:
            st.success("NORMAL TRANSACTION")

        col1, col2 = st.columns(2)

        col1.metric(
            "Fraud Probability",
            f"{fraud_prob:.2%}"
        )

        col2.metric(
            "Normal Probability",
            f"{normal_prob:.2%}"
        )

        st.caption(
            f"Prediction generated using {model_name}."
        )

        # --------------------------------------------------
        # 6. PROBABILITY COMPARISON
        # --------------------------------------------------
        st.subheader("Prediction Probability")

        fig1, ax1 = plt.subplots(figsize=(8, 4))

        labels = ['Normal', 'Fraud']
        values = [normal_prob * 100, fraud_prob * 100]

        bars = ax1.bar(
            labels,
            values,
            color=['#3498DB', '#E74C3C'],
            width=0.5
        )

        for bar, value in zip(bars, values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{value:.2f}%',
                ha='center',
                va='bottom'
            )

        ax1.set_ylabel("Estimated Probability (%)")
        ax1.set_ylim(0, 110)
        ax1.set_title("Model Prediction Probabilities")
        ax1.grid(axis='y', alpha=0.2)

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        # --------------------------------------------------
        # 7. SHAP FEATURE CONTRIBUTIONS
        # --------------------------------------------------
        st.divider()
        st.subheader("What Influenced This Prediction?")

        st.write(
            "This chart explains which features pushed the model's "
            "prediction towards fraud or normal."
        )

        st.caption(
            "Positive SHAP values push towards fraud. Negative values "
            "push towards normal. Features are ranked by absolute contribution."
        )

        try:
            import shap

            # Use a small representative background sample
            background = shap.sample(
                X_train,
                min(30, len(X_train)),
                random_state=42
            )

            # Explain the probability of fraud (class 1)
            def fraud_probability(data):
                data_df = pd.DataFrame(
                    data,
                    columns=X_train.columns
                )

                probabilities = selected_model.predict_proba(data_df)

                fraud_index_local = list(
                    selected_model.classes_
                ).index(1)

                return probabilities[:, fraud_index_local]

            # Permutation explainer works across different model types
            explainer = shap.Explainer(
                fraud_probability,
                background,
                algorithm="permutation"
            )

            shap_values = explainer(
                input_df,
                max_evals=10 * (2 * len(X_train.columns) + 1)
            )

            # Extract feature contributions
            contributions = shap_values.values[0]

            importance_df = pd.DataFrame({
                'Feature': X_train.columns,
                'SHAP Value': contributions
            })

            importance_df['Absolute SHAP'] = (
                importance_df['SHAP Value'].abs()
            )

            # Select the 10 most influential features
            importance_df = importance_df.sort_values(
                'Absolute SHAP',
                ascending=False
            ).head(10)

            importance_df = importance_df.sort_values(
                'SHAP Value'
            )

            # Red = pushes towards fraud
            # Blue = pushes towards normal
            bar_colors = [
                '#E74C3C' if value > 0 else '#3498DB'
                for value in importance_df['SHAP Value']
            ]

            fig_shap, ax_shap = plt.subplots(figsize=(10, 6))

            ax_shap.barh(
                importance_df['Feature'],
                importance_df['SHAP Value'],
                color=bar_colors
            )

            ax_shap.axvline(
                0,
                color='black',
                linewidth=0.8
            )

            ax_shap.set_xlabel(
                "SHAP Value (Contribution to Fraud Probability)"
            )

            ax_shap.set_title(
                "Top 10 Features Influencing This Prediction"
            )

            ax_shap.grid(axis='x', alpha=0.2)

            ax_shap.spines['top'].set_visible(False)
            ax_shap.spines['right'].set_visible(False)

            fig_shap.tight_layout()
            st.pyplot(fig_shap)
            plt.close(fig_shap)

            col1, col2 = st.columns(2)

            col1.markdown(
                "<span style='color:#E74C3C'>"
                "■</span> Pushes towards fraud",
                unsafe_allow_html=True
            )

            col2.markdown(
                "<span style='color:#3498DB'>"
                "■</span> Pushes towards normal",
                unsafe_allow_html=True
            )

            st.caption(
                "SHAP values explain the selected model's individual "
                "prediction. They are contributions to the estimated "
                "fraud probability, not proof of causation."
            )

        except ImportError:
            st.warning(
                "SHAP is not installed. Run: "
                "python -m pip install shap"
            )

        except Exception as e:
            st.warning(
                f"Could not generate SHAP explanation: {e}"
            )

        # --------------------------------------------------
        # 8. INDIVIDUAL FEATURE COMPARISON
        # --------------------------------------------------
        st.divider()
        st.subheader("Transaction Comparison")

        selected_feature = st.selectbox(
            "Choose a feature to compare against existing transactions",
            X_train.columns.tolist(),
            key="comparison_feature"
        )

        y_train_local = np.asarray(artifacts['y_train'])

        normal_values = X_train.iloc[
            np.where(y_train_local == 0)[0]
        ][selected_feature]

        fraud_values = X_train.iloc[
            np.where(y_train_local == 1)[0]
        ][selected_feature]

        transaction_value = input_df[selected_feature].iloc[0]

        fig2, ax2 = plt.subplots(figsize=(10, 5))

        ax2.hist(
            normal_values,
            bins=50,
            density=True,
            alpha=0.5,
            color="#3498DB",
            label="Normal Transactions"
        )

        ax2.hist(
            fraud_values,
            bins=50,
            density=True,
            alpha=0.6,
            color="#E74C3C",
            label="Fraudulent Transactions"
        )

        ax2.axvline(
            transaction_value,
            color="#F39C12",
            linewidth=2.5,
            linestyle='--',
            label="Your Transaction"
        )

        ax2.set_title(
            f"{selected_feature}: Your Transaction vs Training Data"
        )

        ax2.set_xlabel("Processed Feature Value")
        ax2.set_ylabel("Density")
        ax2.legend()
        ax2.grid(alpha=0.2)

        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        st.caption(
            "The orange line represents your transaction's feature value. "
            "The distributions show the corresponding feature values in "
            "the original training data."
        )

        # --------------------------------------------------
        # 9. TRANSACTION INPUT SUMMARY
        # --------------------------------------------------
        st.divider()
        st.subheader("Transaction Input Summary")

        st.dataframe(
            input_df.T.rename(columns={0: 'Entered Value'}),
            use_container_width=True
        )

# --------------------------------------------------
# Page: 6 - TRANSACTION INPUT SUMMARY
# --------------------------------------------------

elif menu == "Synthetic Transaction Generator":

    from imblearn.over_sampling import SMOTE
    from sklearn.neighbors import NearestNeighbors

    st.header("Synthetic Transaction Generator")

    st.write(
        "Generate synthetic normal and fraudulent transactions "
        "and load them directly into the custom transaction page."
    )

    st.info(
        "Synthetic transactions are generated from patterns in "
        "the training data. They are not real-world verified transactions."
    )

    y_train = np.asarray(artifacts['y_train'])

    # ==========================================
    # 1. SYNTHETIC NORMAL TRANSACTION
    # ==========================================

    st.subheader("Synthetic Normal Transaction")

    st.write(
        "Generate a new synthetic transaction using nearby "
        "normal training samples."
    )

    if st.button(
        "Generate Synthetic Normal Transaction",
        key="generate_normal"
    ):

        generation_id = st.session_state.get(
            'normal_generation_id', 0
        ) + 1

        st.session_state['normal_generation_id'] = generation_id

        # Get only normal transactions
        normal_data = X_train.iloc[
            np.where(y_train == 0)[0]
        ].reset_index(drop=True)

        # Find nearby normal transactions
        neighbors = NearestNeighbors(n_neighbors=6)
        neighbors.fit(normal_data)

        rng = np.random.default_rng(generation_id)

        # Randomly select an existing normal transaction
        anchor_index = rng.integers(0, len(normal_data))

        # Find its nearest neighbors
        distances, indices = neighbors.kneighbors(
            normal_data.iloc[[anchor_index]]
        )

        # Exclude the transaction itself
        neighbor_index = rng.choice(indices[0][1:])

        # Generate a random interpolation factor
        alpha = rng.random()

        # Interpolate between the two transactions
        anchor = normal_data.iloc[anchor_index].to_numpy()
        neighbor = normal_data.iloc[neighbor_index].to_numpy()

        synthetic_values = anchor + alpha * (neighbor - anchor)

        synthetic_transaction = pd.Series(
            synthetic_values,
            index=X_train.columns
        )

        # Convert scaled values back to original values
        synthetic_transaction['Amount'] = (
            artifacts['amount_scaler'].inverse_transform(
                [[synthetic_transaction['scaled_amount']]]
            )[0][0]
        )

        synthetic_transaction['Time'] = (
            artifacts['time_scaler'].inverse_transform(
                [[synthetic_transaction['scaled_time']]]
            )[0][0]
        )

        st.session_state['synthetic_normal_transaction'] = (
            synthetic_transaction
        )

    # Display normal transaction
    if 'synthetic_normal_transaction' in st.session_state:

        transaction = st.session_state['synthetic_normal_transaction']

        st.success("Synthetic normal transaction generated!")

        st.metric("Expected Class", "Normal (0)")

        st.subheader("Transaction Features")

        st.dataframe(
            transaction.to_frame(name="Value"),
            use_container_width=True
        )

        if st.button(
            "Load Normal Transaction into Custom Fields",
            key="load_normal"
        ):

            st.session_state['custom_amount'] = float(
                transaction['Amount']
            )

            st.session_state['custom_time'] = float(
                transaction['Time']
            )

            for i in range(1, 29):
                st.session_state[f'custom_V{i}'] = float(
                    transaction[f'V{i}']
                )

            st.success(
                "Normal transaction loaded! "
                "Go to Test Custom Transaction."
            )

    st.divider()

    # ==========================================
    # 2. SYNTHETIC FRAUD TRANSACTION
    # ==========================================

    st.subheader("Synthetic Fraud Transaction")

    st.write(
        "Generate a new synthetic fraudulent transaction using SMOTE."
    )

    if st.button(
        "Generate Synthetic Fraud Transaction",
        key="generate_fraud"
    ):

        generation_id = st.session_state.get(
            'synthetic_generation_id', 0
        ) + 1

        st.session_state['synthetic_generation_id'] = generation_id

        smote = SMOTE(random_state=generation_id)

        X_resampled, y_resampled = smote.fit_resample(
            X_train, y_train
        )

        X_resampled = pd.DataFrame(
            X_resampled,
            columns=X_train.columns
        )

        # Extract newly generated fraud samples
        synthetic_samples = X_resampled.iloc[len(X_train):]

        if len(synthetic_samples) == 0:

            st.error("No synthetic fraud transaction was generated.")

        else:

            synthetic_transaction = synthetic_samples.sample(
                n=1,
                random_state=generation_id
            ).iloc[0].copy()

            # Convert scaled values back to original values
            synthetic_transaction['Amount'] = (
                artifacts['amount_scaler'].inverse_transform(
                    [[synthetic_transaction['scaled_amount']]]
                )[0][0]
            )

            synthetic_transaction['Time'] = (
                artifacts['time_scaler'].inverse_transform(
                    [[synthetic_transaction['scaled_time']]]
                )[0][0]
            )

            st.session_state['synthetic_transaction'] = (
                synthetic_transaction
            )

    # Display fraud transaction
    if 'synthetic_transaction' in st.session_state:

        transaction = st.session_state['synthetic_transaction']

        st.success("Synthetic fraud transaction generated!")

        st.metric("Expected Class", "Fraud (1)")

        st.subheader("Transaction Features")

        st.dataframe(
            transaction.to_frame(name="Value"),
            use_container_width=True
        )

        if st.button(
            "Load Fraud Transaction into Custom Fields",
            key="load_fraud"
        ):

            st.session_state['custom_amount'] = float(
                transaction['Amount']
            )

            st.session_state['custom_time'] = float(
                transaction['Time']
            )

            for i in range(1, 29):
                st.session_state[f'custom_V{i}'] = float(
                    transaction[f'V{i}']
                )

            st.session_state['transaction_loaded'] = True

            st.success(
                "Fraud transaction loaded! "
                "Go to Test Custom Transaction."
            )   


elif menu == "SMOTE - Class Balancing":

    st.header("SMOTE - Class Balancing")

    st.write(
        "The credit card fraud dataset contains significantly "
        "fewer fraudulent transactions than normal transactions. "
        "We use SMOTE (Synthetic Minority Over-sampling Technique) "
        "to generate additional synthetic fraud samples during training."
    )

    st.info(
        "SMOTE is applied only to the training data. "
        "The original test dataset remains completely untouched."
    )

    # ==========================================
    # 1. RETRIEVE TRAINING DATA
    # ==========================================

    original_labels = np.asarray(artifacts['y_train'])
    balanced_labels = np.asarray(artifacts['y_train_smote'])

    synthetic_fraud = artifacts['synthetic_fraud']

    original_counts = pd.Series(
        original_labels
    ).value_counts().sort_index()

    balanced_counts = pd.Series(
        balanced_labels
    ).value_counts().sort_index()

    original_normal = int(original_counts.get(0, 0))
    original_fraud = int(original_counts.get(1, 0))

    balanced_normal = int(balanced_counts.get(0, 0))
    balanced_fraud = int(balanced_counts.get(1, 0))

    synthetic_count = len(synthetic_fraud)

    # ==========================================
    # 2. SUMMARY METRICS
    # ==========================================

    st.subheader("SMOTE Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Original Fraud Records",
        f"{original_fraud:,}"
    )

    col2.metric(
        "Fraud Records After SMOTE",
        f"{balanced_fraud:,}"
    )

    col3.metric(
        "Synthetic Records Generated",
        f"{synthetic_count:,}",
        delta=f"+{synthetic_count:,}"
    )

    st.divider()

    # ==========================================
    # 3. BEFORE AND AFTER DISTRIBUTION
    # ==========================================

    st.subheader("Class Distribution Comparison")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Before SMOTE**")

        before_df = pd.DataFrame({
            "Transaction Type": ["Normal", "Fraud"],
            "Count": [original_normal, original_fraud]
        }).set_index("Transaction Type")

        st.bar_chart(
            before_df,
            color="#3498DB"
        )

        st.caption(
            f"Normal: {original_normal:,} | "
            f"Fraud: {original_fraud:,}"
        )

    with col2:

        st.markdown("**After SMOTE**")

        after_df = pd.DataFrame({
            "Transaction Type": ["Normal", "Fraud"],
            "Count": [balanced_normal, balanced_fraud]
        }).set_index("Transaction Type")

        st.bar_chart(
            after_df,
            color="#E74C3C"
        )

        st.caption(
            f"Normal: {balanced_normal:,} | "
            f"Fraud: {balanced_fraud:,}"
        )

    st.divider()

    # ==========================================
    # 4. DISTRIBUTION TABLE
    # ==========================================

    st.subheader("Detailed Distribution")

    distribution_df = pd.DataFrame({
        "Class": ["Normal", "Fraud"],
        "Before SMOTE": [
            original_normal,
            original_fraud
        ],
        "After SMOTE": [
            balanced_normal,
            balanced_fraud
        ]
    })

    distribution_df["Additional Records"] = (
        distribution_df["After SMOTE"] -
        distribution_df["Before SMOTE"]
    )

    st.dataframe(
        distribution_df,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # ==========================================
    # 5. EXPLORE SYNTHETIC FRAUD RECORDS
    # ==========================================

    st.subheader("Explore Synthetic Fraud Transactions")

    st.write(
        "The following records were generated by SMOTE "
        "and were not present in the original training dataset."
    )

    if synthetic_count > 0:

        number_of_samples = st.slider(
            "Number of synthetic records to display",
            min_value=1,
            max_value=min(50, synthetic_count),
            value=min(5, synthetic_count),
            key="smote_sample_count"
        )

        sample_df = synthetic_fraud.head(
            number_of_samples
        )

        st.dataframe(
            sample_df,
            use_container_width=True
        )

        st.caption(
            f"Showing {number_of_samples} out of "
            f"{synthetic_count:,} synthetic fraud records."
        )

        st.download_button(
            label="Download Synthetic Fraud Records (CSV)",
            data=synthetic_fraud.to_csv(index=False),
            file_name="synthetic_fraud_transactions.csv",
            mime="text/csv"
        )

    else:

        st.warning("No synthetic fraud records were found.")

    st.divider()

    # ==========================================
    # 6. HOW SMOTE WORKS
    # ==========================================

    st.subheader("How Does SMOTE Work?")

    st.write(
        "SMOTE identifies existing minority-class transactions "
        "and their nearest neighbours. It generates new synthetic "
        "transactions by interpolating between these examples."
    )

    st.latex(
        r"X_{new} = X_i + \\lambda (X_{neighbor} - X_i)"
    )

    st.write(
        "Where:"
    )

    st.markdown("""
    - **Xᵢ:** An existing fraudulent transaction.
    - **Xneighbor:** One of its nearest fraudulent neighbours.
    - **λ:** A randomly generated value between 0 and 1.
    - **Xnew:** The newly generated synthetic transaction.
    """)

    st.success(
        "SMOTE helps models learn from a more balanced training "
        "dataset while preserving the original test distribution "
        "for unbiased evaluation."
    )

    st.caption(
        "Note: SMOTE-generated records are synthetic feature "
        "vectors, not verified real-world fraudulent transactions."
    )
