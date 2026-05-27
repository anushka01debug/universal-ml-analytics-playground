import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# PAGE CONFIG  ← must be the very first Streamlit call
# =========================================================
st.set_page_config(
    page_title="Universal ML Analytics Playground",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# SAFE DEFAULT VALUES  (avoids undefined-variable crashes)
# =========================================================
df = None
flower_names = []
feature_names = []
is_custom_dataset = False

# =========================================================
# THEME SELECTION  (sidebar call #1)
# =========================================================
with st.sidebar:
    st.markdown("## 🌸 ML Playground")
    theme_choice = st.selectbox(
        "🎨 App Visual Theme",
        ["Space Dark (Violet)", "Orchid Light (Clean)"],
        help="Toggle between premium dark mode and high-contrast light mode."
    )

if theme_choice == "Space Dark (Violet)":
    card_bg           = "#111827"
    main_bg           = "#090d16"
    text_color        = "#f3f4f6"
    sub_text_color    = "#9ca3af"
    border_color      = "#1f2937"
    plotly_template   = "plotly_dark"
    grid_color        = "#1f2937"
    gradient_colors   = "linear-gradient(135deg, #a855f7 0%, #ec4899 50%, #f43f5e 100%)"
    card_border_hover = "#a855f7"
else:
    card_bg           = "#ffffff"
    main_bg           = "#f3f4f6"
    text_color        = "#111827"
    sub_text_color    = "#4b5563"
    border_color      = "#e5e7eb"
    plotly_template   = "plotly"
    grid_color        = "#e5e7eb"
    gradient_colors   = "linear-gradient(135deg, #7c3aed 0%, #db2777 100%)"
    card_border_hover = "#7c3aed"

# =========================================================
# GLOBAL CSS
# =========================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
    .stApp {{ background-color: {main_bg}; color: {text_color}; }}
    .title-gradient {{
        background: {gradient_colors};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3rem;
        margin-bottom: 0.2rem; letter-spacing: -0.05em;
    }}
    .subtitle-text {{
        color: {sub_text_color}; font-size: 1.1rem;
        margin-bottom: 2rem; font-weight: 400;
    }}
    .premium-card {{
        background-color: {card_bg}; border-radius: 16px; padding: 24px;
        border: 1px solid {border_color};
        box-shadow: 0 4px 6px -1px rgba(0,0,0,.05), 0 2px 4px -1px rgba(0,0,0,.03);
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
        color: {text_color};
    }}
    .premium-card:hover {{
        border-color: {card_border_hover};
        box-shadow: 0 10px 15px -3px rgba(0,0,0,.07);
        transform: translateY(-2px);
    }}
    .metric-value  {{ font-size: 2.2rem; font-weight: 700; color: {card_border_hover}; }}
    .metric-label  {{ font-size: 0.9rem; color: {sub_text_color}; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 4px; }}
    .prediction-container {{
        background: linear-gradient(135deg,rgba(124,58,237,.1) 0%,rgba(219,39,119,.1) 100%);
        border: 2px solid {card_border_hover}; border-radius: 20px; padding: 30px;
        text-align: center; margin-top: 20px;
        backdrop-filter: blur(8px);
        box-shadow: 0 10px 30px rgba(124,58,237,.15);
    }}
    .prediction-title  {{ font-size: 1.8rem; font-weight: 800; color: {text_color}; margin-bottom: 8px; }}
    .prediction-result {{
        font-size: 2.5rem; font-weight: 900;
        background: {gradient_colors};
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-transform: uppercase; margin: 15px 0; letter-spacing: .05em;
    }}
    .confidence-badge {{
        background-color: rgba(124,58,237,.2); border: 1px solid {card_border_hover};
        color: {text_color}; padding: 6px 16px; border-radius: 50px;
        font-weight: 600; display: inline-block; font-size: 1rem;
    }}
    .sidebar-header {{ font-size: 1.3rem; font-weight: 700; color: {card_border_hover}; margin-bottom: 15px; }}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA LOADING HELPER
# =========================================================
@st.cache_data
def load_default_dataset():
    iris = load_iris()
    default_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    default_df["target"]  = iris.target
    default_df["species"] = [iris.target_names[i] for i in iris.target]
    return default_df, list(iris.target_names), list(iris.feature_names)

# =========================================================
# DATASET SOURCE  (sidebar section)
# =========================================================
with st.sidebar:
    st.markdown("<div class='sidebar-header'>📂 Dataset Source</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a Custom CSV Dataset", type=["csv"],
        help="Upload your own classification dataset. Defaults to the Iris dataset."
    )

# --- Resolve which dataset to use ---
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        if len(raw_df) < 10:
            st.error("Uploaded dataset is too small (need ≥ 10 rows). Falling back to Iris.")
            df, flower_names, feature_names = load_default_dataset()
        else:
            is_custom_dataset = True
            with st.sidebar:
                st.success("✅ Custom Dataset Uploaded!")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        df, flower_names, feature_names = load_default_dataset()
else:
    df, flower_names, feature_names = load_default_dataset()

# --- Custom dataset column picker ---
if is_custom_dataset:
    with st.sidebar:
        st.markdown("---")
        st.subheader("🎯 Column Configuration")
        all_columns     = list(raw_df.columns)
        target_col      = st.selectbox("Select Target Column (Classes)", all_columns, index=len(all_columns)-1)
        default_feats   = [c for c in all_columns if c != target_col and pd.api.types.is_numeric_dtype(raw_df[c])]
        feature_cols    = st.multiselect(
            "Select Feature Columns",
            [c for c in all_columns if c != target_col],
            default=default_feats
        )

    if not feature_cols:
        st.warning("Please select at least one numeric feature column.")
        df, flower_names, feature_names = load_default_dataset()
        is_custom_dataset = False
    else:
        clean_df            = raw_df.dropna(subset=[target_col] + feature_cols)
        le                  = LabelEncoder()
        clean_df            = clean_df.copy()
        clean_df["target"]  = le.fit_transform(clean_df[target_col].astype(str))
        clean_df["species"] = clean_df[target_col].astype(str)
        df                  = clean_df[feature_cols + ["target", "species"]].copy()
        flower_names        = list(le.classes_)
        feature_names       = list(feature_cols)

# =========================================================
# SIDEBAR CONFIG DASHBOARD
# =========================================================
with st.sidebar:
    st.markdown("<div class='sidebar-header'>⚙️ Config Dashboard</div>", unsafe_allow_html=True)

    # --- Data Partitioning ---
    st.subheader("📐 Data Partitioning")
    test_size = st.slider("Test Set Ratio", 0.10, 0.90, 0.30, 0.05,
        help="Fraction of data held out for testing.")
    random_state = st.number_input("Seed (Random State)", min_value=1, max_value=1000, value=42, step=1)

    st.markdown("---")
    st.subheader("🔧 Algorithm Hyperparameters")

    with st.expander("K-Nearest Neighbors (KNN)", expanded=False):
        knn_neighbors = st.slider("Neighbors (K)", 1, 25, 5)
        knn_weights   = st.selectbox("Weight Function", ["uniform", "distance"])

    with st.expander("Support Vector Machine (SVM)", expanded=False):
        svm_c      = st.slider("C (Regularization)", 0.01, 10.0, 1.0, step=0.1)
        svm_kernel = st.selectbox("Kernel type", ["rbf", "linear", "poly", "sigmoid"])

    with st.expander("Random Forest", expanded=False):
        rf_estimators = st.slider("Estimators (Trees)", 10, 200, 100, step=10)
        rf_max_depth  = st.slider("Max Depth (RF)",     1,  15,   6)

    with st.expander("Decision Tree", expanded=False):
        dt_max_depth = st.slider("Max Depth (DT)", 1, 15, 4)

    with st.expander("Logistic Regression", expanded=False):
        lr_c = st.slider("C (Inverse Regularization)", 0.01, 10.0, 1.0, step=0.1)

# =========================================================
# SAFETY GUARD  — should never be None at this point
# =========================================================
if df is None:
    df, flower_names, feature_names = load_default_dataset()

# =========================================================
# APP TITLE
# =========================================================
title_label    = "Universal ML Analytics Playground" if is_custom_dataset else "Iris Flower Classification & Analysis"
subtitle_label = "A premium, fully interactive platform for uploading datasets, visualising clusters, and evaluating machine learning models."
st.markdown(f"<h1 class='title-gradient'>{title_label}</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='subtitle-text'>{subtitle_label}</p>", unsafe_allow_html=True)

# =========================================================
# PREPARE DATA ARRAYS
# =========================================================
X = df[feature_names]
y = df["target"]

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 and np.min(np.bincount(y)) > 1 else None
    )
except Exception:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# =========================================================
# MODEL DEFINITIONS & TRAINING
# =========================================================
classifiers = {
    "KNN":                 KNeighborsClassifier(n_neighbors=knn_neighbors, weights=knn_weights),
    "Logistic Regression": LogisticRegression(C=lr_c, max_iter=500),
    "Decision Tree":       DecisionTreeClassifier(max_depth=dt_max_depth, random_state=random_state),
    "Random Forest":       RandomForestClassifier(n_estimators=rf_estimators, max_depth=rf_max_depth, random_state=random_state),
    "SVM":                 SVC(C=svm_c, kernel=svm_kernel, probability=True, random_state=random_state)
}

with st.spinner("Training models & calculating cross-validation scores…"):
    results           = []
    trained_pipelines = {}
    cv_n_splits       = min(5, max(2, len(y_train) // len(np.unique(y_train))))
    cv_strategy       = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=random_state)

    for name, clf in classifiers.items():
        pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', clf)])

        t0 = time.time()
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - t0

        preds    = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, preds)

        try:
            cv_res  = cross_validate(pipeline, X_train, y_train, cv=cv_strategy, scoring='accuracy')
            cv_mean = np.mean(cv_res['test_score'])
            cv_std  = np.std(cv_res['test_score'])
        except Exception:
            cv_mean, cv_std = test_acc, 0.0

        results.append({
            "Algorithm":               name,
            "Test Set Accuracy (%)":   round(test_acc * 100, 2),
            "CV Mean Accuracy (%)":    round(cv_mean * 100, 2),
            "CV Std Dev (%)":          round(cv_std  * 100, 2),
            "Training Time (ms)":      round(fit_time * 1000, 3)
        })
        trained_pipelines[name] = pipeline

    time.sleep(0.3)

results_df = pd.DataFrame(results)

# =========================================================
# TABS
# =========================================================
tab_explorer, tab_comparison, tab_predictor = st.tabs([
    "📊 Data Explorer & PCA",
    "🏆 Model Ranking & Comparison",
    "🤖 Inspect & Predict"
])

# ---------------------------------------------------------
# TAB 1 — DATA EXPLORER
# ---------------------------------------------------------
with tab_explorer:
    st.markdown("### 📊 Dataset Overview")

    col_a, col_b, col_c, col_d = st.columns(4)
    for col, label, value in zip(
        [col_a, col_b, col_c, col_d],
        ["Total Rows", "Features", "Target Classes", "Test Split Size"],
        [len(df), len(feature_names), len(flower_names), f"{int(test_size*100)}%"]
    ):
        with col:
            st.markdown(f"""
            <div class='premium-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{value}</div>
            </div>""", unsafe_allow_html=True)

    col_preview, col_desc = st.columns([3, 2])
    with col_preview:
        st.subheader("📋 Dataset Preview")
        st.dataframe(df, use_container_width=True, height=265)
    with col_desc:
        st.subheader("📝 Dataset Classes")
        class_summary = ", ".join([f"**{n}**" for n in flower_names])
        st.markdown(f"Target classes: {class_summary}.")
        if not is_custom_dataset:
            st.markdown("""
**Iris Dataset** — four botanical measurements (cm):
- **Sepal Length** and **Sepal Width**
- **Petal Length** and **Petal Width**
""")
        else:
            st.markdown("**Custom Dataset** — using your configured feature columns to predict the target label.")

    st.markdown("---")
    st.subheader("📈 High-Fidelity Visualizations")

    vis_mode = st.radio(
        "Choose Plot Type",
        ["PCA 2D Cluster Projection", "3D Feature Scatter Plot", "Parallel Coordinates Path", "Correlation Heatmap"],
        horizontal=True
    )

    if vis_mode == "PCA 2D Cluster Projection":
        st.markdown("**PCA** projects high-dimensional features into 2D to reveal how separable the target classes are.")
        try:
            pca   = PCA(n_components=2)
            X_sc  = StandardScaler().fit_transform(X)
            X_pca = pca.fit_transform(X_sc)
            pca_df = pd.DataFrame(X_pca, columns=["PC1","PC2"])
            pca_df["Class"] = df["species"].values
            ev = pca.explained_variance_ratio_ * 100
            fig = px.scatter(pca_df, x="PC1", y="PC2", color="Class",
                             color_discrete_sequence=["#a855f7","#ec4899","#3b82f6","#10b981","#f59e0b"],
                             title=f"PCA 2D Projection (Variance Explained: {sum(ev):.1f}%)")
            fig.update_layout(template=plotly_template, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font=dict(color=text_color),
                              xaxis=dict(gridcolor=grid_color, title=f"PC1 ({ev[0]:.1f}%)"),
                              yaxis=dict(gridcolor=grid_color, title=f"PC2 ({ev[1]:.1f}%)"))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"PCA failed: {e}")

    elif vis_mode == "3D Feature Scatter Plot":
        if len(feature_names) >= 3:
            c1, c2, c3 = st.columns(3)
            x_ax = c1.selectbox("X-Axis", feature_names, index=0)
            y_ax = c2.selectbox("Y-Axis", feature_names, index=min(2, len(feature_names)-1))
            z_ax = c3.selectbox("Z-Axis", feature_names, index=min(3, len(feature_names)-1))
            fig = px.scatter_3d(df, x=x_ax, y=y_ax, z=z_ax, color="species",
                                color_discrete_sequence=["#a855f7","#ec4899","#3b82f6","#10b981","#f59e0b"],
                                title=f"3D Scatter: {x_ax} vs {y_ax} vs {z_ax}")
            fig.update_layout(template=plotly_template,
                              scene=dict(bgcolor=card_bg,
                                         xaxis=dict(gridcolor=grid_color),
                                         yaxis=dict(gridcolor=grid_color),
                                         zaxis=dict(gridcolor=grid_color)),
                              paper_bgcolor="rgba(0,0,0,0)", font=dict(color=text_color),
                              margin=dict(l=0,r=0,b=0,t=40))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("3D Scatter requires at least 3 feature columns.")

    elif vis_mode == "Parallel Coordinates Path":
        fig = px.parallel_coordinates(df, color="target",
                                      color_continuous_scale=[(0,"#a855f7"),(0.5,"#ec4899"),(1,"#3b82f6")])
        fig.update_layout(template=plotly_template, paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=text_color), margin=dict(l=60,r=30,b=30,t=40))
        st.plotly_chart(fig, use_container_width=True)

    elif vis_mode == "Correlation Heatmap":
        corr = df.drop("species", axis=1).corr()
        fig  = px.imshow(corr.values, x=corr.columns, y=corr.index,
                         color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f",
                         title="Pearson Correlation Heatmap")
        fig.update_layout(template=plotly_template, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(color=text_color))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 2 — MODEL COMPARISON
# ---------------------------------------------------------
with tab_comparison:
    st.markdown("### 🏆 Live Algorithm Leaderboard & Comparison")

    sorted_df    = results_df.sort_values(
        by=["CV Mean Accuracy (%)","Test Set Accuracy (%)","Training Time (ms)"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    best_algo     = sorted_df.loc[0,"Algorithm"]
    best_cv_acc   = sorted_df.loc[0,"CV Mean Accuracy (%)"]
    best_test_acc = sorted_df.loc[0,"Test Set Accuracy (%)"]
    best_time     = sorted_df.loc[0,"Training Time (ms)"]

    st.markdown(f"""
    <div class="premium-card" style="background:linear-gradient(135deg,rgba(168,85,247,.15) 0%,rgba(236,72,153,.15) 100%);
         border:2px solid {card_border_hover}; margin-bottom:25px;">
        <div style="font-size:.95rem;color:{card_border_hover};font-weight:700;text-transform:uppercase;
             letter-spacing:.08em;margin-bottom:4px;">🏆 Top Performing Classifier</div>
        <div style="font-size:2.2rem;font-weight:800;color:{text_color};line-height:1.2;">
            {best_algo} <span style="font-size:1.5rem;color:#eab308;vertical-align:middle;margin-left:10px;">⭐⭐⭐⭐⭐</span>
        </div>
        <div style="color:{sub_text_color};font-size:1.05rem;margin-top:8px;">
            CV Accuracy <strong style="color:{text_color};">{best_cv_acc:.1f}%</strong> ·
            Test Accuracy <strong style="color:{text_color};">{best_test_acc:.1f}%</strong> ·
            Fit time <strong style="color:{text_color};">{best_time:.2f} ms</strong>
        </div>
    </div>""", unsafe_allow_html=True)

    rank_icons  = ["🥇","🥈","🥉","4️⃣","5️⃣"]
    cols_ranks  = st.columns(len(sorted_df))
    for idx, row in sorted_df.iterrows():
        icon = rank_icons[idx] if idx < len(rank_icons) else "🏅"
        with cols_ranks[idx]:
            st.markdown(f"""
            <div class="premium-card" style="text-align:center;
                 border-color:{card_border_hover if idx==0 else border_color};padding:15px;">
                <div style="font-size:1.8rem;">{icon}</div>
                <div style="font-weight:700;font-size:1.1rem;margin-top:5px;">{row['Algorithm']}</div>
                <div style="font-size:1.3rem;font-weight:800;color:{card_border_hover};margin-top:5px;">
                    {row['CV Mean Accuracy (%)']:.1f}%
                </div>
                <div style="font-size:.8rem;color:{sub_text_color};margin-top:3px;">
                    Test: {row['Test Set Accuracy (%)']:.1f}%
                </div>
            </div>""", unsafe_allow_html=True)

    st.subheader("📊 Performance Matrix")
    st.dataframe(
        results_df.style
            .background_gradient(cmap="Purples", subset=["Test Set Accuracy (%)","CV Mean Accuracy (%)"])
            .format({"Training Time (ms)": "{:.2f}"}),
        use_container_width=True
    )

    st.subheader("📉 Dynamic Comparison Graph")
    melted = results_df.melt(
        id_vars="Algorithm",
        value_vars=["Test Set Accuracy (%)","CV Mean Accuracy (%)"],
        var_name="Metric Type", value_name="Accuracy Value (%)"
    )
    fig_bar = px.bar(melted, x="Algorithm", y="Accuracy Value (%)", color="Metric Type",
                     barmode="group",
                     color_discrete_map={"Test Set Accuracy (%)":"#a855f7","CV Mean Accuracy (%)":"#3b82f6"},
                     text="Accuracy Value (%)",
                     title="Test Set Accuracy vs. 5-Fold Cross-Validation Accuracy")
    fig_bar.update_traces(textposition='outside', textfont_size=10)
    fig_bar.update_layout(
        template=plotly_template, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, linecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, linecolor=grid_color,
                   range=[min(50, sorted_df["CV Mean Accuracy (%)"].min()-10), 105]),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------
# TAB 3 — INSPECT & PREDICT
# ---------------------------------------------------------
with tab_predictor:
    col_inspect, col_pred = st.columns([1, 1])

    with col_inspect:
        st.subheader("🔍 Model Inspector")
        selected_model_name = st.selectbox("Select Model to Inspect", list(classifiers.keys()))

        pipeline    = trained_pipelines[selected_model_name]
        predictions = pipeline.predict(X_test)
        accuracy    = accuracy_score(y_test, predictions)

        explanations = {
            "KNN":                 "🧠 **K-Nearest Neighbors** classifies by proximity — it finds the K closest training points and runs a majority vote.",
            "Logistic Regression": "📈 **Logistic Regression** applies a Sigmoid function to a linear combination of features, producing calibrated probabilities.",
            "Decision Tree":       "🌳 **Decision Trees** split features at thresholds (e.g. 'Petal Length > 2.4') and follow branches down to a leaf prediction.",
            "Random Forest":       "🌲 **Random Forest** builds many independent trees on random data/feature subsets, then combines their votes to reduce overfitting.",
            "SVM":                 "🛡️ **Support Vector Machines** find the maximum-margin hyperplane separating classes, using kernels for non-linear boundaries."
        }
        st.markdown(f"""
        <div class="premium-card" style="border-left:5px solid {card_border_hover};padding:15px;margin-bottom:20px;">
            {explanations[selected_model_name]}
        </div>""", unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.metric(f"{selected_model_name} — Test Acc", f"{accuracy*100:.2f}%")
        row = results_df[results_df["Algorithm"] == selected_model_name].iloc[0]
        m2.metric("CV Avg Score", f"{row['CV Mean Accuracy (%)']:.1f}%",
                  delta=f"± {row['CV Std Dev (%)']:.1f}%")

        clf_step = pipeline.named_steps['classifier']
        if hasattr(clf_step, 'feature_importances_'):
            st.markdown("#### ⚡ Feature Importance")
            fi_df = pd.DataFrame({
                'Feature':        feature_names,
                'Importance (%)': np.round(clf_step.feature_importances_ * 100, 2)
            }).sort_values('Importance (%)', ascending=True)
            fig_fi = px.bar(fi_df, x='Importance (%)', y='Feature', orientation='h',
                            color='Importance (%)', color_continuous_scale="Purples",
                            title=f"Feature Importances — {selected_model_name}")
            fig_fi.update_layout(template=plotly_template, paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)", font=dict(color=text_color),
                                 coloraxis_showscale=False, height=250,
                                 margin=dict(l=10,r=10,b=10,t=30))
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info(f"ℹ️ {selected_model_name} has no explicit feature importances. "
                    "Switch to **Random Forest** or **Decision Tree** to see Gini importance bars.")

        st.markdown("**📌 Confusion Matrix**")
        cm = confusion_matrix(y_test, predictions)
        fig_cm = px.imshow(cm, x=flower_names, y=flower_names,
                           color_continuous_scale="Purples", text_auto=True,
                           labels=dict(x="Predicted", y="Actual"))
        fig_cm.update_layout(template=plotly_template, paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)", font=dict(color=text_color),
                             coloraxis_showscale=False, height=280,
                             margin=dict(l=10,r=10,b=10,t=10))
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("**📋 Classification Report**")
        report_df = pd.DataFrame(
            classification_report(y_test, predictions, target_names=flower_names, output_dict=True)
        ).transpose().round(3)
        st.dataframe(
            report_df.style.background_gradient(cmap="Purples",
                subset=[c for c in ["precision","recall","f1-score"] if c in report_df.columns]),
            use_container_width=True
        )

    with col_pred:
        st.subheader("🌼 Live Prediction Panel")
        st.write("Adjust the sliders and watch the model classify your input in real time.")

        user_vals = []
        s1, s2 = st.columns(2)
        for idx, feat in enumerate(feature_names):
            mn   = float(df[feat].min())
            mx   = float(df[feat].max())
            mean = float(df[feat].mean())
            span = mx - mn
            with (s1 if idx % 2 == 0 else s2):
                v = st.slider(feat,
                              min_value=round(mn - span*0.1, 1),
                              max_value=round(mx + span*0.1, 1),
                              value=round(mean, 1), step=0.1)
                user_vals.append(v)

        user_arr  = np.array([user_vals])
        pred      = pipeline.predict(user_arr)
        pred_name = flower_names[pred[0]]
        proba     = pipeline.predict_proba(user_arr)[0]
        conf      = proba[pred[0]] * 100

        st.markdown(f"""
        <div class="prediction-container">
            <div class="prediction-title">Classification Result</div>
            <div class="prediction-result">{pred_name.upper()}</div>
            <div class="confidence-badge">Confidence: {conf:.2f}%</div>
        </div>""", unsafe_allow_html=True)

        if not is_custom_dataset:
            img_urls = {
                "setosa":     "https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg",
                "versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
                "virginica":  "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
            }
            key = pred_name.lower()
            if key in img_urls:
                st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
                st.image(img_urls[key], caption=f"Iris {pred_name.capitalize()}", use_container_width=True)
        else:
            st.info("Custom dataset loaded — prediction complete.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:{sub_text_color};font-size:.85rem;'>"
    "Developed with Streamlit · Scikit-learn · Plotly Analytics</div>",
    unsafe_allow_html=True
        )
