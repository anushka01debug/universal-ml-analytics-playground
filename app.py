import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================================================
# PAGE CONFIG  — must be FIRST streamlit call
# =========================================================
st.set_page_config(
    page_title="Universal ML Analytics Playground",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# SAFE DEFAULTS
# =========================================================
df = None
flower_names = []
feature_names = []
is_custom_dataset = False

# =========================================================
# SIDEBAR — THEME
# =========================================================
with st.sidebar:
    st.markdown("## 🌸 ML Playground")
    theme_choice = st.selectbox(
        "🎨 App Visual Theme",
        ["Space Dark (Violet)", "Orchid Light (Clean)"]
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
# CSS — sidebar forced visible + all styling
# =========================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}}
.stApp {{
    background-color: {main_bg} !important;
}}
section[data-testid="stSidebar"] {{
    display: block !important;
    visibility: visible !important;
    background-color: {card_bg} !important;
    border-right: 1px solid {border_color} !important;
    min-width: 240px !important;
}}
section[data-testid="stSidebar"] > div {{
    display: block !important;
    visibility: visible !important;
}}
button[data-testid="collapsedControl"] {{
    display: none !important;
}}
.title-gradient {{
    background: {gradient_colors};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
    font-size: 3rem;
    margin-bottom: 0.2rem;
    letter-spacing: -0.05em;
    line-height: 1.2;
}}
.subtitle-text {{
    color: {sub_text_color};
    font-size: 1.1rem;
    margin-bottom: 2rem;
}}
.premium-card {{
    background-color: {card_bg};
    border-radius: 16px;
    padding: 24px;
    border: 1px solid {border_color};
    box-shadow: 0 4px 6px -1px rgba(0,0,0,.05);
    margin-bottom: 20px;
    color: {text_color};
    transition: all 0.3s ease;
}}
.premium-card:hover {{
    border-color: {card_border_hover};
    box-shadow: 0 10px 15px -3px rgba(0,0,0,.1);
    transform: translateY(-2px);
}}
.metric-value {{
    font-size: 2.2rem;
    font-weight: 700;
    color: {card_border_hover};
}}
.metric-label {{
    font-size: 0.85rem;
    color: {sub_text_color};
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-bottom: 4px;
}}
.prediction-container {{
    background: linear-gradient(135deg, rgba(124,58,237,.12) 0%, rgba(219,39,119,.12) 100%);
    border: 2px solid {card_border_hover};
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 10px 30px rgba(124,58,237,.15);
}}
.prediction-title {{
    font-size: 1.8rem;
    font-weight: 800;
    color: {text_color};
    margin-bottom: 8px;
}}
.prediction-result {{
    font-size: 2.5rem;
    font-weight: 900;
    background: {gradient_colors};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-transform: uppercase;
    margin: 15px 0;
    letter-spacing: .05em;
}}
.confidence-badge {{
    background-color: rgba(124,58,237,.2);
    border: 1px solid {card_border_hover};
    color: {text_color};
    padding: 6px 18px;
    border-radius: 50px;
    font-weight: 600;
    display: inline-block;
    font-size: 1rem;
}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data
def load_default_dataset():
    iris = load_iris()
    d = pd.DataFrame(iris.data, columns=iris.feature_names)
    d["target"]  = iris.target
    d["species"] = [iris.target_names[i] for i in iris.target]
    return d, list(iris.target_names), list(iris.feature_names)

# =========================================================
# SIDEBAR — DATASET SOURCE
# =========================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📂 Dataset Source")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        if len(raw_df) < 10:
            st.error("Dataset too small (need >= 10 rows). Using Iris instead.")
            df, flower_names, feature_names = load_default_dataset()
        else:
            is_custom_dataset = True
            with st.sidebar:
                st.success("Custom Dataset Loaded!")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        df, flower_names, feature_names = load_default_dataset()
else:
    df, flower_names, feature_names = load_default_dataset()

if is_custom_dataset:
    with st.sidebar:
        st.markdown("### 🎯 Column Config")
        all_cols     = list(raw_df.columns)
        target_col   = st.selectbox("Target Column", all_cols, index=len(all_cols)-1)
        def_feats    = [c for c in all_cols if c != target_col and pd.api.types.is_numeric_dtype(raw_df[c])]
        feature_cols = st.multiselect("Feature Columns", [c for c in all_cols if c != target_col], default=def_feats)

    if not feature_cols:
        st.warning("Select at least one feature column.")
        df, flower_names, feature_names = load_default_dataset()
        is_custom_dataset = False
    else:
        cdf = raw_df.dropna(subset=[target_col] + feature_cols).copy()
        le  = LabelEncoder()
        cdf["target"]  = le.fit_transform(cdf[target_col].astype(str))
        cdf["species"] = cdf[target_col].astype(str)
        df             = cdf[feature_cols + ["target","species"]].copy()
        flower_names   = list(le.classes_)
        feature_names  = list(feature_cols)

# =========================================================
# SIDEBAR — CONFIG DASHBOARD
# =========================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("### ⚙️ Config Dashboard")
    st.markdown("**📐 Data Partitioning**")
    test_size    = st.slider("Test Set Ratio", 0.10, 0.90, 0.30, 0.05)
    random_state = st.number_input("Random Seed", min_value=1, max_value=1000, value=42, step=1)
    st.markdown("---")
    st.markdown("**🔧 Algorithm Hyperparameters**")
    with st.expander("K-Nearest Neighbors (KNN)"):
        knn_neighbors = st.slider("Neighbors (K)", 1, 25, 5)
        knn_weights   = st.selectbox("Weight Function", ["uniform","distance"])
    with st.expander("Support Vector Machine (SVM)"):
        svm_c      = st.slider("C (Regularization)", 0.01, 10.0, 1.0, 0.1)
        svm_kernel = st.selectbox("Kernel", ["rbf","linear","poly","sigmoid"])
    with st.expander("Random Forest"):
        rf_estimators = st.slider("Trees", 10, 200, 100, 10)
        rf_max_depth  = st.slider("Max Depth (RF)", 1, 15, 6)
    with st.expander("Decision Tree"):
        dt_max_depth = st.slider("Max Depth (DT)", 1, 15, 4)
    with st.expander("Logistic Regression"):
        lr_c = st.slider("C (Inv. Regularization)", 0.01, 10.0, 1.0, 0.1)

if df is None:
    df, flower_names, feature_names = load_default_dataset()

# =========================================================
# TITLE
# =========================================================
title = "Universal ML Analytics Playground" if is_custom_dataset else "Iris Flower Classification & Analysis"
st.markdown(f"<h1 class='title-gradient'>{title}</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>A premium, fully interactive platform for uploading datasets, visualising clusters, and evaluating machine learning models.</p>", unsafe_allow_html=True)

# =========================================================
# DATA PREP
# =========================================================
X = df[feature_names]
y = df["target"]

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if np.min(np.bincount(y)) > 1 else None
    )
except:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# =========================================================
# MODEL TRAINING
# =========================================================
classifiers = {
    "KNN":                 KNeighborsClassifier(n_neighbors=knn_neighbors, weights=knn_weights),
    "Logistic Regression": LogisticRegression(C=lr_c, max_iter=500),
    "Decision Tree":       DecisionTreeClassifier(max_depth=dt_max_depth, random_state=random_state),
    "Random Forest":       RandomForestClassifier(n_estimators=rf_estimators, max_depth=rf_max_depth, random_state=random_state),
    "SVM":                 SVC(C=svm_c, kernel=svm_kernel, probability=True, random_state=random_state)
}

with st.spinner("Training models & running cross-validation..."):
    results, trained_pipelines = [], {}
    cv_splits   = min(5, max(2, len(y_train) // len(np.unique(y_train))))
    cv_strategy = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    for name, clf in classifiers.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
        t0   = time.time()
        pipe.fit(X_train, y_train)
        ft   = time.time() - t0
        preds     = pipe.predict(X_test)
        test_acc  = accuracy_score(y_test, preds)
        try:
            cv_res  = cross_validate(pipe, X_train, y_train, cv=cv_strategy, scoring="accuracy")
            cv_mean = np.mean(cv_res["test_score"])
            cv_std  = np.std(cv_res["test_score"])
        except:
            cv_mean, cv_std = test_acc, 0.0
        results.append({
            "Algorithm":             name,
            "Test Set Accuracy (%)": round(test_acc*100, 2),
            "CV Mean Accuracy (%)":  round(cv_mean*100, 2),
            "CV Std Dev (%)":        round(cv_std*100,  2),
            "Training Time (ms)":    round(ft*1000,     3)
        })
        trained_pipelines[name] = pipe

results_df = pd.DataFrame(results)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["📊 Data Explorer & PCA", "🏆 Model Ranking & Comparison", "🤖 Inspect & Predict"])

# ── TAB 1 ──────────────────────────────────────────────
with tab1:
    st.markdown("### 📊 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in zip(
        [c1,c2,c3,c4],
        ["Total Rows","Features","Target Classes","Test Split Size"],
        [len(df), len(feature_names), len(flower_names), f"{int(test_size*100)}%"]
    ):
        with col:
            st.markdown(f"<div class='premium-card'><div class='metric-label'>{lbl}</div><div class='metric-value'>{val}</div></div>", unsafe_allow_html=True)

    col_prev, col_desc = st.columns([3,2])
    with col_prev:
        st.subheader("📋 Dataset Preview (Sample)")
        st.dataframe(df, use_container_width=True, height=265)
    with col_desc:
        st.subheader("📝 Dataset Classes")
        classes_str = ", ".join([f"**{n}**" for n in flower_names])
        st.markdown(f"The active dataset contains measurements categorized into the following target classes: {classes_str}.")
        if not is_custom_dataset:
            st.markdown("""
**Iris Dataset Overview**:  
Four distinct botanical measurements are recorded (in cm):
- **Sepal Length** and **Sepal Width**
- **Petal Length** and **Petal Width**
""")
        else:
            st.markdown("**Custom Dataset Overview**: You are operating on a custom uploaded dataset.")

    st.markdown("---")
    st.subheader("📈 High-Fidelity Visualizations")
    vis_mode = st.radio("Choose Plot Type",
        ["PCA 2D Cluster Projection","3D Feature Scatter Plot","Parallel Coordinates Path","Correlation Heatmap"],
        horizontal=True)

    if vis_mode == "PCA 2D Cluster Projection":
        st.markdown("""
**Principal Component Analysis (PCA)** projects high-dimensional features into a 2D space.
This reveals how easily target categories form clusters and helps visualize model separability.
""")
        try:
            pca   = PCA(n_components=2)
            X_sc  = StandardScaler().fit_transform(X)
            X_pca = pca.fit_transform(X_sc)
            pdf   = pd.DataFrame(X_pca, columns=["PC1","PC2"])
            pdf["Class"] = df["species"].values
            ev    = pca.explained_variance_ratio_ * 100
            fig   = px.scatter(pdf, x="PC1", y="PC2", color="Class",
                color_discrete_sequence=["#a855f7","#ec4899","#3b82f6","#10b981","#f59e0b"],
                title=f"PCA 2D Projection (Total Variance Explained: {sum(ev):.1f}%)")
            fig.update_layout(template=plotly_template, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font=dict(color=text_color),
                xaxis=dict(gridcolor=grid_color, title=f"PC1 ({ev[0]:.1f}% variance)"),
                yaxis=dict(gridcolor=grid_color, title=f"PC2 ({ev[1]:.1f}% variance)"))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"PCA failed: {e}")

    elif vis_mode == "3D Feature Scatter Plot":
        if len(feature_names) >= 3:
            a1,a2,a3 = st.columns(3)
            xa = a1.selectbox("X-Axis Feature", feature_names, index=0)
            ya = a2.selectbox("Y-Axis Feature", feature_names, index=min(2,len(feature_names)-1))
            za = a3.selectbox("Z-Axis Feature", feature_names, index=min(3,len(feature_names)-1))
            fig = px.scatter_3d(df, x=xa, y=ya, z=za, color="species",
                color_discrete_sequence=["#a855f7","#ec4899","#3b82f6","#10b981","#f59e0b"],
                title=f"3D Scatter: {xa} vs {ya} vs {za}")
            fig.update_layout(template=plotly_template,
                scene=dict(bgcolor=card_bg, xaxis=dict(gridcolor=grid_color),
                    yaxis=dict(gridcolor=grid_color), zaxis=dict(gridcolor=grid_color)),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color=text_color),
                margin=dict(l=0,r=0,b=0,t=40))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("3D Scatter requires at least 3 feature columns.")

    elif vis_mode == "Parallel Coordinates Path":
        st.markdown("This plot illustrates the profile profiles of all flowers. You can see how each species forms clusters along the feature pathways.")
        fig = px.parallel_coordinates(df, color="target",
            labels={"target":"Target Class"},
            color_continuous_scale=[(0,"#a855f7"),(0.5,"#ec4899"),(1,"#3b82f6")])
        fig.update_layout(template=plotly_template, paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=text_color), margin=dict(l=60,r=30,b=30,t=40))
        st.plotly_chart(fig, use_container_width=True)

    elif vis_mode == "Correlation Heatmap":
        st.markdown("Pearson correlation matrix shows the linear dependence between the numerical features and the target label.")
        corr = df.drop("species", axis=1).corr()
        fig  = px.imshow(corr.values, x=corr.columns, y=corr.index,
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f",
            title="Pearson Correlation Heatmap")
        fig.update_layout(template=plotly_template, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font=dict(color=text_color),
            coloraxis_colorbar=dict(title="Correlation"))
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 2 ──────────────────────────────────────────────
with tab2:
    st.markdown("### 🏆 Live Algorithm Leaderboard & Comparison")
    sdf = results_df.sort_values(
        by=["CV Mean Accuracy (%)","Test Set Accuracy (%)","Training Time (ms)"],
        ascending=[False,False,True]).reset_index(drop=True)
    best = sdf.loc[0]

    st.markdown(f"""
<div class="premium-card" style="background:linear-gradient(135deg,rgba(168,85,247,.15),rgba(236,72,153,.15));border:2px solid {card_border_hover};margin-bottom:25px;">
  <div style="font-size:.9rem;color:{card_border_hover};font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">🏆 Top Performing Classifier</div>
  <div style="font-size:2.2rem;font-weight:800;color:{text_color};line-height:1.2;">{best['Algorithm']} <span style="font-size:1.4rem;color:#eab308;">⭐⭐⭐⭐⭐</span></div>
  <div style="color:{sub_text_color};font-size:1rem;margin-top:8px;">
    This model achieved the highest stability with an average 5-Fold Cross-Validation Accuracy of
    <strong style="color:{text_color};">{best['CV Mean Accuracy (%)']:.1f}%</strong> and a Test Set Accuracy of
    <strong style="color:{text_color};">{best['Test Set Accuracy (%)']:.1f}%</strong>, fitting in just
    <strong style="color:{text_color};">{best['Training Time (ms)']:.2f} ms</strong>.
  </div>
</div>""", unsafe_allow_html=True)

    st.subheader("🥇 Real-time Rankings")
    icons = ["🥇","🥈","🥉","4️⃣","5️⃣"]
    rank_cols = st.columns(len(sdf))
    for idx, row in sdf.iterrows():
        with rank_cols[idx]:
            st.markdown(f"""
<div class="premium-card" style="text-align:center;border-color:{card_border_hover if idx==0 else border_color};padding:15px;">
  <div style="font-size:1.8rem;">{icons[idx]}</div>
  <div style="font-weight:700;font-size:1.05rem;margin-top:5px;">{row['Algorithm']}</div>
  <div style="font-size:1.3rem;font-weight:800;color:{card_border_hover};margin-top:5px;">{row['CV Mean Accuracy (%)']:.1f}%</div>
  <div style="font-size:.8rem;color:{sub_text_color};margin-top:3px;">Test Acc: {row['Test Set Accuracy (%)']:.1f}%</div>
</div>""", unsafe_allow_html=True)

    st.write("To prevent all models from showing flat 100% accuracy, we evaluate them using two distinct methods: "
             "1) The **Test Set Accuracy** (measured on the held-out split) and 2) **5-Fold Cross-Validation Accuracy** "
             "(the average accuracy when splitting the training data into 5 folds sequentially). This offers a highly reliable performance baseline.")

    st.subheader("📊 Performance Matrix")
    st.dataframe(results_df.style.format({"Training Time (ms)":"{:.2f}"}), use_container_width=True)

    st.subheader("📉 Dynamic Comparison Graph")
    melted = results_df.melt(id_vars="Algorithm",
        value_vars=["Test Set Accuracy (%)","CV Mean Accuracy (%)"],
        var_name="Metric Type", value_name="Accuracy Value (%)")
    fig = px.bar(melted, x="Algorithm", y="Accuracy Value (%)", color="Metric Type",
        barmode="group",
        color_discrete_map={"Test Set Accuracy (%)":"#a855f7","CV Mean Accuracy (%)":"#3b82f6"},
        text="Accuracy Value (%)",
        title="Test Set Accuracy vs. 5-Fold Cross-Validation Accuracy")
    fig.update_traces(textposition="outside", textfont_size=10)
    fig.update_layout(template=plotly_template, plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color), yaxis=dict(gridcolor=grid_color,
        range=[min(50, sdf["CV Mean Accuracy (%)"].min()-10),105]),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 3 ──────────────────────────────────────────────
with tab3:
    col_ins, col_pred = st.columns([1,1])

    with col_ins:
        st.subheader("🔍 Model Inspector")
        sel  = st.selectbox("Select Model to Inspect", list(classifiers.keys()))
        pipe = trained_pipelines[sel]
        preds = pipe.predict(X_test)
        acc   = accuracy_score(y_test, preds)

        explanations = {
            "KNN":                 "🧠 **K-Nearest Neighbors (KNN)** classifies data points based on proximity. It maps your query point to its closest neighbors (like distance coordinates) and runs a majority vote. If K=5, it searches for the 5 closest neighbors.",
            "Logistic Regression": "📈 **Logistic Regression** is a probabilistic linear classifier. It computes linear boundaries and applies a Sigmoid function to map inputs into a probability score between 0 and 1, making it highly robust for linearly separable tasks.",
            "Decision Tree":       "🌳 **Decision Trees** split features sequentially using a tree-structured hierarchy. At each node, it tests threshold logic (e.g. 'Petal Length > 2.4cm') and continues branching down to the leaf node.",
            "Random Forest":       "🌲🌲 **Random Forest** is an ensemble method containing a group of independent Decision Trees. It injects random subsets of data and features to build unique trees and combines their votes, reducing overfitting.",
            "SVM":                 "🛡️ **Support Vector Machines (SVM)** seek the maximum margin hyperplane. It maps points into multi-dimensional space to draw clear separators. The Kernel allows it to classify non-linear clusters by projecting them upward."
        }
        st.markdown(f"""<div class="premium-card" style="border-left:5px solid {card_border_hover};padding:15px;margin-bottom:20px;">{explanations[sel]}</div>""", unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.metric(f"Selected model: {sel}", f"{acc*100:.2f}%")
        row = results_df[results_df["Algorithm"]==sel].iloc[0]
        m2.metric("CV Avg Score", f"{row['CV Mean Accuracy (%)']:.1f}%", delta=f"± {row['CV Std Dev (%)']:.1f}%")

        clf_step = pipe.named_steps["classifier"]
        if hasattr(clf_step, "feature_importances_"):
            st.markdown("#### ⚡ Feature Importance Analysis")
            fi = pd.DataFrame({
                "Feature": feature_names,
                "Importance (%)": np.round(clf_step.feature_importances_*100, 2)
            }).sort_values("Importance (%)", ascending=True)
            fig = px.bar(fi, x="Importance (%)", y="Feature", orientation="h",
                color="Importance (%)", color_continuous_scale="Purples",
                title=f"Feature Importances for {sel}")
            fig.update_layout(template=plotly_template, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font=dict(color=text_color),
                coloraxis_showscale=False, height=250, margin=dict(l=10,r=10,b=10,t=30))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"ℹ️ {sel} does not calculate explicit feature importances. Switch to **Random Forest** or **Decision Tree** in the inspector to view Gini feature importance bars.")

        st.markdown("**📌 Confusion Matrix**")
        cm  = confusion_matrix(y_test, preds)
        fig = px.imshow(cm, x=flower_names, y=flower_names,
            color_continuous_scale="Purples", text_auto=True,
            labels=dict(x="Predicted Species", y="Actual Species"))
        fig.update_layout(template=plotly_template, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font=dict(color=text_color),
            coloraxis_showscale=False, height=280, margin=dict(l=10,r=10,b=10,t=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**📋 Classification Performance Report**")
        rep = classification_report(y_test, preds, target_names=flower_names, output_dict=True)
        st.dataframe(pd.DataFrame(rep).transpose().round(3), use_container_width=True)

    with col_pred:
        st.subheader("🌼 Live Prediction Panel")
        st.write("Adjust the features below and click predict to see how the selected model classifies the input values.")

        user_vals = []
        s1, s2 = st.columns(2)
        for idx, feat in enumerate(feature_names):
            mn, mx, mean = float(df[feat].min()), float(df[feat].max()), float(df[feat].mean())
            span = mx - mn
            with (s1 if idx%2==0 else s2):
                v = st.slider(feat,
                    min_value=round(mn - span*.1, 1),
                    max_value=round(mx + span*.1, 1),
                    value=round(mean, 1), step=0.1)
                user_vals.append(v)

        user_arr  = np.array([user_vals])
        pred      = pipe.predict(user_arr)
        pred_name = flower_names[pred[0]]
        proba     = pipe.predict_proba(user_arr)[0]
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
                st.image(img_urls[key], caption=f"Iris {pred_name.capitalize()} in the wild", use_container_width=True)
        else:
            st.info("Custom dataset loaded — prediction complete.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(f"<div style='text-align:center;color:{sub_text_color};font-size:.85rem;'>Developed with Streamlit • Scikit-learn • Plotly Analytics</div>", unsafe_allow_html=True)
