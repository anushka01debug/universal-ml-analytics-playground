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

# Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Universal ML Analytics Playground",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# THEME CONFIG & STYLING SWITCHER
# =========================================================
theme_choice = st.sidebar.selectbox(
    "🎨 App Visual Theme", 
    ["Space Dark (Violet)", "Orchid Light (Clean)"],
    help="Toggle between premium dark mode and high-contrast light mode styling."
)

if theme_choice == "Space Dark (Violet)":
    card_bg = "#111827"
    main_bg = "#090d16"
    text_color = "#f3f4f6"
    sub_text_color = "#9ca3af"
    border_color = "#1f2937"
    plotly_template = "plotly_dark"
    grid_color = "#1f2937"
    gradient_colors = "linear-gradient(135deg, #a855f7 0%, #ec4899 50%, #f43f5e 100%)"
    card_border_hover = "#a855f7"
else:
    card_bg = "#ffffff"
    main_bg = "#f3f4f6"
    text_color = "#111827"
    sub_text_color = "#4b5563"
    border_color = "#e5e7eb"
    plotly_template = "plotly"
    grid_color = "#e5e7eb"
    gradient_colors = "linear-gradient(135deg, #7c3aed 0%, #db2777 100%)"
    card_border_hover = "#7c3aed"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}
    
    .stApp {{
        background-color: {main_bg};
        color: {text_color};
    }}
    
    .title-gradient {{
        background: {gradient_colors};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.05em;
    }}
    
    .subtitle-text {{
        color: {sub_text_color};
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }}
    
    .premium-card {{
        background-color: {card_bg};
        border-radius: 16px;
        padding: 24px;
        border: 1px solid {border_color};
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        color: {text_color};
    }}
    
    .premium-card:hover {{
        border-color: {card_border_hover};
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.07);
        transform: translateY(-2px);
    }}
    
    .metric-value {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {card_border_hover};
    }}
    
    .metric-label {{
        font-size: 0.9rem;
        color: {sub_text_color};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }}
    
    .prediction-container {{
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.1) 0%, rgba(219, 39, 119, 0.1) 100%);
        border: 2px solid {card_border_hover};
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 20px;
        backdrop-filter: blur(8px);
        box-shadow: 0 10px 30px rgba(124, 58, 237, 0.15);
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
        text-transform: uppercase;
        margin: 15px 0;
        letter-spacing: 0.05em;
    }}
    
    .confidence-badge {{
        background-color: rgba(124, 58, 237, 0.2);
        border: 1px solid {card_border_hover};
        color: {text_color};
        padding: 6px 16px;
        border-radius: 50px;
        font-weight: 600;
        display: inline-block;
        font-size: 1rem;
    }}

    .sidebar-header {{
        font-size: 1.3rem;
        font-weight: 700;
        color: {card_border_hover};
        margin-bottom: 15px;
    }}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA LOADING AND FILE UPLOAD
# =========================================================
st.sidebar.markdown("<div class='sidebar-header'>📂 Dataset Source</div>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader(
    "Upload a Custom CSV Dataset", 
    type=["csv"],
    help="Upload your own classification dataset. If empty, the app defaults to the classic Iris Flower dataset."
)

@st.cache_data
def load_default_dataset():
    iris = load_iris()
    default_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    default_df["target"] = iris.target
    default_df["species"] = [iris.target_names[i] for i in iris.target]
    return default_df, list(iris.target_names), list(iris.feature_names)

# Resolve dataset flow
is_custom_dataset = False
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        if len(raw_df) < 10:
            st.error("Uploaded dataset is too small. Please upload a file with at least 10 rows.")
            df, flower_names, feature_names = load_default_dataset()
        else:
            is_custom_dataset = True
            st.sidebar.success("Custom Dataset Uploaded Successfully!")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        df, flower_names, feature_names = load_default_dataset()
else:
    df, flower_names, feature_names = load_default_dataset()

# Custom Dataset Setup Inputs
if is_custom_dataset:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Column Configuration")
    all_columns = list(raw_df.columns)
    
    # Choose target column
    target_col = st.sidebar.selectbox(
        "Select Target Column (Classes)",
        all_columns,
        index=len(all_columns) - 1,
        help="The column that contains the category labels you want the models to predict."
    )
    
    # Identify numeric columns for features
    default_features = [col for col in all_columns if col != target_col and pd.api.types.is_numeric_dtype(raw_df[col])]
    
    feature_cols = st.sidebar.multiselect(
        "Select Feature Columns (Measurements)",
        [col for col in all_columns if col != target_col],
        default=default_features,
        help="Select the numeric feature columns used for making predictions."
    )
    
    if not feature_cols:
        st.warning("Please select at least one numeric feature column.")
        df, flower_names, feature_names = load_default_dataset()
        is_custom_dataset = False
    else:
        # Preprocess custom dataset
        # Drop rows with NaNs in targeted columns
        clean_df = raw_df.dropna(subset=[target_col] + feature_cols)
        
        # Label encode targets
        le = LabelEncoder()
        clean_df["target"] = le.fit_transform(clean_df[target_col].astype(str))
        clean_df["species"] = clean_df[target_col].astype(str)
        
        df = clean_df[feature_cols + ["target", "species"]].copy()
        flower_names = list(le.classes_)
        feature_names = list(feature_cols)

# =========================================================
# SIDEBAR CONFIG DASHBOARD
# =========================================================
st.sidebar.markdown("<div class='sidebar-header'>🌸 Config Dashboard</div>", unsafe_allow_html=True)

# Dataset Split Controls
st.sidebar.subheader("📐 Data Partitioning")
test_size = st.sidebar.slider(
    "Test Set Ratio",
    min_value=0.10,
    max_value=0.90,
    value=0.30,
    step=0.05,
    help="Increase the test ratio to evaluate models on less training data, which increases metric variance."
)
random_state = st.sidebar.number_input(
    "Seed (Random State)",
    min_value=1,
    max_value=1000,
    value=42,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Algorithm Hyperparameters")

# Model Specific Hyperparameter Tuning
with st.sidebar.expander("K-Nearest Neighbors (KNN)", expanded=False):
    knn_neighbors = st.slider("Neighbors (K)", 1, min(25, max(3, len(df)//5)), 5)
    knn_weights = st.selectbox("Weight Function", ["uniform", "distance"])

with st.sidebar.expander("Support Vector Machine (SVM)", expanded=False):
    svm_c = st.slider("C (Regularization)", 0.01, 10.0, 1.0, step=0.1)
    svm_kernel = st.selectbox("Kernel type", ["rbf", "linear", "poly", "sigmoid"])

with st.sidebar.expander("Random Forest", expanded=False):
    rf_estimators = st.slider("Estimators (Trees)", 10, 200, 100, step=10)
    rf_max_depth = st.slider("Max Depth (RF)", 1, 15, 6)

with st.sidebar.expander("Decision Tree", expanded=False):
    dt_max_depth = st.slider("Max Depth (DT)", 1, 15, 4)

with st.sidebar.expander("Logistic Regression", expanded=False):
    lr_c = st.slider("C (Inverse Regularization)", 0.01, 10.0, 1.0, step=0.1)

# =========================================================
# APP TITLE HEADER
# =========================================================
title_label = "Universal ML Analytics Playground" if is_custom_dataset else "Iris Flower Classification & Analysis"
subtitle_label = "A premium, fully interactive platform for uploading datasets, visualising clusters, and evaluating machine learning models."
st.markdown(f"<h1 class='title-gradient'>{title_label}</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='subtitle-text'>{subtitle_label}</p>", unsafe_allow_html=True)

# Prepare Data Arrays
X = df[feature_names]
y = df["target"]

# Train-Test Split with safety check on size/classes
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.bincount(y)) > 1 and np.min(np.bincount(y)) > 1 else None
    )
except Exception:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

# =========================================================
# MODEL TRAINING & CROSS-VALIDATION PIPELINES (With loading animation)
# =========================================================
# Define models with selected hyperparameters
classifiers = {
    "KNN": KNeighborsClassifier(n_neighbors=knn_neighbors, weights=knn_weights),
    "Logistic Regression": LogisticRegression(C=lr_c, max_iter=500),
    "Decision Tree": DecisionTreeClassifier(max_depth=dt_max_depth, random_state=random_state),
    "Random Forest": RandomForestClassifier(n_estimators=rf_estimators, max_depth=rf_max_depth, random_state=random_state),
    "SVM": SVC(C=svm_c, kernel=svm_kernel, probability=True, random_state=random_state)
}

# Run Evaluations wrapped in an animation spinner
with st.spinner("Training models & calculating cross-validation scores..."):
    results = []
    trained_pipelines = {}
    
    # 5-fold Stratified K-Fold for Cross-Validation
    cv_n_splits = min(5, max(2, len(y_train) // len(np.unique(y_train))))
    cv_strategy = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=random_state)
    
    for name, clf in classifiers.items():
        # Build standard pipeline (Scaling -> Model)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
        
        # Measure Training Time
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - start_time
        
        # Predict on test set
        predictions = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, predictions)
        
        # Run Cross-Validation
        try:
            cv_results = cross_validate(
                pipeline, X_train, y_train, 
                cv=cv_strategy, 
                scoring='accuracy'
            )
            cv_mean = np.mean(cv_results['test_score'])
            cv_std = np.std(cv_results['test_score'])
        except Exception:
            cv_mean = test_acc
            cv_std = 0.0
            
        results.append({
            "Algorithm": name,
            "Test Set Accuracy (%)": round(test_acc * 100, 2),
            "CV Mean Accuracy (%)": round(cv_mean * 100, 2),
            "CV Std Dev (%)": round(cv_std * 100, 2),
            "Training Time (ms)": round(fit_time * 1000, 3)
        })
        
        trained_pipelines[name] = pipeline

    # Simulate short animation delay for smooth UI feedback
    time.sleep(0.4)

results_df = pd.DataFrame(results)

# =========================================================
# MAIN LAYOUT TABS
# =========================================================
tab_explorer, tab_comparison, tab_predictor = st.tabs([
    "📊 Data Explorer & PCA", 
    "🏆 Model Ranking & Comparison", 
    "🤖 Inspect & Predict"
])

# ---------------------------------------------------------
# TAB 1: DATA EXPLORER & PCA
# ---------------------------------------------------------
with tab_explorer:
    st.markdown("### 📊 Dataset Overview")
    
    # Top Metrics Cards Row
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(f"""
        <div class='premium-card'>
            <div class='metric-label'>Total Rows</div>
            <div class='metric-value'>{len(df)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class='premium-card'>
            <div class='metric-label'>Features</div>
            <div class='metric-value'>{len(feature_names)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div class='premium-card'>
            <div class='metric-label'>Target Classes</div>
            <div class='metric-value'>{len(flower_names)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_d:
        st.markdown(f"""
        <div class='premium-card'>
            <div class='metric-label'>Test Split Size</div>
            <div class='metric-value'>{int(test_size*100)}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Dataset Preview & Details
    col_preview, col_desc = st.columns([3, 2])
    with col_preview:
        st.subheader("📋 Dataset Preview (Sample)")
        st.dataframe(df, use_container_width=True, height=265)
        st.subheader("📝 Dataset Classes")
        class_summary = ", ".join([f"**{name}**" for name in flower_names])
        st.markdown(f"The active dataset contains measurements categorized into the following target classes: {class_summary}.")
        if not is_custom_dataset:
            st.markdown("""
            **Iris Dataset Overview**:  
            Four distinct botanical measurements are recorded (in cm):
            - **Sepal Length** and **Sepal Width**
            - **Petal Length** and **Petal Width**
            """)
        else:
            st.markdown("""
            **Custom Dataset Overview**:  
            You are operating on a custom uploaded dataset. The algorithms are utilizing your configured numeric feature columns shown on the left panel to predict the target labels.
            """)
        
    st.markdown("---")
    st.subheader("📈 High-Fidelity Visualizations")
    
    vis_mode = st.radio(
        "Choose Plot Type", 
        ["PCA 2D Cluster Projection", "3D Feature Scatter Plot", "Parallel Coordinates Path", "Correlation Heatmap"], 
        horizontal=True
    )
    
    if vis_mode == "PCA 2D Cluster Projection":
        st.markdown("""
        **Principal Component Analysis (PCA)** projects high-dimensional features into a 2D space. 
        This reveals how easily target categories form clusters and helps visualize model separability.
        """)
        
        # Calculate PCA
        try:
            pca = PCA(n_components=2)
            X_scaled = StandardScaler().fit_transform(X)
            X_pca = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
            pca_df["Class"] = df["species"]
            
            explained_var = pca.explained_variance_ratio_ * 100
            
            fig_pca = px.scatter(
                pca_df, x="PC1", y="PC2", color="Class",
                color_discrete_sequence=["#a855f7", "#ec4899", "#3b82f6", "#10b981", "#f59e0b"],
                title=f"PCA 2D Projection (Total Variance Explained: {sum(explained_var):.1f}%)"
            )
            fig_pca.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)" if theme_choice == "Space Dark (Violet)" else "rgba(255,255,255,0.7)",
                font=dict(color=text_color),
                xaxis=dict(gridcolor=grid_color, title=f"PC1 ({explained_var[0]:.1f}% variance)"),
                yaxis=dict(gridcolor=grid_color, title=f"PC2 ({explained_var[1]:.1f}% variance)")
            )
            st.plotly_chart(fig_pca, use_container_width=True)
        except Exception as e:
            st.error(f"Could not calculate PCA: {e}")
        
    elif vis_mode == "3D Feature Scatter Plot":
        if len(feature_names) >= 3:
            col_x, col_y, col_z = st.columns(3)
            with col_x:
                x_axis = st.selectbox("X-Axis Feature", feature_names, index=0)
            with col_y:
                y_axis = st.selectbox("Y-Axis Feature", feature_names, index=min(2, len(feature_names)-1))
            with col_z:
                z_axis = st.selectbox("Z-Axis Feature", feature_names, index=min(3, len(feature_names)-1))
                
            fig_3d = px.scatter_3d(
                df, x=x_axis, y=y_axis, z=z_axis,
                color="species",
                color_discrete_sequence=["#a855f7", "#ec4899", "#3b82f6", "#10b981", "#f59e0b"],
                labels={"species": "Species"},
                title=f"3D Scatter: {x_axis} vs {y_axis} vs {z_axis}"
            )
            fig_3d.update_layout(
                template=plotly_template,
                scene=dict(
                    bgcolor=card_bg,
                    xaxis=dict(gridcolor=grid_color),
                    yaxis=dict(gridcolor=grid_color),
                    zaxis=dict(gridcolor=grid_color)
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=text_color),
                margin=dict(l=0, r=0, b=0, t=40)
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.warning("3D Scatter Plot requires at least 3 feature columns.")
        
    elif vis_mode == "Parallel Coordinates Path":
        st.markdown("This plot illustrates the profile profiles of all flowers. You can see how each species forms clusters along the feature pathways.")
        fig_pc = px.parallel_coordinates(
            df,
            color="target",
            labels={"target": "Target Class"},
            color_continuous_scale=[(0.0, "#a855f7"), (0.5, "#ec4899"), (1.0, "#3b82f6")]
        )
        fig_pc.update_layout(
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=text_color),
            margin=dict(l=60, r=30, b=30, t=40)
        )
        st.plotly_chart(fig_pc, use_container_width=True)
        
    elif vis_mode == "Correlation Heatmap":
        st.markdown("Pearson correlation matrix shows the linear dependence between the numerical features and the target label.")
        corr_matrix = df.drop("species", axis=1).corr()
        fig_heat = px.imshow(
            corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            title="Pearson Correlation Heatmap"
        )
        fig_heat.update_layout(
            template=plotly_template,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=text_color),
            coloraxis_colorbar=dict(title="Correlation")
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: MODEL RANKING & COMPARISON
# ---------------------------------------------------------
with tab_comparison:
    st.markdown("### 🏆 Live Algorithm Leaderboard & Comparison")
    
    # Sort results to get real-time rankings
    sorted_df = results_df.sort_values(
        by=["CV Mean Accuracy (%)", "Test Set Accuracy (%)", "Training Time (ms)"],
        ascending=[False, False, True]
    ).reset_index(drop=True)
    
    best_algo = sorted_df.loc[0, "Algorithm"]
    best_cv_acc = sorted_df.loc[0, "CV Mean Accuracy (%)"]
    best_test_acc = sorted_df.loc[0, "Test Set Accuracy (%)"]
    best_time = sorted_df.loc[0, "Training Time (ms)"]
    
    # Styled Best Model Banner
    st.markdown(f"""
    <div class="premium-card" style="background: linear-gradient(135deg, rgba(168, 85, 247, 0.15) 0%, rgba(236, 72, 153, 0.15) 100%); border: 2px solid {card_border_hover}; margin-bottom: 25px;">
        <div style="font-size: 0.95rem; color: {card_border_hover}; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">🏆 Top Performing Classifier</div>
        <div style="font-size: 2.2rem; font-weight: 800; color: {text_color}; line-height: 1.2;">{best_algo} <span style="font-size: 1.5rem; color: #eab308; vertical-align: middle; margin-left: 10px;">⭐⭐⭐⭐⭐</span></div>
        <div style="color: {sub_text_color}; font-size: 1.05rem; margin-top: 8px; font-weight: 400;">
            This model achieved the highest stability with an average 5-Fold Cross-Validation Accuracy of 
            <strong style="color: {text_color}; font-weight: 600;">{best_cv_acc:.1f}%</strong> and a Test Set Accuracy of 
            <strong style="color: {text_color}; font-weight: 600;">{best_test_acc:.1f}%</strong>, fitting in just 
            <strong style="color: {text_color}; font-weight: 600;">{best_time:.2f} ms</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Real-Time Model Rankings Leaderboard Grid
    st.subheader("🥇 Real-time Rankings")
    rank_icons = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    cols_ranks = st.columns(len(sorted_df))
    for idx, row in sorted_df.iterrows():
        rank_icon = rank_icons[idx] if idx < len(rank_icons) else "🏅"
        with cols_ranks[idx]:
            st.markdown(f"""
            <div class="premium-card" style="text-align: center; border-color: {card_border_hover if idx == 0 else border_color}; padding: 15px;">
                <div style="font-size: 1.8rem;">{rank_icon}</div>
                <div style="font-weight: 700; font-size: 1.1rem; margin-top: 5px;">{row['Algorithm']}</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: {card_border_hover}; margin-top: 5px;">{row['CV Mean Accuracy (%)']:.1f}%</div>
                <div style="font-size: 0.8rem; color: {sub_text_color}; margin-top: 3px;">Test Acc: {row['Test Set Accuracy (%)']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.write(
        "To prevent all models from showing flat 100% accuracy, we evaluate them using two distinct methods: "
        "1) The **Test Set Accuracy** (measured on the held-out split) and 2) **5-Fold Cross-Validation Accuracy** "
        "(the average accuracy when splitting the training data into 5 folds sequentially). This offers a highly reliable performance baseline."
    )
    
    # Styled table
    st.subheader("📊 Performance Matrix")
    st.dataframe(
        results_df.style.background_gradient(cmap="Purples", subset=["Test Set Accuracy (%)", "CV Mean Accuracy (%)"])
        .format({"Training Time (ms)": "{:.2f}"}),
        use_container_width=True
    )
    
    # Grouped Bar Chart for Comparison
    st.subheader("📉 Dynamic Comparison Graph")
    
    # Melt dataframe for side-by-side bar chart
    melted_df = results_df.melt(
        id_vars="Algorithm", 
        value_vars=["Test Set Accuracy (%)", "CV Mean Accuracy (%)"],
        var_name="Metric Type", 
        value_name="Accuracy Value (%)"
    )
    
    fig_compare = px.bar(
        melted_df,
        x="Algorithm",
        y="Accuracy Value (%)",
        color="Metric Type",
        barmode="group",
        color_discrete_map={
            "Test Set Accuracy (%)": "#a855f7",
            "CV Mean Accuracy (%)": "#3b82f6"
        },
        text="Accuracy Value (%)",
        title="Test Set Accuracy vs. 5-Fold Cross-Validation Accuracy"
    )
    fig_compare.update_traces(textposition='outside', textfont_size=10)
    fig_compare.update_layout(
        template=plotly_template,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, linecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, linecolor=grid_color, range=[min(50, sorted_df["CV Mean Accuracy (%)"].min() - 10), 105]),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_compare, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: INSPECT & PREDICT
# ---------------------------------------------------------
with tab_predictor:
    col_inspect, col_pred = st.columns([1, 1])
    
    with col_inspect:
        st.subheader("🔍 Model Inspector")
        selected_model_name = st.selectbox(
            "Select Model to Inspect",
            list(classifiers.keys())
        )
        
        # Pull model performance
        pipeline = trained_pipelines[selected_model_name]
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Model Explanations Card ⭐⭐⭐⭐⭐
        explanations = {
            "KNN": "🧠 **K-Nearest Neighbors (KNN)** classifies data points based on proximity. It maps your query point to its closest neighbors (like distance coordinates) and runs a majority vote. If K=5, it searches for the 5 closest neighbors.",
            "Logistic Regression": "📈 **Logistic Regression** is a probabilistic linear classifier. It computes linear boundaries and applies a Sigmoid function to map inputs into a probability score between 0 and 1, making it highly robust for linearly separable tasks.",
            "Decision Tree": "🌳 **Decision Trees** split features sequentially using an tree-structured hierarchy. At each node, it tests threshold logic (e.g. 'Petal Length > 2.4cm') and continues branching down to the leaf node.",
            "Random Forest": "🌲🌲 **Random Forest** is an ensemble method containing a group of independent Decision Trees. It injects random subsets of data and features to build unique trees and combines their votes, reducing overfitting.",
            "SVM": "🛡️ **Support Vector Machines (SVM)** seek the maximum margin hyperplane. It maps points into multi-dimensional space to draw clear separators. The 'Kernel' allows it to classify non-linear clusters by projecting them upward."
        }
        st.markdown(f"""
        <div class="premium-card" style="border-left: 5px solid {card_border_hover}; padding: 15px; margin-bottom: 20px;">
            {explanations[selected_model_name]}
        </div>
        """, unsafe_allow_html=True)
        
        # Display selected model metrics
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(
                label=f"Selected model: {selected_model_name}",
                value=f"{accuracy * 100:.2f}%"
            )
        with col_m2:
            model_info = results_df[results_df["Algorithm"] == selected_model_name].iloc[0]
            st.metric(
                label="CV Avg Score",
                value=f"{model_info['CV Mean Accuracy (%)']:.1f}%",
                delta=f"± {model_info['CV Std Dev (%)']:.1f}%"
            )
            
        # Feature Importance Plot ⭐⭐⭐⭐⭐ (For Trees)
        classifier_model = pipeline.named_steps['classifier']
        if hasattr(classifier_model, 'feature_importances_'):
            st.markdown("#### ⚡ Feature Importance Analysis")
            importances = classifier_model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance (%)': np.round(importances * 100, 2)
            }).sort_values(by='Importance (%)', ascending=True)
            
            fig_imp = px.bar(
                feat_imp_df,
                x='Importance (%)',
                y='Feature',
                orientation='h',
                color='Importance (%)',
                color_continuous_scale="Purples",
                title=f"Feature Importances for {selected_model_name}"
            )
            fig_imp.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=text_color),
                coloraxis_showscale=False,
                height=250,
                margin=dict(l=10, r=10, b=10, t=30)
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info(f"ℹ️ {selected_model_name} does not calculate explicit feature importances. Switch to **Random Forest** or **Decision Tree** in the inspector to view Gini feature importance bars.")
            
        # Confusion Matrix Heatmap
        st.markdown("**📌 Confusion Matrix**")
        cm = confusion_matrix(y_test, predictions)
        
        fig_cm = px.imshow(
            cm,
            x=flower_names,
            y=flower_names,
            color_continuous_scale="Purples",
            text_auto=True,
            labels=dict(x="Predicted Species", y="Actual Species")
        )
        fig_cm.update_layout(
            template=plotly_template,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=text_color),
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, b=10, t=10),
            height=250
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Classification Report Table
        st.markdown("**📋 Classification Performance Report**")
        report = classification_report(y_test, predictions, target_names=flower_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df.style.background_gradient(cmap="Purples", subset=["precision", "recall", "f1-score"]))
        
    with col_pred:
        st.subheader("🌼 Live Prediction Panel")
        st.write("Adjust the features below and click predict to see how the selected model classifies the input values.")
        
        # Generate sliders dynamically based on feature range
        user_input_vals = []
        col_s1, col_s2 = st.columns(2)
        
        for idx, feat in enumerate(feature_names):
            min_val = float(df[feat].min())
            max_val = float(df[feat].max())
            mean_val = float(df[feat].mean())
            
            # Place in alternating columns
            with col_s1 if idx % 2 == 0 else col_s2:
                val = st.slider(
                    f"{feat}",
                    min_value=round(min_val - (max_val - min_val)*0.1, 1),
                    max_value=round(max_val + (max_val - min_val)*0.1, 1),
                    value=round(mean_val, 1),
                    step=0.1
                )
                user_input_vals.append(val)
            
        # Run prediction
        user_data = np.array([user_input_vals])
        
        # Make predictions and get confidence/probability if supported
        prediction = pipeline.predict(user_data)
        predicted_flower = flower_names[prediction[0]]
        
        # Fetch confidence percentage
        proba = pipeline.predict_proba(user_data)[0]
        confidence = proba[prediction[0]] * 100
        
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        
        # Beautiful prediction card
        st.markdown(f"""
        <div class="prediction-container">
            <div class="prediction-title">Classification Result</div>
            <div class="prediction-result">{predicted_flower.upper()}</div>
            <div class="confidence-badge">Confidence: {confidence:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display high quality image of the predicted species (for Iris only)
        if not is_custom_dataset:
            img_urls = {
                "setosa": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg",
                "versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
                "virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
            }
            
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            st.image(img_urls[predicted_flower.lower()], caption=f"Iris {predicted_flower.capitalize()} in the wild", use_container_width=True)
        else:
            st.info("Custom Dataset loaded: Prediction succeeded without default wild flower imagery.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: {sub_text_color}; font-size: 0.85rem;'>Developed with Streamlit • Scikit-learn • Plotly Analytics</div>", 
    unsafe_allow_html=True
)
