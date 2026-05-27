🚀 Universal ML Analytics Playground

An advanced interactive Machine Learning analytics platform built using Streamlit, Scikit-learn, and Plotly.
This project allows users to upload custom datasets, visualize patterns using PCA & 3D graphs, compare multiple ML algorithms, and generate real-time predictions through an elegant dashboard interface.

✨ Features
📂 Upload Custom CSV Datasets
🌸 Default Iris Flower Classification Support
📊 Interactive Data Visualization Dashboard
📈 PCA 2D Cluster Projection
🌐 3D Scatter Plot Visualization
🔥 Correlation Heatmap Analysis
🧠 Multiple Machine Learning Algorithms
⚡ Real-Time Hyperparameter Tuning
🏆 Live Model Ranking System
📉 Cross Validation Performance Analysis
🤖 Real-Time Prediction Panel
🌲 Feature Importance Visualization
📌 Confusion Matrix Heatmaps
📋 Classification Reports
🎨 Dynamic Theme Switching (Dark / Light)
🚀 Fully Deployable Streamlit Web Application

🏗️ Project Structure
Universal-ML-Analytics-Playground/

│── app.py                    # Main Streamlit Application
│── main.py                   # Additional execution file (optional)
│── requirements.txt          # Project dependencies
│── README.md                 # Project documentation

│── screenshot/               # Application screenshots
│   ├── pca.png
│   ├── 3D scatter plot.png
│   ├── correlation heatmap.png
│   ├── performance matrix.png
│   ├── inspect and predict.png
│   ├── Model ranking and comparision.png
│
└── .gitignore

🛠️ Tech Stack
Frontend / UI
--Streamlit — Interactive Web App Framework
--Plotly — Advanced Interactive Visualizations
--HTML/CSS — Custom UI Styling

Machine Learning
--Scikit-learn — ML Algorithms & Pipelines
--PCA — Dimensionality Reduction
--Cross Validation — Model Evaluation

Data Processing
--Pandas — Data Manipulation
--NumPy — Numerical Computation
📦 Installation
Prerequisites
--Make sure you have installed:
---Python 3.9 or higher
---pip

⚙️ Setup Instructions
Clone Repository
git clone https://github.com/your-username/Universal-ML-Analytics-Playground.git
Move Into Project Folder
cd Universal-ML-Analytics-Playground
Install Dependencies
pip install -r requirements.txt

🚀 Running the Application
Start Streamlit App
streamlit run app.py
Open in Browser
http://localhost:8501

🧠 Supported Machine Learning Algorithms
The platform currently supports:
K-Nearest Neighbors (KNN)
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
Support Vector Machine (SVM)

📊 Visualization Modules
PCA 2D Projection--Projects high-dimensional data into 2D space for cluster visualization.
3D Scatter Plot--Interactive 3D feature exploration.
Parallel Coordinates Plot--Analyze feature patterns across classes.
Correlation Heatmap--Identify linear relationships between features.

🏆 Model Evaluation Metrics
The application compares models using:
Test Accuracy
Cross Validation Accuracy
Standard Deviation
Training Time
Confusion Matrix
Precision
Recall
F1-Score

🤖 Live Prediction System
Users can:
Adjust feature sliders dynamically
Generate real-time predictions
View prediction confidence scores
Analyze model behavior instantly

🎨 UI Features
🌙 Space Dark Theme
☀️ Orchid Light Theme
Responsive Dashboard Layout
Animated Metric Cards
Gradient UI Components
Interactive Charts

📂 Custom Dataset Support
Users can upload their own CSV datasets and:
Select target columns
Choose feature columns
Train models dynamically
Generate predictions instantly

Screenshots
# 📸 Screenshots

## PCA Visualization
![PCA Visualization](screenshot/pca.png)

---

## 3D Scatter Plot
![3D Scatter Plot](screenshot/3D%20scatter%20plot.png)

---

## Correlation Heatmap
![Correlation Heatmap](screenshot/correlation%20heatmap.png)

---

## Model Ranking Dashboard
![Model Ranking](screenshot/Model%20ranking%20and%20comparision.png)

---

## Performance Matrix
![Performance Matrix](screenshot/performance%20matrix.png)

---

## Inspect & Predict Panel
![Inspect Predict](screenshot/inspect%20and%20predict.png)

🌐 Deployment

This project can be deployed easily on:
Streamlit Community Cloud
Render
Hugging Face Spaces
Railway

☁️ Streamlit Deployment
Deploy on Streamlit Cloud
Push project to GitHub
Open Streamlit Cloud
Select repository
Choose:
app.py
Click Deploy

📝 requirements.txt
streamlit
pandas
numpy
plotly
scikit-learn

🐛 Troubleshooting
Module Not Found Error
--Install dependencies again:pip install -r requirements.txt
Streamlit Not Starting
--Run:streamlit run app.py

Port Already in Use
--Kill existing Streamlit process or restart terminal.

🤝 Contributing
Contributions are welcome.
To contribute:
Fork the repository
Create a new branch
Commit changes
Push updates
Open Pull Request

📜 License

This project is created for educational and academic purposes.

👩‍💻 Developed By
Anushka Bakshi
Built using ❤️ with Streamlit, Scikit-learn & Plotly.
