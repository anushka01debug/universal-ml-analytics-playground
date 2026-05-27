from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report
)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Enable interactive plotting for VS Code
plt.ion()

# =========================================
# LOAD IRIS DATASET
# =========================================

iris = load_iris()

# Convert dataset into dataframe
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add target column
df['target'] = iris.target

print("=" * 60)
print("IRIS FLOWER DATASET")
print("=" * 60)

print("\nFirst 5 Rows of Dataset:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nFlower Categories:")
print(iris.target_names)

# =========================================
# DATA VISUALIZATION
# =========================================

print("\nGenerating Visualizations...")

# -----------------------------------------
# HISTOGRAMS
# -----------------------------------------

df.hist(figsize=(10, 8))

plt.suptitle("Feature Distribution Histograms")

plt.savefig("histograms.png")

plt.show(block=False)
plt.pause(3)

# -----------------------------------------
# CORRELATION HEATMAP
# -----------------------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.savefig("heatmap.png")

plt.show(block=False)
plt.pause(3)

# -----------------------------------------
# PAIRPLOT
# -----------------------------------------

pair_plot = sns.pairplot(
    df,
    hue='target',
    palette='husl'
)

pair_plot.fig.suptitle(
    "Feature Relationships",
    y=1.02
)

plt.savefig("pairplot.png")

plt.show(block=False)
plt.pause(3)

# =========================================
# FEATURES AND LABELS
# =========================================

X = df.drop('target', axis=1)

y = df['target']

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n" + "=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

print(f"\nTraining Data Shape : {X_train.shape}")

print(f"Testing Data Shape  : {X_test.shape}")

# =========================================
# FEATURE SCALING
# =========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("\nFeature Scaling Applied Successfully!")

# =========================================
# FIND BEST K VALUE
# =========================================

scores = []

k_values = range(1, 21)

for k in k_values:

    temp_model = KNeighborsClassifier(
        n_neighbors=k
    )

    temp_model.fit(X_train, y_train)

    temp_pred = temp_model.predict(X_test)

    temp_accuracy = accuracy_score(
        y_test,
        temp_pred
    )

    scores.append(temp_accuracy)

best_k = scores.index(max(scores)) + 1

print(f"\nBest K Value Found : {best_k}")

# -----------------------------------------
# K VALUE VS ACCURACY GRAPH
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    scores,
    marker='o'
)

plt.title("K Value vs Accuracy")

plt.xlabel("K Value")

plt.ylabel("Accuracy")

plt.grid(True)

plt.savefig("k_vs_accuracy.png")

plt.show(block=False)
plt.pause(3)

# =========================================
# CREATE KNN MODEL
# =========================================

model = KNeighborsClassifier(
    n_neighbors=best_k
)

# Train model
model.fit(X_train, y_train)

print("\nKNN Model Trained Successfully!")

# =========================================
# MAKE PREDICTIONS
# =========================================

predictions = model.predict(X_test)

# =========================================
# MODEL EVALUATION
# =========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

cm = confusion_matrix(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions,
    average='weighted'
)

# =========================================
# HUMAN READABLE OUTPUT
# =========================================

flower_names = iris.target_names

print("\n" + "=" * 60)
print("FLOWER PREDICTION RESULTS")
print("=" * 60)

for i in range(len(predictions)):

    predicted_flower = flower_names[
        predictions[i]
    ]

    actual_flower = flower_names[
        y_test.iloc[i]
    ]

    print(f"\nFlower {i+1}")

    print(
        f"Predicted Flower : {predicted_flower}"
    )

    print(
        f"Actual Flower    : {actual_flower}"
    )

    if predictions[i] == y_test.iloc[i]:

        print(
            "Result            : Correct Prediction ✅"
        )

    else:

        print(
            "Result            : Wrong Prediction ❌"
        )

# =========================================
# PERFORMANCE SUMMARY
# =========================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 60)

print(
    f"\nAccuracy Score : {accuracy * 100:.2f}%"
)

print(
    "Meaning        : Percentage of correct predictions"
)

print(f"\nF1 Score       : {f1:.2f}")

print(
    "Meaning        : Overall balanced model performance"
)

# -----------------------------------------
# CLASSIFICATION REPORT
# -----------------------------------------

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions
    )
)

# -----------------------------------------
# CONFUSION MATRIX HEATMAP
# -----------------------------------------

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=flower_names,
    yticklabels=flower_names
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix Heatmap")

plt.savefig("confusion_matrix.png")

plt.show(block=False)
plt.pause(3)

print("\nConfusion Matrix:\n")

print(cm)

print("""
Confusion Matrix Meaning:
- Diagonal values = Correct predictions
- Other values    = Wrong predictions

Higher diagonal values mean better model performance.
""")

print("=" * 60)

print("PROJECT COMPLETED SUCCESSFULLY!")

print("=" * 60)

# =========================================
# USER INPUT PREDICTION
# =========================================

print("\n" + "=" * 60)

print("PREDICT YOUR OWN FLOWER")

print("=" * 60)

# Take user input

sepal_length = float(
    input("Enter Sepal Length (cm): ")
)

sepal_width = float(
    input("Enter Sepal Width (cm): ")
)

petal_length = float(
    input("Enter Petal Length (cm): ")
)

petal_width = float(
    input("Enter Petal Width (cm): ")
)

# Create input data

user_data = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

# Apply scaling

user_data = scaler.transform(user_data)

# Predict flower

prediction = model.predict(user_data)

# Convert prediction into flower name

flower_name = flower_names[prediction[0]]

print(
    "\nPredicted Flower Type:",
    flower_name
)

# Keep plots open

print("\nClose graph windows to end program.")

plt.ioff()
plt.show()