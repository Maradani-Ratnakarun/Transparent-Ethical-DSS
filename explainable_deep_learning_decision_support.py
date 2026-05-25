
# explainable_deep_learning_decision_support.py
# Implementation of an Explainable Deep Learning Model for Transparent and Ethical Decision Support Systems
# Domain: Deep Learning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1. Synthetic Decision Support Dataset
# -----------------------------
def generate_decision_data(samples=4000):
    np.random.seed(42)

    age = np.random.randint(18, 70, samples)
    income = np.random.normal(60000, 20000, samples)
    credit_score = np.random.normal(650, 60, samples)
    risk_factor = np.random.uniform(0, 1, samples)

    score = (
        0.3 * income +
        0.4 * credit_score +
        200 * age -
        50000 * risk_factor +
        np.random.normal(0, 8000, samples)
    )

    label = (score > np.percentile(score, 55)).astype(int)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "risk_factor": risk_factor,
        "decision": label
    })

    return df


# -----------------------------
# 2. Dataset Creation & Saving
# -----------------------------
def create_and_save_dataset():
    df = generate_decision_data()
    df.to_csv("xai_decision_data.csv", index=False)
    print("Synthetic decision-support dataset saved.")


# -----------------------------
# 3. Data Loading & Inspection
# -----------------------------
def load_and_inspect_data():
    df = pd.read_csv("xai_decision_data.csv")
    print("\nDataset Head:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nClass Distribution:")
    print(df["decision"].value_counts())
    return df


# -----------------------------
# 4. Preprocessing
# -----------------------------
def preprocess_data(df):
    X = df.drop("decision", axis=1)
    y = df["decision"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved as scaler.pkl")

    return X_scaled, y, X.columns


# -----------------------------
# 5. Explainable Deep Learning Model
# -----------------------------
def build_model(input_dim):
    model = Sequential([
        Dense(64, activation="relu", input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# 6. Training
# -----------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = build_model(X.shape[1])

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    model.save("explainable_dl_model.h5")
    print("Deep learning model saved")

    return model, history, X_test, y_test


# -----------------------------
# 7. Evaluation
# -----------------------------
def evaluate_model(model, X_test, y_test):
    preds = (model.predict(X_test) > 0.5).astype(int)

    print("\nAccuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()


# -----------------------------
# 8. Explainability (Permutation Importance)
# -----------------------------
def explain_model(model, X_test, y_test, feature_names):
    baseline = accuracy_score(y_test, (model.predict(X_test) > 0.5).astype(int))

    importances = []

    for i in range(X_test.shape[1]):
        X_permuted = X_test.copy()
        np.random.shuffle(X_permuted[:, i])
        score = accuracy_score(
            y_test,
            (model.predict(X_permuted) > 0.5).astype(int)
        )
        importances.append(baseline - score)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print("\nFeature Importance (Explainability):")
    print(importance_df)

    plt.figure(figsize=(6, 4))
    sns.barplot(data=importance_df, x="importance", y="feature")
    plt.title("Explainable Deep Learning Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()


# -----------------------------
# 9. Transparent Prediction
# -----------------------------
def explainable_predict(input_features):
    scaler = joblib.load("scaler.pkl")
    model = tf.keras.models.load_model("explainable_dl_model.h5")

    input_features = np.array(input_features).reshape(1, -1)
    input_scaled = scaler.transform(input_features)
    prediction = (model.predict(input_scaled) > 0.5).astype(int)

    return prediction[0][0]


# -----------------------------
# 10. Main Execution
# -----------------------------
def main():
    if not os.path.exists("xai_decision_data.csv"):
        create_and_save_dataset()

    df = load_and_inspect_data()
    X, y, feature_names = preprocess_data(df)

    model, history, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)
    explain_model(model, X_test, y_test, feature_names)

    sample = X_test[0]
    decision = explainable_predict(sample)
    print("\nExplainable Prediction Output:", decision)


if __name__ == "__main__":
    main()
