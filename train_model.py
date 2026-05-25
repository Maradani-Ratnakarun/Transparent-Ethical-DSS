import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# -------------------------------
# Neural Network Architecture
# -------------------------------
class ExplainableDSSModel(nn.Module):

    def __init__(self, input_dim):
        super(ExplainableDSSModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------
# Model Training Function
# -------------------------------
def train():
    try:
        df = pd.read_csv("loan_approval_dataset.csv")
    except FileNotFoundError:
        print("Dataset not found.")
        return

    df.columns = df.columns.str.strip()
    y = (df['loan_status'].str.strip() == 'Approved').astype(float).values
    features = ['income_annum', 'cibil_score', 'loan_amount', 'loan_term']
    X = df[features].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = ExplainableDSSModel(len(features))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Training Started")
    for epoch in range(100):
        model.train()
        inputs = torch.tensor(X_train_scaled, dtype=torch.float32)
        labels = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    # -------------------------------
    # Evaluation
    # -------------------------------
    model.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_prob = model(test_inputs).numpy()
        y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nModel Performance")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)

    torch.save(model.state_dict(), "model.pth")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(X_train_scaled[:100], "background.pkl")
    joblib.dump(X_train_scaled, "training_data.pkl")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Rejected', 'Approved'])
    
    plt.figure(figsize=(6,6))
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    train()