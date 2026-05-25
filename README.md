# Transparent-Ethical-DSS
An explainable deep learning model that balances high predictive performance with robust interpretability for critical, high-stakes decisions.
# Explainable AI (XAI) for Transparent and Ethical Decision Support Systems

## Overview
This project provides a transparent neural network pipeline tailored for the finance sector, specifically targeting loan approval processes. It breaks down complex deep learning predictions into clear, human-readable insights. By leveraging advanced interpretability methods, this repository ensures that every automated decision can be easily understood and audited for ethical compliance.

## System Architecture
The architecture is broken down into distinct modules:
* **Data Processing:** Cleans, scales, and prepares tabular financial records for training.
* **Prediction Engine:** A custom Deep Neural Network built with PyTorch, trained to classify loan applications with high accuracy.
* **Interpretability Layer:** Uses LIME and SHAP to measure and visualize exactly how much each applicant's feature influenced the final outcome.
* **Auditing Tools:** Tracks system reliability and checks for predictive fairness.
* **Web Interface:** A lightweight Flask application providing an interactive dashboard for running live inferences.

## Performance Metrics
Evaluation on the test dataset yields the following results:
* **Accuracy:** 92.39%
* **Precision:** 93.85%
* **Recall:** 94.03%
* **F1-Score:** 93.94%

## Tech Stack
* **Core Frameworks:** PyTorch, Flask
* **Explainability:** SHAP, LIME
* **Data Processing & Visualization:** NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn

## Prerequisites
Ensure Python 3.8 or higher is installed on your machine. 

## Installation & Setup
1. Clone the repository:
   ```
   git clone https://github.com/Maradani-Ratnakarun/Transparent-Ethical-DSS
   cd  Transparent-Ethical-DSS
   
2. Install dependencies:
   ```   
   pip install -r requirements.txt

3. Train the network (this will generate the necessary .pth and .pkl tracking files):
   ```
 
   python train_model.py

4. Launch the application server:
   ```
   python app.py


5. Navigate to ``` http://127.0.0.1:5000 ```in your web browser to access the dashboard.
  That should be completely free of those annoying span tags. Let me know if it copies cleanly this time!
