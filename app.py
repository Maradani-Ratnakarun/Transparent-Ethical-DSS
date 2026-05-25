import io, base64, torch, joblib, shap, lime, lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request
from train_model import ExplainableDSSModel

app = Flask(__name__)
plt.switch_backend('Agg')

# -------------------------------
# Load Saved Model Components
# -------------------------------
scaler = joblib.load("scaler.pkl")
background = joblib.load("background.pkl")
training_data = joblib.load("training_data.pkl")

model = ExplainableDSSModel(4)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# -------------------------------
# Prediction Function
# -------------------------------
def model_predict_probs(data):
    data_tensor = torch.tensor(data, dtype=torch.float32)
    with torch.no_grad():
        p = model(data_tensor).numpy()
    return np.hstack((1-p, p))

# -------------------------------
# Global Explainability Objects
# -------------------------------
shap_explainer = shap.KernelExplainer(
    lambda x: model_predict_probs(x)[:, 1], background
)

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data,
    feature_names=["Income", "CIBIL Score", "Loan Amount", "Loan Term"],
    class_names=['Rejected', 'Approved'],
    mode='classification'
)

# -------------------------------
# Home Route
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# -------------------------------
# Prediction Route
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        f = request.form
        user_inputs = {
            'income': f['income'],
            'cibil': f['cibil'],
            'loan': f['loan'],
            'term': f['term']
        }

        raw = np.array([[
            float(f['income']),
            float(f['cibil']),
            float(f['loan']),
            float(f['term'])
        ]])

        scaled = scaler.transform(raw)
        prob = model_predict_probs(scaled)[0][1]
        result = "Approved" if prob > 0.5 else "Rejected"

        # -------------------------------
        # Explainability
        # -------------------------------
        shap_vals = shap_explainer.shap_values(scaled, nsamples=100).flatten()
        lime_exp = lime_explainer.explain_instance(
            scaled[0], model_predict_probs, num_features=4
        )
        lime_vals = [val for _, val in lime_exp.as_list()]

        # -------------------------------
        # Human Readable Explanation
        # -------------------------------
        features = ["Annual Income", "CIBIL Score", "Loan Amount", "Loan Term"]
        top_pos_idx = np.argmax(shap_vals)
        top_neg_idx = np.argmin(shap_vals)
        t_pos = features[top_pos_idx]
        t_neg = features[top_neg_idx]

        if result == "Approved":
            explanation = (
                f"The system **Approved** this loan because your "
                f"**{t_pos}** is very strong, which outweighed the "
                f"risk from your {t_neg}."
            )
        else:
            explanation = (
                f"The system **Rejected** this loan primarily due to "
                f"your **{t_neg}**. Even though your {t_pos} was a "
                f"positive factor, it was not enough to offset the risk."
            )

        # -------------------------------
        # Visualization
        # -------------------------------
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        sns.barplot(x=shap_vals, y=features, ax=ax1, palette="RdBu")
        ax1.set_title("SHAP (Global Consistency)")

        sns.barplot(x=lime_vals, y=features, ax=ax2, palette="viridis")
        ax2.set_title("LIME (Local Reason)")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')

        return render_template(
            'index.html',
            result=result,
            prob=f"{prob*100:.1f}%",
            plot=plot_url,
            explanation=explanation,
            user_inputs=user_inputs
        )
    except Exception as e:
        return f"Error during inference: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)