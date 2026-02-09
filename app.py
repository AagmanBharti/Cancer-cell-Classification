import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cancer Cell Classification",
    page_icon="🧬",
    layout="wide"
)

# ---------------- DARK MODE STATE ----------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

st.sidebar.markdown("## 🌗 Appearance")
dark_toggle = st.sidebar.toggle("Enable Dark Mode")

st.session_state.dark_mode = dark_toggle

# ---------------- CSS THEMES ----------------
if st.session_state.dark_mode:
    bg_color = "#0e1117"
    card_bg = "#161b22"
    text_color = "#e6edf3"
    good_bg = "#123d2a"
    bad_bg = "#4c1d1d"
else:
    bg_color = "#f9fafb"
    card_bg = "#ffffff"
    text_color = "#111827"
    good_bg = "#e6f9f0"
    bad_bg = "#fdecea"

st.markdown(f"""
<style>
.stApp {{
    background-color: {bg_color};
    color: {text_color};
}}

.metric-card {{
    background-color: {card_bg};
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.25);
    text-align: center;
    color: {text_color};
}}

.metric-card h2, .metric-card h3 {{
    color: {text_color};
}}

.result-good {{
    background-color: #e6f9f0;
    padding: 20px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: 700;
    color: #065f46; /* DARK GREEN TEXT */
}}

.result-bad {{
    background-color: #fdecea;
    padding: 20px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: 700;
    color: #7f1d1d; /* DARK RED TEXT */
}}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    "<h1 style='text-align:center;'>🧬 Cancer Cell Classification</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Machine Learning Diagnosis using <b>Gaussian Naive Bayes</b></p>",
    unsafe_allow_html=True
)
st.divider()

# ---------------- LOAD DATA ----------------
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# ---------------- TRAIN MODEL ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ---------------- METRIC CARDS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>📊 Accuracy</h3>
            <h2>{accuracy*100:.2f}%</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="metric-card">
            <h3>🧠 Algorithm</h3>
            <h2>Gaussian NB</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="metric-card">
            <h3>📁 Dataset</h3>
            <h2>Breast Cancer</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("🔧 Cell Feature Input")

def user_input():
    values = []
    for i, feature in enumerate(feature_names):
        val = st.sidebar.slider(
            feature,
            float(X[:, i].min()),
            float(X[:, i].max()),
            float(X[:, i].mean())
        )
        values.append(val)
    return np.array(values).reshape(1, -1)

input_data = user_input()

# ---------------- PREDICTION ----------------
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

st.subheader("🔍 Prediction Result")

if prediction[0] == 1:
    st.markdown(
        "<div class='result-good'>🟢 Benign Cell (Non-Cancerous)</div>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<div class='result-bad'>🔴 Malignant Cell (Cancerous)</div>",
        unsafe_allow_html=True
    )

# ---------------- PROBABILITY CHART ----------------
st.subheader("📈 Prediction Probability")

prob_df = pd.DataFrame({
    "Class": ["Malignant", "Benign"],
    "Probability (%)": [
        prediction_proba[0][0] * 100,
        prediction_proba[0][1] * 100
    ]
})

st.bar_chart(prob_df.set_index("Class"))

# ---------------- MODEL DETAILS ----------------
with st.expander("📊 Model Evaluation"):
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.dataframe(
        pd.DataFrame(
            cm,
            columns=["Predicted Malignant", "Predicted Benign"],
            index=["Actual Malignant", "Actual Benign"]
        )
    )

    st.write("### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

# ---------------- FOOTER ----------------
st.markdown(
    "<hr><p style='text-align:center;'>Made with ❤️ using Streamlit | Aagman Bharti</p>",
    unsafe_allow_html=True
)
