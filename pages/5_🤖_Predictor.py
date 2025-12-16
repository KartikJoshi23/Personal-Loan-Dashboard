import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve

st.set_page_config(page_title="Predictor", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .prediction-card {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .accept {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
    }
    .reject {
        background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
        color: white;
    }
    .prob-text {
        font-size: 3rem;
        font-weight: 700;
        margin: 15px 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border-left: 5px solid #667eea;
        padding: 20px;
        border-radius: 0 15px 15px 0;
        margin: 15px 0;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        paths = ["UniversalBank.xlsx", "data/UniversalBank.xlsx",
                "UniversalBank with description.xls", "data/UniversalBank with description.xls"]
        for path in paths:
            try:
                df = pd.read_excel(path, sheet_name="Data")
                df.columns = df.columns.str.strip().str.replace(' ', '_')
                df['Experience'] = df['Experience'].abs() if 'Experience' in df.columns else 0
                return df
            except:
                continue
    except:
        pass
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame({
        'Age': np.random.randint(23, 67, n), 'Experience': np.random.randint(0, 43, n),
        'Income': np.clip(np.random.exponential(60, n), 8, 224).astype(int),
        'Family': np.random.choice([1, 2, 3, 4], n),
        'CCAvg': np.round(np.random.exponential(1.9, n), 1),
        'Education': np.random.choice([1, 2, 3], n),
        'Mortgage': np.where(np.random.random(n) > 0.5, np.random.exponential(100, n).astype(int), 0),
        'Securities_Account': np.random.choice([0, 1], n, p=[0.90, 0.10]),
        'CD_Account': np.random.choice([0, 1], n, p=[0.94, 0.06]),
        'Online': np.random.choice([0, 1], n), 'CreditCard': np.random.choice([0, 1], n)
    })
    prob = 0.02 + 0.15*(df['Income']>100) + 0.25*(df['CD_Account']==1)
    df['Personal_Loan'] = (np.random.random(n) < prob).astype(int)
    return df

df = load_data()

# Features
features = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education', 
            'Mortgage', 'Securities_Account', 'CD_Account', 'Online', 'CreditCard']
features = [f for f in features if f in df.columns]

X = df[features]
y = df['Personal_Loan']

# Train models
@st.cache_resource
def train_models():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_s, y_train)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    
    return {'lr': lr, 'rf': rf, 'scaler': scaler, 'X_test': X_test_s, 'y_test': y_test, 'features': features}

models = train_models()

st.title("🤖 Personal Loan Acceptance Predictor")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.radio("Select Model", ['Logistic Regression', 'Random Forest'])
model = models['lr'] if model_choice == 'Logistic Regression' else models['rf']

# Metrics
y_pred = model.predict(models['X_test'])
y_prob = model.predict_proba(models['X_test'])[:, 1]

acc = accuracy_score(models['y_test'], y_pred)
prec = precision_score(models['y_test'], y_pred)
rec = recall_score(models['y_test'], y_pred)
auc = roc_auc_score(models['y_test'], y_prob)

st.subheader(f"📊 Model Performance: {model_choice}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{acc:.1%}")
col2.metric("Precision", f"{prec:.1%}")
col3.metric("Recall", f"{rec:.1%}")
col4.metric("ROC-AUC", f"{auc:.3f}")

st.markdown(f"""
<div class="insight-box">
    <strong>📌 Model Insight:</strong> {model_choice} achieves <strong>{auc:.1%} ROC-AUC</strong> 
    ({'excellent' if auc > 0.9 else 'good'} discrimination). Precision {prec:.1%} = of predicted accepters, 
    {prec:.1%} actually accept. Recall {rec:.1%} = we identify {rec:.1%} of all actual accepters.
</div>
""", unsafe_allow_html=True)

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(models['y_test'], y_pred)
    fig = go.Figure(data=go.Heatmap(z=cm, x=['Pred: No', 'Pred: Yes'], y=['Actual: No', 'Actual: Yes'],
                                    colorscale='Blues', text=cm, texttemplate='%{text}', textfont={'size': 18}))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(models['y_test'], y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC={auc:.3f})', line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random', line=dict(dash='dash', color='gray')))
    fig.update_layout(height=300, xaxis_title='FPR', yaxis_title='TPR')
    st.plotly_chart(fig, use_container_width=True)

# Feature Importance
st.markdown("---")
st.subheader("🔍 Feature Importance")

if model_choice == 'Logistic Regression':
    coef = model.coef_[0]
    imp_df = pd.DataFrame({'Feature': models['features'], 'Coef': coef}).sort_values('Coef')
    fig = px.bar(imp_df, x='Coef', y='Feature', orientation='h', color='Coef', color_continuous_scale='RdBu_r',
                 title='Logistic Regression Coefficients')
else:
    imp = model.feature_importances_
    imp_df = pd.DataFrame({'Feature': models['features'], 'Importance': imp}).sort_values('Importance')
    fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', color='Importance', 
                 color_continuous_scale='Greens', title='Random Forest Feature Importance')

fig.update_layout(height=400, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# Prediction Form
st.markdown("---")
st.subheader("🎯 Make a Prediction")
st.markdown("Enter customer details below to predict loan acceptance probability:")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 100, 35)
    experience = st.number_input("Experience (years)", 0, 50, 10)
    income = st.number_input("Income ($K/year)", 0, 500, 60)
    family = st.selectbox("Family Size", [1, 2, 3, 4], index=1)

with col2:
    ccavg = st.number_input("CC Avg Spending ($K/month)", 0.0, 15.0, 1.5, 0.1)
    education = st.selectbox("Education", [1, 2, 3], index=1, 
                             format_func=lambda x: {1: 'Undergraduate', 2: 'Graduate', 3: 'Advanced'}[x])
    mortgage = st.number_input("Mortgage ($K)", 0, 1000, 0)

with col3:
    securities = st.selectbox("Securities Account", [0, 1], format_func=lambda x: 'Yes' if x else 'No')
    cd_account = st.selectbox("CD Account", [0, 1], format_func=lambda x: 'Yes' if x else 'No')
    online = st.selectbox("Online Banking", [0, 1], format_func=lambda x: 'Yes' if x else 'No')
    creditcard = st.selectbox("Credit Card", [0, 1], format_func=lambda x: 'Yes' if x else 'No')

if st.button("🔮 Predict Loan Acceptance", type="primary", use_container_width=True):
    # Prepare input
    input_data = np.array([[age, experience, income, family, ccavg, education, mortgage, 
                           securities, cd_account, online, creditcard]])
    input_scaled = models['scaler'].transform(input_data)
    
    # Predict
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    
    st.markdown("---")
    
    # Result display
    col1, col2 = st.columns(2)
    
    with col1:
        if pred == 1:
            st.markdown(f"""
            <div class="prediction-card accept">
                <h2>✅ LIKELY TO ACCEPT</h2>
                <div class="prob-text">{prob:.1%}</div>
                <p>Probability of Acceptance</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card reject">
                <h2>❌ UNLIKELY TO ACCEPT</h2>
                <div class="prob-text">{prob:.1%}</div>
                <p>Probability of Acceptance</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#28a745" if prob > 0.5 else "#dc3545"},
                'steps': [
                    {'range': [0, 30], 'color': "#ffebee"},
                    {'range': [30, 70], 'color': "#fff3e0"},
                    {'range': [70, 100], 'color': "#e8f5e9"}
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}
            },
            title={'text': "Acceptance Probability (%)"}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Key factors
    st.subheader("📋 Key Factors for This Prediction")
    
    factors = []
    if income > 100: factors.append("✅ High income - strongly positive")
    elif income < 50: factors.append("⚠️ Lower income - may reduce probability")
    if cd_account == 1: factors.append("✅ Has CD Account - strongly positive")
    if education == 3: factors.append("✅ Advanced education - positive")
    if ccavg > 3: factors.append("✅ High CC spending - positive")
    if securities == 1: factors.append("ℹ️ Has Securities - moderate positive")
    
    if factors:
        for f in factors: st.write(f)
    else:
        st.write("ℹ️ No strongly influential factors identified")

# Notes
with st.expander("📝 Model Notes"):
    st.markdown("""
    - **Train/Test:** 80/20 split with stratification
    - **Preprocessing:** StandardScaler normalization
    - **Logistic Regression:** Interpretable baseline
    - **Random Forest:** 100 trees for higher accuracy
    
    **Limitations:**
    - Class imbalance (~10% acceptance)
    - ZIP code not used (too many categories)
    - Retrain periodically as patterns change
    """)
